"""
MultiDiffusion-style tiled denoising for SDXL ControlNet img2img.

Lets the existing Pixagram pipeline generate at arbitrary resolution (up to 4K)
by splitting the latent into overlapping tiles, denoising each independently
at every diffusion step, and weighted-averaging the noise predictions across
overlaps. This keeps tiles globally coherent — every pixel is influenced by
every neighbouring tile at every step.

Reference: Bar-Tal et al., "MultiDiffusion: Fusing Diffusion Paths for
Controlled Image Generation" (ICML 2023).

Why this works with our LCM + LoRA + ControlNet stack:
  - The LoRA is fused into UNet weights, so per-tile UNet calls already
    carry the pixel-art style.
  - LCM only needs ~10 steps, so per-step tile overhead stays manageable.
  - ControlNet conditioning is tiled in lockstep with the latent, so depth
    cues stay aligned.

We borrow the pipeline's own helpers (encode_prompt, prepare_latents,
prepare_control_image, get_timesteps, _get_add_time_ids) to keep behavior
identical to the reference call path — we only replace the inner denoising
loop.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image


# ============================================================
# Tile grid
# ============================================================

def _compute_tile_positions(
    total: int,
    tile: int,
    overlap: int,
) -> List[int]:
    """
    Return starting offsets for tiles that cover [0, total) with the given
    tile size and overlap. The last tile is shifted left so it stays in
    bounds; this means the last overlap may be larger than `overlap`.
    """
    if total <= tile:
        return [0]

    stride = tile - overlap
    positions: List[int] = []
    pos = 0
    while pos + tile < total:
        positions.append(pos)
        pos += stride
    # Final tile flush against the right edge
    positions.append(total - tile)
    # Dedupe (can happen for exact-fit grids)
    out: List[int] = []
    for p in positions:
        if not out or p != out[-1]:
            out.append(p)
    return out


def _make_tile_weight(tile_h: int, tile_w: int, device, dtype) -> torch.Tensor:
    """
    2D cosine-bell weight mask for blending tiles. Edges are ~0, center is 1.
    Shape: (1, 1, tile_h, tile_w).
    """
    yy = torch.arange(tile_h, device=device, dtype=torch.float32)
    xx = torch.arange(tile_w, device=device, dtype=torch.float32)
    # Cosine taper: peaks in the middle, falls to ~0 at the edges
    wy = torch.sin(math.pi * (yy + 0.5) / tile_h)
    wx = torch.sin(math.pi * (xx + 0.5) / tile_w)
    w = wy[:, None] * wx[None, :]
    # Floor so even edge pixels get non-zero weight (avoids div-by-zero
    # at the literal corners; the overlap region dominates anyway).
    w = w.clamp(min=1e-3)
    return w.to(dtype=dtype).unsqueeze(0).unsqueeze(0)


# ============================================================
# Main entry point
# ============================================================

@torch.no_grad()
def tiled_controlnet_img2img(
    pipe,
    *,
    prompt: str,
    negative_prompt: str,
    image: Image.Image,
    control_images: List[Image.Image],
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    controlnet_conditioning_scales: List[float],
    generator: torch.Generator,
    clip_skip: Optional[int] = None,
    tile_size_px: int = 768,
    tile_overlap_px: int = 192,
    callback=None,
) -> Image.Image:
    """
    Tiled MultiDiffusion denoising over an SDXL ControlNet img2img pipeline.

    Supports MultiControlNet: `control_images` and
    `controlnet_conditioning_scales` are parallel lists (e.g. [depth, canny]
    and [depth_scale, canny_scale]) whose order matches the order the
    ControlNets were registered on the pipeline.

    Args:
        pipe: StableDiffusionXLControlNetImg2ImgPipeline whose `controlnet`
              is a MultiControlNetModel (or a single ControlNetModel, in
              which case pass single-element lists).
        prompt / negative_prompt: Text conditioning, identical for every tile.
        image: Full-resolution input PIL image (RGB), sized to (width, height).
        control_images: List of full-resolution control PIL images.
        width, height: Target generation dimensions (multiples of 8).
        num_inference_steps, guidance_scale, strength: Standard knobs.
        controlnet_conditioning_scales: Per-ControlNet conditioning scales.
        generator: torch.Generator for reproducibility.
        clip_skip: Optional clip skip.
        tile_size_px: Pixel tile size (default 768). Converted to latent (÷8).
        tile_overlap_px: Pixel overlap between adjacent tiles.
        callback: Optional fn(step_index, total_steps) for progress updates.

    Returns:
        Decoded PIL image at (width, height).
    """
    device = pipe._execution_device
    dtype = pipe.unet.dtype

    # --- 1. Encode prompt (once) --------------------------------------------
    do_cfg = guidance_scale > 1.0
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        clip_skip=clip_skip,
    )

    # --- 2. Preprocess image and control_images to tensors -----------------
    image_t = pipe.image_processor.preprocess(image, height=height, width=width).to(
        dtype=torch.float32
    )

    # ControlNet dtype: MultiControlNetModel has no .dtype, so fall back to
    # its first sub-net. Works for a single ControlNetModel too.
    cn_dtype = getattr(pipe.controlnet, "dtype", None)
    if cn_dtype is None:
        cn_dtype = pipe.controlnet.nets[0].dtype

    control_ts = [
        pipe.prepare_control_image(
            image=ci,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device=device,
            dtype=cn_dtype,
            do_classifier_free_guidance=do_cfg,
            guess_mode=False,
        )
        for ci in control_images
    ]
    # each control_t shape: (2, 3, H, W) if CFG else (1, 3, H, W)

    # --- 3. Timesteps --------------------------------------------------------
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps, strength, device
    )
    latent_timestep = timesteps[:1].repeat(1)

    # --- 4. Prepare full-image latents (img2img: noisy version of input) ----
    latents = pipe.prepare_latents(
        image_t,
        latent_timestep,
        batch_size=1,
        num_images_per_prompt=1,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        add_noise=True,
    )
    # latents shape: (1, 4, H/8, W/8)

    # --- 5. Added time ids (SDXL conditioning) ------------------------------
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids, add_neg_time_ids = pipe._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        negative_original_size=(height, width),
        negative_crops_coords_top_left=(0, 0),
        negative_target_size=(height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(1, 1)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # --- 6. Build tile grid (in latent units) -------------------------------
    # SDXL VAE downsamples by 8x.
    tile_lat = tile_size_px // 8
    overlap_lat = tile_overlap_px // 8

    lat_h = latents.shape[2]
    lat_w = latents.shape[3]

    # If the whole image already fits in one tile, just run the standard path.
    if lat_h <= tile_lat and lat_w <= tile_lat:
        tile_lat_h = lat_h
        tile_lat_w = lat_w
        y_positions = [0]
        x_positions = [0]
    else:
        tile_lat_h = min(tile_lat, lat_h)
        tile_lat_w = min(tile_lat, lat_w)
        y_positions = _compute_tile_positions(lat_h, tile_lat_h, overlap_lat)
        x_positions = _compute_tile_positions(lat_w, tile_lat_w, overlap_lat)

    num_tiles = len(y_positions) * len(x_positions)
    print(
        f"[Tiled] Latent grid {lat_w}x{lat_h}, tile {tile_lat_w}x{tile_lat_h} "
        f"(px {tile_lat_w*8}x{tile_lat_h*8}), "
        f"overlap {overlap_lat} lat ({overlap_lat*8}px), "
        f"{len(x_positions)}x{len(y_positions)} = {num_tiles} tiles"
    )

    # Reusable Gaussian-ish weight mask for blending overlaps
    tile_weight = _make_tile_weight(tile_lat_h, tile_lat_w, device, latents.dtype)

    # --- 7. ControlNet keep schedule (matches reference pipeline) -----------
    controlnet_keep = []
    control_guidance_start = 0.0
    control_guidance_end = 1.0
    for i in range(len(timesteps)):
        keeps = 1.0 - float(
            i / len(timesteps) < control_guidance_start
            or (i + 1) / len(timesteps) > control_guidance_end
        )
        controlnet_keep.append(keeps)

    # --- 8. Denoising loop --------------------------------------------------
    for i, t in enumerate(timesteps):
        # Per-step accumulators on the full latent canvas
        noise_pred_full = torch.zeros_like(latents)
        weight_sum = torch.zeros(
            (1, 1, lat_h, lat_w), device=device, dtype=latents.dtype
        )

        for ty in y_positions:
            for tx in x_positions:
                y1, y2 = ty, ty + tile_lat_h
                x1, x2 = tx, tx + tile_lat_w

                # Slice latent for this tile
                lat_tile = latents[:, :, y1:y2, x1:x2]

                # Slice each control image — control_ts are pixel space (8x).
                py1, py2 = y1 * 8, y2 * 8
                px1, px2 = x1 * 8, x2 * 8
                ctrl_tiles = [ct[:, :, py1:py2, px1:px2] for ct in control_ts]

                # Expand for CFG and scale
                latent_model_input = (
                    torch.cat([lat_tile] * 2) if do_cfg else lat_tile
                )
                latent_model_input = pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Per-ControlNet conditioning scales, gated by the keep schedule.
                cond_scales = [
                    s * controlnet_keep[i] for s in controlnet_conditioning_scales
                ]

                # ControlNet (MultiControlNetModel: lists in, summed residuals out)
                down_res, mid_res = pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=ctrl_tiles,
                    conditioning_scale=cond_scales,
                    guess_mode=False,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                # UNet
                noise_pred_tile = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                if do_cfg:
                    uncond, text = noise_pred_tile.chunk(2)
                    noise_pred_tile = uncond + guidance_scale * (text - uncond)

                # Weighted accumulate
                noise_pred_full[:, :, y1:y2, x1:x2] += noise_pred_tile * tile_weight
                weight_sum[:, :, y1:y2, x1:x2] += tile_weight

                # Free per-tile activations
                del down_res, mid_res, noise_pred_tile, latent_model_input, ctrl_tiles

        # Normalize accumulated noise prediction
        noise_pred_full = noise_pred_full / weight_sum.clamp(min=1e-8)

        # Step scheduler once on the merged prediction
        latents = pipe.scheduler.step(
            noise_pred_full, t, latents, return_dict=False
        )[0]

        del noise_pred_full, weight_sum
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if callback is not None:
            callback(i + 1, len(timesteps))

    # --- 9. VAE decode ------------------------------------------------------
    # Cast latents back to VAE's dtype.
    needs_upcast = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcast:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    # Use the pipeline's VAE scaling factor.
    latents = latents / pipe.vae.config.scaling_factor
    image_out = pipe.vae.decode(latents, return_dict=False)[0]

    if needs_upcast:
        pipe.vae.to(dtype=torch.float16)

    image_out = pipe.image_processor.postprocess(image_out, output_type="pil")
    return image_out[0]
