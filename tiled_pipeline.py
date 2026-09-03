"""
Image-space tiled img2img for SDXL ControlNet (Pixagram).

Earlier this module used latent-space MultiDiffusion (blending per-tile noise
predictions on a shared latent canvas every step). In theory that keeps tiles
globally coupled; in practice — with LCM's very low step count, a heavy style
LoRA, and an img2img init — each tile only ever saw a small slice of the latent
and never a coherent picture, so the model "didn't understand the image":
soft/confused detail, smeared overlaps, and (worst of all) the portrait-biased
LoRA inventing faces tile-by-tile.

This version does straightforward IMAGE-SPACE tiling instead — the same thing
you'd do by hand:

    1. Cut the prepared image (and its identity-kps + depth control maps) into
       overlapping PIXEL tiles.
    2. Run each tile through the *normal* img2img pipeline as a complete,
       standalone generation. Because it's a full generation on real cropped
       pixels, the model has full context for that tile and reconstructs the
       actual content (faces included) instead of hallucinating new content.
    3. Feather the overlapping tiles back together with a cosine window.

Why this also kills phantom faces:
    Every tile is an img2img of the *actual* pixels under it. A background tile
    starts from background latents and is conditioned with a face-free,
    face-suppressed prompt, so it has no reason to grow a face. A face tile
    redraws the real face already in its crop. Even a face straddling two tiles
    is reconstructed from source pixels in both and merged by the feather —
    never duplicated.

Trade-off: each tile is a full pipeline call, so wall-clock cost scales with
the number of tiles. With LCM (~8-10 steps) a 2x2 grid is comfortable inside
the ZeroGPU window; very large grids (4K) may need a longer
@spaces.GPU(duration=...).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Union

import torch
import numpy as np
import cv2
from PIL import Image


# ============================================================
# Tile grid (pixel space)
# ============================================================

def _compute_tile_positions(total: int, tile: int, overlap: int) -> List[int]:
    """
    Start offsets covering [0, total) with `tile`-wide windows. The requested
    `overlap` sets the window COUNT (enough windows that no seam gets less
    than `overlap`); the actual overlap is then distributed EVENLY across all
    seams. The old walk-then-flush scheme piled the whole remainder onto the
    last seam and, on unlucky sizes, emitted a nearly-duplicate sliver tile
    offset a few pixels from its neighbour. Even spacing gives uniform seams
    and never wastes a window.
    """
    if total <= tile:
        return [0]
    stride = max(1, tile - overlap)
    span = total - tile
    n = 1 + math.ceil(span / stride)        # guarantees seam overlap >= `overlap`
    step = span / float(n - 1)
    positions = [int(round(i * step)) for i in range(n)]
    positions[-1] = span                    # exact flush against the far edge
    out: List[int] = []
    for p in positions:                     # dedupe exact-fit grids
        if not out or p != out[-1]:
            out.append(p)
    return out


def _cosine_window(h: int, w: int) -> np.ndarray:
    """
    2D cosine-bell weight of shape (H, W, 1): ~0 at the edges, 1 in the center.
    Used to feather tile overlaps. Floored so single-tile (non-overlap) pixels
    keep a non-zero weight (the floor cancels in the final division).
    """
    wy = np.sin(math.pi * (np.arange(h) + 0.5) / max(h, 1))
    wx = np.sin(math.pi * (np.arange(w) + 0.5) / max(w, 1))
    win = np.outer(wy, wx).astype(np.float32)
    win = np.clip(win, 1e-3, None)
    return win[:, :, None]


def _point_in_rect(px: float, py: float, rect: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px < x2 and y1 <= py < y2


def _anchor_tile_tone(
    tile: Image.Image, ref: Image.Image, strength: float
) -> Image.Image:
    """
    Match a tile's LOW-FREQUENCY tone to a coherent reference crop WITHOUT
    touching its detail.

    Inter-tile tone/exposure differences (the cause of patchwork) are
    low-frequency by nature. We heavily blur both the tile and the reference
    and shift the tile by `strength * (ref_low - tile_low)`. The high-frequency
    component `tile - tile_low` — i.e. all the fine detail we just generated —
    is preserved exactly, while the smooth tone field is pulled onto the
    reference. Because every tile is corrected toward the SAME coherent
    reference, neighbouring tiles agree at their seam.

    This is deliberately NOT a Reinhard mean/std match: rescaling a tile's std
    toward a smooth (blurry) reference would crush the very detail we want.
    """
    strength = float(max(0.0, min(1.0, strength)))
    if strength == 0.0:
        return tile
    t = np.asarray(tile.convert("RGB"), dtype=np.float32)
    r = np.asarray(
        ref.convert("RGB").resize(tile.size, Image.LANCZOS), dtype=np.float32
    )
    sigma = max(8.0, 0.25 * min(t.shape[0], t.shape[1]))  # large => low-freq only
    t_low = cv2.GaussianBlur(t, (0, 0), sigmaX=sigma, sigmaY=sigma)
    r_low = cv2.GaussianBlur(r, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = t + strength * (r_low - t_low)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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
    bg_prompt: Optional[str] = None,
    bg_negative_prompt: Optional[str] = None,
    face_bboxes_px: Optional[List[Tuple[int, int, int, int]]] = None,
    reference_image: Optional[Image.Image] = None,
    tile_color_match: float = 0.0,
    image_embeds=None,
    ip_adapter_scale: float = 0.0,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0,
    bg_zero_control_indices: Optional[List[int]] = None,
    tile_batch_size: int = 1,
) -> Image.Image:
    """
    Image-space tiled img2img. Signature matches the previous latent
    implementation so it stays drop-in for generator.py.

    Each tile is cropped from the full-resolution image and its control maps,
    then run through the standard SDXL ControlNet img2img pipeline as a normal
    generation, and feathered back into the output canvas.

    Regional prompting is per tile: a tile uses the main `prompt` /
    `negative_prompt` when it CONTAINS a face center, otherwise the face-free
    `bg_prompt` / `bg_negative_prompt` (when supplied). Each face is therefore
    handled by the single tile it sits in, and face-free tiles are actively
    suppressed from growing one.

    Args:
        pipe: StableDiffusionXLControlNetImg2ImgPipeline (multi-controlnet:
              control_images and controlnet_conditioning_scales are parallel
              lists whose order matches the registered ControlNets).
        prompt / negative_prompt: main (face) text conditioning.
        image: full-resolution input PIL image (RGB), (width, height).
        control_images: list of full-resolution control PIL images (e.g.
              [identity kps, depth]). Cropped per tile in lockstep with the image.
        width, height: target generation dimensions (multiples of 8).
        num_inference_steps, guidance_scale, strength: standard img2img knobs,
              applied identically to every tile.
        controlnet_conditioning_scales: per-ControlNet scales.
        generator: torch.Generator; its seed seeds a decorrelated per-tile seed.
        clip_skip: optional clip skip.
        tile_size_px: requested tile edge. Tiles are always SQUARE: the
              effective edge is min(tile_size_px, width, height) snapped
              to a multiple of 8, and the overlap scales with it.
        tile_overlap_px: pixel overlap between adjacent tiles (snapped to 8).
        callback: optional fn(tiles_done, tiles_total) for progress.
        bg_prompt / bg_negative_prompt: face-free prompt set for tiles that do
              not contain a face center. When bg_prompt is None, every tile
              uses the main prompt (correct for face-free whole images, whose
              main prompt is already face-free).
        face_bboxes_px: face bboxes (x1, y1, x2, y2) in pixel coords of the
              prepared image. Needed (with bg_prompt) for regional routing.
        reference_image: a globally-coherent image (e.g. the upscaled base
              pass) at any size. When given with tile_color_match > 0, each
              generated tile's tone is matched to the SAME region of this
              reference BEFORE stitching, so neighbouring tiles agree at their
              seam and the mosaic is patchwork-free.
        tile_color_match: per-tile local colour-anchoring strength (0 = off,
              ~0.8 = strong tone lock). Affine LAB match, so detail is kept.
        image_embeds: InstantID ArcFace identity embedding (512-d, numpy or
              tensor). When provided AND the pipe exposes
              set_ip_adapter_scale, identity is injected on FACE tiles at
              `ip_adapter_scale` and switched OFF (scale 0) on background
              tiles — an identity embedding attended into a faceless tile is
              phantom-face pressure. Pass None for non-InstantID pipelines.
        ip_adapter_scale: IP-Adapter scale applied on face tiles.
        control_guidance_start / control_guidance_end: per-ControlNet
              guidance window (fraction of each tile's denoise schedule),
              passed straight through to the pipeline. With the pipeline's
              zero-scale skip, steps outside the window don't evaluate the
              net at all.
        bg_zero_control_indices: indices (into control_images /
              controlnet_conditioning_scales) of ControlNets forced to
              scale 0 on background tiles — in practice [0], the
              IdentityNet, whose kps crop is black there anyway. Combined
              with the pipeline's zero-scale skip this removes that net's
              forward from every background tile.
        tile_batch_size: how many same-group tiles to run through the UNet
              as ONE batch. Face and background tiles batch separately
              (they differ in embeddings, ControlNet scales and IP scale).
              VRAM scales ~linearly with it (x2 again while CFG is active);
              1 reproduces the sequential behaviour exactly — per-tile
              generators keep the noise identical either way, so output is
              invariant to this setting up to conv batching numerics.

    Returns:
        Stitched PIL image at (width, height).
    """
    device = pipe._execution_device

    # Quieten per-tile tqdm bars (best-effort).
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    # ---- Prompt-embedding cache ---------------------------------------
    # A tiled run has at most TWO distinct (prompt, negative) pairs — the
    # face prompt and the background prompt — yet both CLIP encoders used
    # to run for EVERY tile. Encode each pair once up front and feed the
    # cached tensors to every tile call (the pipeline skips encode_prompt
    # when embeds are provided). clip_skip is baked into the cached embeds.
    do_cfg = (
        float(guidance_scale) > 1.0
        and getattr(pipe.unet.config, "time_cond_proj_dim", None) is None
    )
    _embed_cache = {}

    def _embeds_for(p: str, n: str):
        key = (p, n)
        if key not in _embed_cache:
            _embed_cache[key] = pipe.encode_prompt(
                prompt=p,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=n,
                negative_prompt_2=None,
                clip_skip=clip_skip,
            )
        return _embed_cache[key]

    # Ensure the working image and every control map are exactly (width, height)
    # so crops line up pixel-for-pixel.
    def _fit(im: Image.Image) -> Image.Image:
        im = im.convert("RGB")
        return im if im.size == (width, height) else im.resize(
            (width, height), Image.LANCZOS
        )

    base_image = _fit(image)
    base_controls = [_fit(c) for c in control_images]

    # Per-tile colour anchor: a coherent reference (e.g. the upscaled base
    # pass) whose LOCAL tone each tile is matched to before stitching. Because
    # the reference is globally coherent, neighbouring tiles end up agreeing in
    # tone at their shared seam — which is what removes the patchwork that
    # otherwise forces a manual GIMP fix-up. Matching is per-channel LAB
    # mean/std (affine), so it aligns tone WITHOUT removing the tile's detail.
    ref_image = _fit(reference_image) if reference_image is not None else None
    do_tile_cm = ref_image is not None and float(tile_color_match) > 0.0
    if do_tile_cm:
        print(
            f"[Tiled/image-space] per-tile colour anchor ON "
            f"(strength {float(tile_color_match):.2f})"
        )

    # SQUARE tiles, snapped to multiples of 8 within image bounds.
    # The tile edge is clamped to the SHORT canvas edge and used on BOTH
    # axes. The old per-axis clamp (tw=min(tile,w), th=min(tile,h)) produced
    # stretched crops like 768x664 whenever one canvas dimension dropped
    # below the tile size (every 16:9 rung, wide budgets, forced tiling on
    # small canvases) — and SDXL + the pixel LoRAs render noticeably better
    # on square crops (consistent size conditioning, isotropic dither/grid).
    # The short axis now holds exactly one row/col; the long axis gets more,
    # smaller, square tiles instead.
    tile_req = max(64, int(tile_size_px))
    tile_px = max(64, min(tile_req, width, height))
    tile_px -= tile_px % 8
    # Scale the overlap with the tile so the feather stays the same FRACTION
    # of a tile (192/768 = 25%) instead of eating a growing share of smaller
    # tiles. Unchanged whenever the tile isn't shrunk.
    overlap_px = int(round(max(0, int(tile_overlap_px)) * tile_px / float(tile_req)))
    overlap_px = max(0, min(overlap_px, tile_px - 8))
    overlap_px -= overlap_px % 8

    tw = th = tile_px
    x_positions = _compute_tile_positions(width, tw, overlap_px)
    y_positions = _compute_tile_positions(height, th, overlap_px)
    num_tiles = len(x_positions) * len(y_positions)

    use_regional = bool(bg_prompt) and bool(face_bboxes_px)
    bg_neg = bg_negative_prompt or negative_prompt

    print(
        f"[Tiled/image-space] {len(x_positions)}x{len(y_positions)} = "
        f"{num_tiles} tile(s), tile {tw}x{th}px, overlap {overlap_px}px, "
        f"regional={use_regional}"
    )

    # Float accumulators in pixel space.
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)

    try:
        base_seed = int(generator.initial_seed())
    except Exception:
        base_seed = 0

    # ------------------------------------------------------------------
    # Build the tile work list first (rects + face/background routing),
    # then run it in BATCHED groups. All tiles in a group share prompt
    # embeddings, ControlNet scales and IP-Adapter scale, so they can go
    # through the UNet as one batch. Per-tile generators keep the noise
    # identical to the sequential path, so output is invariant to
    # tile_batch_size (up to conv batching numerics).
    # ------------------------------------------------------------------
    n_face_tiles = 0
    tile_jobs: List[Tuple[int, Tuple[int, int, int, int], bool]] = []
    for ty in y_positions:
        for tx in x_positions:
            tile_rect = (tx, ty, tx + tw, ty + th)

            # Per-tile regional routing: a tile is a "face tile" iff a face
            # center falls inside it. Center-in-tile assigns each face to a
            # single tile, so the portrait prompt is never broadcast to a
            # whole neighbourhood of tiles.
            is_face_tile = True
            if use_regional:
                is_face_tile = any(
                    _point_in_rect(
                        (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, tile_rect
                    )
                    for b in face_bboxes_px
                )
            if is_face_tile:
                n_face_tiles += 1
            tile_jobs.append((len(tile_jobs), tile_rect, is_face_tile))

    if use_regional:
        print(
            f"[Tiled/image-space] {n_face_tiles} face tile(s), "
            f"{num_tiles - n_face_tiles} background tile(s), "
            f"batch={max(1, int(tile_batch_size))}"
        )

    # Per-group ControlNet scales: background tiles get the nets in
    # bg_zero_control_indices (the IdentityNet, in practice) zeroed —
    # combined with the pipeline's zero-scale skip they are not evaluated
    # at all on those tiles.
    face_scales = list(controlnet_conditioning_scales)
    bg_scales = list(controlnet_conditioning_scales)
    if bg_zero_control_indices:
        for _ci in bg_zero_control_indices:
            if 0 <= _ci < len(bg_scales):
                bg_scales[_ci] = 0.0

    # InstantID routing: identity is injected only where the face actually
    # is. Background tiles run at IP scale 0 — attending an identity
    # embedding into a faceless tile invites a phantom face. The kps
    # control crop is black there anyway.
    face_jobs = [j for j in tile_jobs if j[2]]
    bg_jobs = [j for j in tile_jobs if not j[2]]
    groups = []
    if face_jobs:
        groups.append(
            (face_jobs, prompt, negative_prompt, face_scales, float(ip_adapter_scale))
        )
    if bg_jobs:
        groups.append((bg_jobs, (bg_prompt or prompt), bg_neg, bg_scales, 0.0))

    batch = max(1, int(tile_batch_size))
    # All tiles are the same (square) size now, so the feather window can
    # be built once.
    win = _cosine_window(th, tw)

    done = 0
    for jobs, g_prompt, g_neg, g_scales, g_ip in groups:
        g_pe, g_npe, g_ppe, g_nppe = _embeds_for(g_prompt, g_neg)
        if image_embeds is not None and hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(g_ip)

        for chunk_start in range(0, len(jobs), batch):
            chunk = jobs[chunk_start : chunk_start + batch]
            n = len(chunk)

            tile_imgs = [base_image.crop(rect) for _, rect, _ in chunk]
            # check_inputs rejects nested lists for MultiControlNet, so
            # each net's per-tile batch goes in as an [n,3,H,W] float
            # tensor in [0,1] — numerically identical to what the control
            # image processor derives from PILs.
            tile_controls = [
                torch.stack(
                    [
                        torch.from_numpy(
                            np.asarray(c.crop(rect), dtype=np.float32) / 255.0
                        ).permute(2, 0, 1)
                        for _, rect, _ in chunk
                    ]
                )
                for c in base_controls
            ]

            # Deterministic, decorrelated per-tile seeds — same derivation
            # as the sequential path, keyed by the tile's row-major index,
            # one generator per sample so the noise matches tile-for-tile.
            gens = [
                torch.Generator(device=device).manual_seed(
                    (base_seed + tile_i) % (2**63 - 1)
                )
                for tile_i, _, _ in chunk
            ]

            extra_kwargs = {}
            if image_embeds is not None:
                extra_kwargs["image_embeds"] = image_embeds

            result = pipe(
                image=tile_imgs,
                control_image=tile_controls,
                prompt=None,
                negative_prompt=None,
                prompt_embeds=g_pe.repeat(n, 1, 1),
                negative_prompt_embeds=(
                    g_npe.repeat(n, 1, 1) if g_npe is not None else None
                ),
                pooled_prompt_embeds=g_ppe.repeat(n, 1),
                negative_pooled_prompt_embeds=(
                    g_nppe.repeat(n, 1) if g_nppe is not None else None
                ),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                controlnet_conditioning_scale=list(g_scales),
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                generator=gens,
                width=tw,
                height=th,
                **extra_kwargs,
            )

            for (_, rect, _), tile_pil in zip(chunk, result.images):
                x1, y1, x2, y2 = rect
                tile_pil = tile_pil.convert("RGB")
                if do_tile_cm:
                    # Pull THIS tile's tone onto the same region of the
                    # coherent reference, so it matches its neighbours
                    # before feathering.
                    ref_crop = ref_image.crop(rect)
                    tile_pil = _anchor_tile_tone(
                        tile_pil, ref_crop, float(tile_color_match)
                    )
                tile_out = np.asarray(tile_pil, dtype=np.float32)
                canvas[y1:y2, x1:x2, :] += tile_out * win
                weights[y1:y2, x1:x2, :] += win
                done += 1
                if callback is not None:
                    callback(done, num_tiles)

            del result, tile_imgs, tile_controls
            # No empty_cache() here: chunks allocate identical shapes, so
            # the caching allocator reuses the same blocks; empty_cache()
            # forced a device sync + a full re-allocation for no benefit
            # (and fights PYTORCH_CUDA_ALLOC_CONF=expandable_segments).

    # Restore the caller's IP-Adapter scale — the background group may
    # have left it at 0, which would silently disable identity for any
    # later full-image pass on the same pipe.
    if image_embeds is not None and hasattr(pipe, "set_ip_adapter_scale"):
        pipe.set_ip_adapter_scale(float(ip_adapter_scale))

    blended = canvas / np.clip(weights, 1e-6, None)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
