"""
Image generation logic for the Pixagram Pixel Art Converter.
Optimized for ZeroGPU with speed improvements and aspect ratio preservation.
Includes optional face identity preservation using facenet-pytorch (MIT License).

Quality features:
  - BLIP auto-captioning: describes actual image content in prompt
  - Face attribute enrichment: adds face descriptors when AUTO_FACEID is enabled
  - Face-aware ControlNet tuning: boosts depth and reduces img2img strength
    in the face region so facial features are better preserved
  - Combined style triggers for stronger LoRA activation
"""

import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from typing import Tuple, Optional, List

from config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_NEGATIVE_PROMPT,
    IMG_STRENGTH,
    DEPTH_STRENGTH,
    CLIP_SKIP,
    FACE_BLEND_STRENGTH,
    FACE_SIMILARITY_THRESHOLD,
    FACE_DEPTH_BOOST,
    FACE_STRENGTH_REDUCTION,
    AUTO_FACEID,
    STYLE_TRIGGER1,
    STYLE_TRIGGER2,
    TRIGGER_WORD,
    MAX_RESOLUTION,
    DEFAULT_RESOLUTION,
    TILE_SIZE,
    TILE_OVERLAP,
    DEFAULT_USE_TILED,
    CANNY_LOW,
    CANNY_HIGH,
    FACE_CANNY_DILATE,
    FACE_CANNY_SCALE,
    DEFAULT_LORA_INTENSITY,
    LORA_FACE_MULTIPLIER,
    LORA_NOFACE_MULTIPLIER,
    LORA_INTENSITY_MAX,
    IMG_FACE_MULTIPLIER,
    IMG_NOFACE_MULTIPLIER,
    IMG_STRENGTH_MIN,
    IMG_STRENGTH_MAX,
)
from model import get_pipeline, get_zoe_detector, get_face_analyzer
from utils import prepare_image, create_seed, format_prompt, get_caption
from tiled_pipeline import tiled_controlnet_img2img


# "portrait" baits SDXL into generating faces. Only include it when we
# have CONFIRMED a face exists in the input.
def _style_trigger(has_face: bool) -> str:
    """Return the LoRA trigger string, with the 'portrait' token added
    only when a face has been confirmed in the input."""
    if has_face:
        return TRIGGER_WORD  # full string includes "retro game art portrait"
    # Strip portrait-related tokens for non-face content. The first
    # trigger already establishes style; the second one carries the
    # face-bait keywords we want to drop.
    safe_trigger2 = "artwork, illustration, pixel art, retro game art, art."
    return STYLE_TRIGGER1 + ", " + safe_trigger2


def _confirm_face_with_mtcnn(
    image: Image.Image,
    min_prob: float = 0.985,
) -> bool:
    """
    Verify a face really exists using MTCNN (much lower false-positive
    rate than the Haar cascade in faceid.py).

    Returns True only when MTCNN finds at least one face with
    detection probability >= min_prob. We use a HIGHER bar here than
    in face_preserve.py because a false positive here corrupts the
    *prompt*, which is far worse than a missed blend.
    """
    return len(_get_mtcnn_face_bboxes(image, min_prob=min_prob)) > 0


def _get_mtcnn_face_bboxes(
    image: Image.Image,
    min_prob: float = 0.985,
) -> List[Tuple[int, int, int, int]]:
    """
    Return high-confidence MTCNN face bboxes (x1, y1, x2, y2).
    Returns [] on any error or if no faces meet the confidence bar.

    The threshold here intentionally exceeds the face_preserve default
    (0.95). This function gates the *prompt construction* and the
    *depth boost*; both are operations whose failure mode is "paint a
    face that wasn't in the input." Better to miss a borderline face
    than to hallucinate one.
    """
    try:
        from face_preserve import detect_faces
        faces = detect_faces(image, device="cuda", min_prob=min_prob)
        return [f.bbox for f in faces]
    except Exception as e:
        print(f"[Generator] MTCNN face detection failed (non-fatal): {e}")
        return []


def _build_rich_prompt(
    prepared_image: Image.Image,
    additional_prompt: str,
    face_confirmed: bool,
    preserve_identity: bool = False,
) -> str:
    """
    Build a high-quality prompt by combining:
      1. Conditional style trigger words (drops "portrait" when no face)
      2. BLIP auto-caption of the actual image content
      3. Face attribute descriptors (only when face_confirmed is True)
      4. User-supplied additional style instructions

    Args:
        prepared_image: The resized input image.
        additional_prompt: User-provided extra style instructions.
        face_confirmed: True only when MTCNN has confirmed a face with
                        high probability. When False, no face-related
                        tokens are added to the prompt — this prevents
                        the model from hallucinating faces in landscapes,
                        still lifes, animal photos, etc.
    """
    parts = [_style_trigger(face_confirmed)]

    # Auto-caption: describe what's actually in the image
    try:
        caption = get_caption(prepared_image)
        if caption and caption.strip():
            parts.append(caption.strip())
    except Exception as e:
        print(f"[Generator] Captioning failed (non-fatal): {e}")

    # Face attribute enrichment.
    #
    # IMPORTANT: generic descriptors like "young woman, brown hair, oval
    # face" describe a *category*, not a specific person. Feeding them to
    # SDXL alongside a depth-only ControlNet (which fixes pose but not
    # identity) is a recipe for the model to paint a brand-new, plausible
    # face that merely *matches the description* — i.e. "an alternative
    # face that doesn't exist". We therefore only add these descriptors
    # when the user has explicitly opted out of identity preservation;
    # otherwise we let the img2img latents + depth carry the face.
    if face_confirmed and AUTO_FACEID and not preserve_identity:
        try:
            analyzer = get_face_analyzer()
            if analyzer is not None:
                face_desc = analyzer.get_prompt_enrichment(prepared_image)
                if face_desc:
                    parts.append(face_desc)
        except Exception as e:
            print(f"[Generator] Face attribute enrichment failed (non-fatal): {e}")

    # User's additional instructions come last so they can override
    if additional_prompt and additional_prompt.strip():
        parts.append(additional_prompt.strip())

    return ", ".join(parts)


def _boost_depth_for_faces(
    depth_image: Image.Image,
    face_bboxes: List[Tuple[int, int, int, int]],
    boost_factor: float = FACE_DEPTH_BOOST,
) -> Image.Image:
    """
    Increase depth values in face regions so ControlNet preserves
    more facial structure during generation.

    Args:
        depth_image: The Zoe depth map.
        face_bboxes: List of (x1, y1, x2, y2) bboxes from MTCNN. Uses
                     the same trusted detections as the prompt-gating
                     step so we never boost depth on a phantom face.
        boost_factor: Multiplicative boost applied to depth values in
                      the face region.
    """
    if not face_bboxes:
        return depth_image

    depth_np = np.array(depth_image).astype(np.float32)
    h, w = depth_np.shape[:2]

    for (x1_f, y1_f, x2_f, y2_f) in face_bboxes:
        fw = x2_f - x1_f
        fh = y2_f - y1_f
        # Expand the region slightly
        pad_x = int(fw * 0.2)
        pad_y = int(fh * 0.2)
        x1 = max(0, x1_f - pad_x)
        y1 = max(0, y1_f - pad_y)
        x2 = min(w, x2_f + pad_x)
        y2 = min(h, y2_f + pad_y)

        # Create a soft mask for the boost region
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=15))
        mask = np.array(mask_pil).astype(np.float32) / 255.0

        # Apply boost: blend between original and boosted depth
        if depth_np.ndim == 3:
            mask = np.stack([mask] * depth_np.shape[2], axis=-1)
        boosted = depth_np * boost_factor
        depth_np = depth_np * (1.0 - mask) + boosted * mask

    depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
    return Image.fromarray(depth_np)


def _make_face_canny(
    prepared_image: Image.Image,
    face_bboxes: List[Tuple[int, int, int, int]],
    low: int = CANNY_LOW,
    high: int = CANNY_HIGH,
    dilate_frac: float = FACE_CANNY_DILATE,
) -> Image.Image:
    """
    Build a Canny edge map of the input, then keep edges ONLY inside the
    (slightly expanded) face boxes. Everything outside the face is black.

    This is the identity lock: a Canny ControlNet treats black as "no
    constraint", so it imposes the real person's facial feature-lines on
    the face region while leaving the rest of the image free for the LoRA
    to stylize. With no face boxes the whole map is black -> the canny
    ControlNet contributes nothing and behavior matches depth-only.

    OpenCV (BSD) only — no neural model.

    Returns:
        RGB PIL image (white edges on black), same size as prepared_image.
    """
    rgb = np.array(prepared_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    h, w = edges.shape

    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in face_bboxes:
        fw, fh = x2 - x1, y2 - y1
        px, py = int(fw * dilate_frac), int(fh * dilate_frac)
        mx1 = max(0, x1 - px)
        my1 = max(0, y1 - py)
        mx2 = min(w, x2 + px)
        my2 = min(h, y2 + py)
        mask[my1:my2, mx1:mx2] = 255

    edges = cv2.bitwise_and(edges, edges, mask=mask)
    canny_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(canny_rgb)


def generate_pixel_art(
    input_image: Image.Image,
    additional_prompt: str = "",
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    seed: int = -1,
    lora_intensity: float = DEFAULT_LORA_INTENSITY,
    img2img_strength: float = IMG_STRENGTH,
    face_preserve: bool = False,
    face_blend_strength: float = FACE_BLEND_STRENGTH,
    face_similarity_threshold: float = FACE_SIMILARITY_THRESHOLD,
    resolution: int = DEFAULT_RESOLUTION,
    use_tiled: bool = DEFAULT_USE_TILED,
    tile_size: int = TILE_SIZE,
    tile_overlap: int = TILE_OVERLAP,
) -> Tuple[Image.Image, int, Optional[str], str]:
    """
    Generate pixel art from an input image.
    This function should be called within a GPU context (via @spaces.GPU decorator).
    Preserves the original image aspect ratio.

    Args:
        input_image: Input PIL Image
        additional_prompt: Additional style instructions to append to trigger word
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        seed: Random seed (-1 for random)
        face_preserve: Whether to enable face identity preservation
        face_blend_strength: Max blend strength for face preservation (0-1)
        face_similarity_threshold: Cosine similarity threshold for blending
        resolution: Longer-edge target resolution in pixels (up to MAX_RESOLUTION).
        use_tiled: If True, run MultiDiffusion tiled denoising. Auto-enabled
                   whenever the prepared image exceeds the tile size.
        tile_size: Pixel tile size for tiled denoising (default 768).
        tile_overlap: Pixel overlap between adjacent tiles (default 192).

    Returns:
        Tuple of (generated image, seed used, face report or None, full prompt)
    """
    # Save original dimensions for final restore
    original_size = input_image.size  # (w, h)

    # Clamp the requested resolution to the hard cap.
    resolution = max(512, min(int(resolution), MAX_RESOLUTION))

    # Prepare the image and get target dimensions (preserves aspect ratio).
    prepared_image, target_width, target_height = prepare_image(
        input_image, long_side=resolution
    )

    # Auto-enable tiling whenever the prepared image is bigger than the
    # tile in either dimension. The user toggle only matters in the
    # ambiguous middle range.
    must_tile = target_width > tile_size or target_height > tile_size
    use_tiled = bool(use_tiled) or must_tile

    # ---- Face detection gate -----------------------------------------
    # We need a HIGH-CONFIDENCE signal here because it controls the
    # prompt ("portrait of young woman, brown hair...") and the depth
    # boost. A false positive forces SDXL to paint a face into a
    # landscape. Trust MTCNN over the Haar cascade, and use 0.985 —
    # higher than the face_preserve default so the prompt/depth path
    # is even more conservative than the (already conservative)
    # blending path.
    face_bboxes: List[Tuple[int, int, int, int]] = _get_mtcnn_face_bboxes(
        prepared_image, min_prob=0.985
    )
    face_confirmed = len(face_bboxes) > 0

    if face_confirmed:
        print(f"[Generator] MTCNN confirmed {len(face_bboxes)} face(s)")
    else:
        # Fall back to Haar ONLY for the analyzer-attribute path — and
        # even then we only use it if we have an actual MTCNN face,
        # which we don't here. So we don't fall back.
        print("[Generator] No high-confidence face — treating as non-portrait")

    # Build a rich prompt. When face_confirmed is False, the trigger
    # drops "portrait" and no face attributes are added — so SDXL
    # won't be biased into generating a face that wasn't there.
    prompt = _build_rich_prompt(
        prepared_image, additional_prompt, face_confirmed,
        preserve_identity=face_preserve,
    )
    print(f"[Generator] Prompt: {prompt[:120]}...")
    print(
        f"[Generator] Resolution {target_width}x{target_height} "
        f"(requested long side {resolution}), tiled={use_tiled}"
    )

    # Create seed
    actual_seed = create_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(actual_seed)

    # Get the pipeline and depth detector
    pipe = get_pipeline()
    zoe = get_zoe_detector()

    # Generate depth map (required by ControlNet). Zoe handles arbitrary
    # input sizes — we then resize the result to match the prepared image.
    depth_image = zoe(prepared_image)
    if depth_image.size != (target_width, target_height):
        depth_image = depth_image.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )

    # Face-aware tuning. With the canny lock now preserving facial structure
    # during diffusion, we apply simple per-context img2img multipliers to the
    # caller's base strength (instead of the old strength-reduction):
    #   face    -> x IMG_FACE_MULTIPLIER   (1.0: full redraw, canny holds ID)
    #   no face -> x IMG_NOFACE_MULTIPLIER (0.5: stay close to the source)
    depth_strength = DEPTH_STRENGTH

    if face_confirmed:
        print("[Generator] Confirmed face: depth boost + full img2img (canny holds identity)")
        depth_image = _boost_depth_for_faces(depth_image, face_bboxes)
        img_strength = img2img_strength * IMG_FACE_MULTIPLIER
        depth_strength = min(1.0, DEPTH_STRENGTH + 0.1)
    else:
        img_strength = img2img_strength * IMG_NOFACE_MULTIPLIER

    img_strength = max(IMG_STRENGTH_MIN, min(IMG_STRENGTH_MAX, img_strength))
    _img_mult = IMG_FACE_MULTIPLIER if face_confirmed else IMG_NOFACE_MULTIPLIER
    print(
        f"[Generator] img2img: base={img2img_strength} x {_img_mult} "
        f"({'face' if face_confirmed else 'no-face'}) = {img_strength:.3f}"
    )

    # Face structure lock: Canny edges masked to the face box. This is the
    # identity constraint depth can't provide. When no face was confirmed
    # the map is all-black and the canny ControlNet is a no-op.
    canny_image = _make_face_canny(prepared_image, face_bboxes)
    canny_strength = FACE_CANNY_SCALE if face_confirmed else 0.0
    if face_confirmed:
        print(f"[Generator] Canny face lock active (scale={canny_strength})")

    # Control-image / scale lists. ORDER MUST MATCH model.py: [depth, canny].
    control_images = [depth_image, canny_image]
    control_scales = [depth_strength, canny_strength]

    # Style intensity. `lora_intensity` is the API-exposed BASE value; we
    # multiply it by a face/non-face factor (faces boosted since the canny
    # lock holds their structure; non-faces attenuated to avoid over-cooking),
    # then clamp. The LoRA is unfused so we set the scale live per request.
    lora_mult = LORA_FACE_MULTIPLIER if face_confirmed else LORA_NOFACE_MULTIPLIER
    lora_scale = max(0.0, min(LORA_INTENSITY_MAX, float(lora_intensity) * lora_mult))
    try:
        pipe.set_adapters(["retroart"], adapter_weights=[lora_scale])
        print(
            f"[Generator] LoRA: base={lora_intensity} x {lora_mult} "
            f"({'face' if face_confirmed else 'no-face'}) = scale {lora_scale:.3f}"
        )
    except Exception as e:
        print(f"[Generator] Could not set LoRA scale ({e}); using loaded default")

    # Free VRAM from captioning/depth before heavy diffusion step
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Run diffusion: tiled (MultiDiffusion) or standard pipeline call
    # ----------------------------------------------------------------
    if use_tiled:
        output_image = tiled_controlnet_img2img(
            pipe,
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            image=prepared_image,
            control_images=control_images,
            width=target_width,
            height=target_height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=img_strength,
            controlnet_conditioning_scales=control_scales,
            generator=generator,
            clip_skip=CLIP_SKIP,
            tile_size_px=tile_size,
            tile_overlap_px=tile_overlap,
        )
    else:
        with torch.inference_mode():
            result = pipe(
                image=prepared_image,
                control_image=control_images,
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=img_strength,
                controlnet_conditioning_scale=control_scales,
                clip_skip=CLIP_SKIP,
                generator=generator,
                width=target_width,
                height=target_height,
            )
        output_image = result.images[0]

    # NOTE: we used to resize back to the original input resolution here.
    # That silently softened pixel-art crispness, so we now return the
    # output at the (potentially larger) generation resolution.

    face_report = None

    # Apply face identity preservation if enabled
    if face_preserve:
        try:
            from face_preserve import preserve_face_identity

            output_image, face_report = preserve_face_identity(
                original_image=prepared_image,
                generated_image=output_image,
                blend_strength=face_blend_strength,
                similarity_threshold=face_similarity_threshold,
                device="cuda",
            )
        except Exception as e:
            face_report = f"Face preservation error: {str(e)}"

    return output_image, actual_seed, face_report, prompt
