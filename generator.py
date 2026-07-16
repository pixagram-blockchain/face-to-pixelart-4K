"""
Image generation logic for the Pixagram Pixel Art Converter.
Optimized for ZeroGPU with speed improvements and aspect ratio preservation.

Two generation modes:
  - IMG2IMG (an input image is provided): the image is redrawn in the
    pixel-art style, guided by Zoe depth + InstantID conditioning. Face
    identity is preserved DURING diffusion via InstantID:
      * IdentityNet ControlNet conditioned on the real face's 5-point kps
      * IP-Adapter cross-attending the ArcFace embedding into the UNet
  - TEXT-TO-IMAGE (no input image): the prompt is MANDATORY and drives the
    content. The canvas comes from the requested resolution + aspect ratio,
    and the img2img pipeline is run at strength 1.0 over a neutral init
    (equivalent to txt2img) with both ControlNets zeroed out — there is no
    control image in this mode.

High-resolution path (image-space tiling):
  Below BASE_PASS_LONG_EDGE every generation is a SINGLE pipeline pass.
  Above it (or when forced with use_tiled), generation runs COARSE-TO-FINE:
  one coherent single-pass base at a model-native size, then low-strength
  image-space TILED refinement rungs climbing to the target resolution
  (see _coarse_to_fine / tiled_pipeline.py). InstantID routes per tile —
  the identity embedding is injected only into tiles that own a face, so
  background tiles can't grow a phantom face from the embedding.

Quality features:
  - Trigger-word-only prompting: img2img generation is driven purely by the
    LoRA style trigger word(s); BLIP auto-captioning is disabled by default
    (Config.USE_CAPTION) because injected person-nouns ("a man...", "a
    person...") spawned phantom/duplicated faces
  - insightface antelopev2 supplies detection, keypoints and embeddings in
    one pass (replaces MTCNN + the Haar attribute heuristics)
  - Face-aware ControlNet tuning: boosts depth in the face region
  - Combined style triggers for stronger LoRA activation
"""

import re
import math
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from typing import Tuple, Optional, List

from config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_NEGATIVE_PROMPT,
    NOFACE_NEGATIVE_ADDON,
    IMG_STRENGTH,
    DEPTH_STRENGTH,
    CLIP_SKIP,
    DEFAULT_IDENTITYNET_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_FACE_PRESERVE_ENABLED,
    FACE_DET_SCORE_MIN,
    PHANTOM_FACE_MIN_SCORE,
    FACE_DEPTH_BOOST,
    FACE_BOX_DILATE,
    AUTO_FACEID,
    USE_CAPTION,
    STYLE_TRIGGER1,
    STYLE_TRIGGER2,
    TRIGGER_WORD,
    MAX_RESOLUTION,
    DEFAULT_RESOLUTION,
    DEFAULT_LORA_INTENSITY,
    LORA_FACE_MULTIPLIER,
    LORA_NOFACE_MULTIPLIER,
    LORA_INTENSITY_MAX,
    IMG_FACE_MULTIPLIER,
    IMG_NOFACE_MULTIPLIER,
    IMG_STRENGTH_MIN,
    IMG_STRENGTH_MAX,
    FACE_NEGATIVE_ADDON,
    SOLO_PROMPT_ADDON,
    FACE_GUIDANCE_MIN,
    PHANTOM_FACE_MAX_RETRIES,
    ASPECT_RATIOS,
    DEFAULT_ASPECT_RATIO,
    TXT2IMG_STRENGTH,
    TILE_SIZE,
    TILE_OVERLAP,
    DEFAULT_USE_TILED,
    TWO_PASS_REFINE,
    BASE_PASS_LONG_EDGE,
    REFINE_STRENGTH,
    COLOR_MATCH,
    COLOR_MATCH_STRENGTH,
    COLOR_MATCH_REFERENCE,
    TILE_COLOR_MATCH,
    USE_ESCALATION,
    ESCALATION_FACTOR,
    ESCALATION_MAX_STEPS,
)
from model import get_pipeline, get_zoe_detector, get_face_app
from utils import create_seed, match_colors_lab
from tiled_pipeline import tiled_controlnet_img2img
from pipeline_stable_diffusion_xl_instantid_img2img import draw_kps


# "portrait" baits SDXL into generating faces. Only include it when we
# have CONFIRMED a face exists in the input.
def _style_trigger(has_face: bool) -> str:
    """Return the LoRA trigger string, with the 'portrait' token added
    only when a face has been confirmed in the input."""
    if has_face:
        return TRIGGER_WORD  # full string includes "retro game art portrait"
    # Strip the portrait bait for non-face content: drop the "portrait"
    # token from the second trigger ("retro game art portrait" ->
    # "retro game art"). The first trigger already establishes the style.
    safe_trigger2 = STYLE_TRIGGER2.replace(" portrait", "")
    return STYLE_TRIGGER1 + ", " + safe_trigger2


# Nouns/pronouns that bait SDXL's portrait-biased LoRA into synthesizing a
# face. Stripped from the BLIP caption whenever insightface did NOT confirm a real
# face — BLIP happily writes "a man standing...", "a statue of a woman",
# "a close up of a person" for things the (deliberately strict) insightface
# gate skips: small/profile/occluded faces, statues, dolls, crowds, animals.
# That single noun, plus the portrait-trained LoRA, is what paints a face
# where none exists.
_PERSON_TOKENS = {
    "person", "persons", "people", "man", "men", "woman", "women", "boy",
    "boys", "girl", "girls", "child", "children", "kid", "kids", "baby",
    "babies", "lady", "ladies", "guy", "guys", "gentleman", "gentlemen",
    "human", "humans", "face", "faces", "portrait", "portraits", "selfie",
    "selfies", "headshot", "eyes", "eye", "mouth", "nose", "he", "she",
    "him", "her", "his", "hers", "they", "them", "male", "female",
    "teenager", "teenagers", "adult", "adults", "elderly", "model",
    "models", "crowd", "crowds", "someone", "somebody",
    # apostrophe-collapsed possessives: after stripping non-alpha chars
    # "woman's" -> "womans", "man's" -> "mans", etc. Catch those too.
    "mans", "womans", "womens", "mens", "peoples", "childs", "ladys", "babys",
}


def _scrub_person_tokens(caption: str) -> str:
    """
    Remove person/face nouns from a caption so the portrait-biased LoRA is
    not told to draw a person when insightface found no real face. Grammar is
    irrelevant for SDXL conditioning, so we simply drop the offending words
    and tidy up whitespace/punctuation. Returns "" if nothing meaningful
    survives (e.g. "a man" -> "").
    """
    if not caption:
        return caption
    kept = [
        w for w in caption.split()
        if re.sub(r"[^a-z]", "", w.lower()) not in _PERSON_TOKENS
    ]
    cleaned = re.sub(r"\s+([,.;])", r"\1", " ".join(kept))  # no space before punct
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")
    return cleaned if len(cleaned) >= 3 else ""


def _detect_faces_insight(
    image: Image.Image,
    min_score: float = FACE_DET_SCORE_MIN,
) -> List:
    """
    Detect faces with insightface antelopev2 (SCRFD). Returns the raw
    insightface Face objects — each carries .bbox, .kps (5 points),
    .embedding (ArcFace, 512-d), .det_score, .sex, .age — sorted by
    bbox area, largest first. Returns [] when insightface is
    unavailable, on any error, or when no face meets the score bar.

    This single call replaces both MTCNN (detection gate) and the Haar
    attribute heuristics: the same objects drive the prompt gate, the
    depth boost, AND the InstantID conditioning (kps + embedding).
    """
    face_app = get_face_app()
    if face_app is None:
        return []
    try:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = [
            f for f in face_app.get(bgr)
            if float(getattr(f, "det_score", 1.0)) >= float(min_score)
        ]
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces
    except Exception as e:
        print(f"[Generator] insightface detection failed (non-fatal): {e}")
        return []


def _faces_to_bboxes(faces: List) -> List[Tuple[int, int, int, int]]:
    """Convert insightface Face objects to integer (x1, y1, x2, y2) tuples."""
    return [
        tuple(int(round(float(v))) for v in f.bbox[:4]) for f in faces
    ]


def _insight_prompt_enrichment(face) -> str:
    """
    Coarse face descriptors from insightface's genderage head — used only
    when identity preservation is OFF (see _build_prompts). Replaces the
    old OpenCV-Haar guesswork with the model's actual estimates.
    """
    if face is None:
        return ""
    tokens: List[str] = []
    age = getattr(face, "age", None)
    sex = getattr(face, "sex", None)  # 'M' / 'F'
    noun = {"M": "man", "F": "woman"}.get(sex, "person")
    if age is not None:
        age = int(age)
        if age < 18:
            tokens.append("young person")
        elif age < 30:
            tokens.append(f"young {noun}")
        elif age < 55:
            tokens.append(noun)
        else:
            tokens.append(f"older {noun}")
    elif noun != "person":
        tokens.append(noun)
    return ", ".join(tokens)


def _make_identity_kps(
    prepared_image: Image.Image,
    primary_face,
) -> Image.Image:
    """
    Build the IdentityNet conditioning image: the InstantID 5-point facial
    keypoint map of the primary (largest) face, drawn at the prepared
    image's size. With no face, an all-black canvas is returned and the
    generator sets the IdentityNet conditioning scale to 0 (no-op).

    Multi-face note: InstantID conditions ONE identity (a single
    embedding), so the kps map carries only the primary face; other faces
    are still preserved by the img2img latents + depth.
    """
    if primary_face is None:
        return Image.new("RGB", prepared_image.size, (0, 0, 0))
    return draw_kps(prepared_image, primary_face.kps)


def _build_prompts(
    prepared_image: Image.Image,
    additional_prompt: str,
    face_confirmed: bool,
    face_count: int = 0,
    preserve_identity: bool = False,
    primary_face=None,
    allow_face_attributes: bool = True,
) -> Tuple[str, str]:
    """
    Build the main generation prompt and a face-free "background" prompt
    (img2img mode).

    Generation is driven by the style TRIGGER WORD(S) only — BLIP
    auto-captioning is disabled by default (Config.USE_CAPTION). The caption
    used to inject person-nouns ("a man...", "a person...") that the
    portrait-biased LoRA turned into phantom/duplicated faces; this was
    worst in tiled mode, where every tile re-rendered the caption and grew
    its own face. With it off, the model is never *told* to draw a person —
    the real face (when there is one) is carried by the img2img latents +
    the InstantID conditioning (IdentityNet kps + IP-Adapter embedding),
    not by words.

    The main prompt is:
      1. Style trigger word(s).
      2. (optional) BLIP caption — ONLY if Config.USE_CAPTION is True.
      3. (optional) Face attribute descriptors — ONLY in single-pass mode
         when a face is confirmed and the user opted OUT of identity
         preservation. NEVER added in tiled mode
         (allow_face_attributes=False), because one face's description
         leaking into several tiles re-introduces duplicates.
      4. User-supplied additional style instructions (last, so they win).

    The background prompt is the face-free variant (style trigger + user
    extra, plus the scrubbed caption when captioning is on). In tiled mode
    it conditions every tile that does NOT own a face, so the
    portrait-biased LoRA can't grow a face on background tiles.

    Returns:
        (main_prompt, bg_prompt)
    """
    parts = [_style_trigger(face_confirmed)]

    if face_confirmed and face_count == 1 and SOLO_PROMPT_ADDON:
        parts.append(SOLO_PROMPT_ADDON)

    # Optional content caption — OFF by default. Kept behind a flag so the
    # behavior is reversible, but disabled because it was the primary
    # phantom-face source (a person-noun in the positive prompt + the
    # portrait-trained LoRA paints a face). See Config.USE_CAPTION.
    scrubbed_caption = ""
    if USE_CAPTION:
        from utils import get_caption  # local import: only needed when on
        caption = ""
        try:
            caption = (get_caption(prepared_image) or "").strip()
        except Exception as e:
            print(f"[Generator] Captioning failed (non-fatal): {e}")
        scrubbed_caption = _scrub_person_tokens(caption)
        main_caption = caption if face_confirmed else scrubbed_caption
        if main_caption:
            parts.append(main_caption)

    # Face attribute enrichment.
    #
    # IMPORTANT: generic descriptors like "young woman" describe a
    # *category*, not a specific person — feeding them to SDXL invites a
    # brand-new plausible face that merely matches the words. With InstantID
    # active the actual identity travels through the embedding, so words are
    # unnecessary AND risky. We add them ONLY when (a) a face is confirmed,
    # (b) the user opted OUT of identity preservation, and (c) we are NOT
    # tiling (a per-face description must not leak across tiles and spawn
    # duplicates).
    if (
        allow_face_attributes
        and face_confirmed
        and AUTO_FACEID
        and not preserve_identity
    ):
        try:
            face_desc = _insight_prompt_enrichment(primary_face)
            if face_desc:
                parts.append(face_desc)
        except Exception as e:
            print(f"[Generator] Face attribute enrichment failed (non-fatal): {e}")

    # User's additional instructions come last so they can override
    extra = additional_prompt.strip() if additional_prompt else ""
    if extra:
        parts.append(extra)

    main_prompt = ", ".join(parts)

    # Background prompt: face-free style trigger (+ scrubbed caption if
    # captioning is on) + user extra. No "portrait", no attributes.
    bg_parts = [_style_trigger(False)]
    if scrubbed_caption:
        bg_parts.append(scrubbed_caption)
    if extra:
        bg_parts.append(extra)
    bg_prompt = ", ".join(bg_parts)

    return main_prompt, bg_prompt


def _boost_depth_for_faces(
    depth_image: Image.Image,
    face_bboxes: List[Tuple[int, int, int, int]],
    boost_factor: float = FACE_DEPTH_BOOST,
) -> Image.Image:
    """
    Increase depth values in face regions so ControlNet preserves
    more facial structure during generation.

    Uses a feathered ELLIPSE (not a filled rectangle) with a blur radius
    scaled to face size, so the boost fades gradually into the surrounding
    depth. A hard rectangular plateau here is what made the depth ControlNet
    render a boxy halo ("white square") around the face — worst on full-body
    shots, where a small, sharply-bounded face box sits against smooth body
    depth and the box edge gets drawn as a silhouette.

    Args:
        depth_image: The Zoe depth map.
        face_bboxes: List of (x1, y1, x2, y2) bboxes from insightface. Uses
                     the same trusted detections as the prompt-gating
                     step so we never boost depth on a phantom face.
        boost_factor: Multiplicative boost applied to depth values in
                      the face region.
    """
    if not face_bboxes:
        return depth_image

    depth_np = np.array(depth_image).astype(np.float32)
    h, w = depth_np.shape[:2]

    # One soft, rounded mask for ALL faces, then apply the boost once. The
    # ellipse has no corners and the size-scaled feather removes the hard
    # edge, so there is no rectangular plateau for ControlNet to render.
    mask = np.zeros((h, w), dtype=np.float32)
    max_feather = 0
    for (x1_f, y1_f, x2_f, y2_f) in face_bboxes:
        fw = max(1, x2_f - x1_f)
        fh = max(1, y2_f - y1_f)
        cx, cy = (x1_f + x2_f) // 2, (y1_f + y2_f) // 2
        # Ellipse axes a bit larger than the face core; soft edge added below.
        cv2.ellipse(
            mask, (cx, cy), (int(fw * 0.62), int(fh * 0.62)),
            0, 0, 360, 1.0, thickness=-1,
        )
        max_feather = max(max_feather, int(round(0.45 * min(fw, fh))))

    # Feather proportional to face size (min 15px) — removes the hard edge.
    feather = max(15, max_feather)
    mask = np.array(
        Image.fromarray((mask * 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(radius=feather))
    ).astype(np.float32) / 255.0

    if depth_np.ndim == 3:
        mask = np.stack([mask] * depth_np.shape[2], axis=-1)

    boosted = np.clip(depth_np * boost_factor, 0, 255)
    depth_np = depth_np * (1.0 - mask) + boosted * mask
    return Image.fromarray(np.clip(depth_np, 0, 255).astype(np.uint8))


def _expand_bboxes(
    face_bboxes: List[Tuple[int, int, int, int]],
    frac: float,
    w: int,
    h: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Expand each (x1, y1, x2, y2) bbox by `frac` of its own size per edge,
    clamped to the image bounds. Used by the tiled regional-prompt routing
    so a face straddling a tile edge is still owned by the right tile.
    """
    out: List[Tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in face_bboxes:
        fw, fh = x2 - x1, y2 - y1
        px, py = int(fw * frac), int(fh * frac)
        out.append((
            max(0, int(x1) - px),
            max(0, int(y1) - py),
            min(w, int(x2) + px),
            min(h, int(y2) + py),
        ))
    return out


def _snap8(v: float) -> int:
    """Snap a dimension to the nearest multiple of 8 (min 64) — the SDXL
    UNet/VAE requirement. Tile sizes and escalation rungs snap themselves
    the same way, and /8 keeps requested aspect ratios far more faithfully
    than a /64 snap would."""
    return max(64, int(round(float(v) / 8.0)) * 8)


def _budget_dims(resolution: int, ratio_w: float, ratio_h: float) -> Tuple[int, int]:
    """
    Compute output dimensions from a PIXEL BUDGET: total pixels ≈
    resolution² (e.g. 832 -> ~832x832 = ~0.69 MP) distributed over the
    requested aspect ratio, so every shape costs the same VRAM/runtime.
        w = resolution * sqrt(r),  h = resolution / sqrt(r),  r = w:h
    Both dimensions are snapped to multiples of 8.
    """
    r = float(ratio_w) / float(ratio_h)
    scale = r ** 0.5
    return _snap8(resolution * scale), _snap8(resolution / scale)


def _txt2img_dims(resolution: int, aspect_ratio: str) -> Tuple[int, int]:
    """
    Canvas size for TEXT-TO-IMAGE mode: a `resolution`² pixel budget shaped
    by the chosen aspect ratio (one of the keys of Config.ASPECT_RATIOS,
    spanning 2:1 wide to 1:2 tall). Unknown ratios fall back to the default.
    """
    if aspect_ratio not in ASPECT_RATIOS:
        print(
            f"[Generator] Unknown aspect ratio '{aspect_ratio}' — "
            f"falling back to {DEFAULT_ASPECT_RATIO}"
        )
        aspect_ratio = DEFAULT_ASPECT_RATIO
    rw, rh = ASPECT_RATIOS[aspect_ratio]
    return _budget_dims(resolution, rw, rh)


def _scale_bboxes(
    bboxes: Optional[List[Tuple[int, int, int, int]]],
    from_w: int,
    from_h: int,
    to_w: int,
    to_h: int,
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Rescale pixel face boxes from one image size to another (None passes
    through, so the tiler's regional routing stays off when there's no face)."""
    if not bboxes:
        return bboxes
    sx, sy = to_w / float(from_w), to_h / float(from_h)
    return [
        (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        )
        for (x1, y1, x2, y2) in bboxes
    ]


def _compute_escalation_rungs(
    base_w: int,
    base_h: int,
    target_w: int,
    target_h: int,
    factor: float,
    max_steps: int,
) -> List[Tuple[int, int]]:
    """
    Geometric resolution rungs climbing from the base size up to the target
    size (inclusive). Both dimensions grow together so aspect is preserved,
    each step is ~`factor`x on the long edge, and the FINAL rung is always
    exactly (target_w, target_h). Degrades to a single [target] rung when the
    target is at/below the base or escalation can't help.
    """
    base_long = max(base_w, base_h)
    target_long = max(target_w, target_h)
    if target_long <= base_long or factor <= 1.0:
        return [(target_w, target_h)]

    n = int(math.ceil(math.log(target_long / base_long) / math.log(factor)))
    n = max(1, min(int(max_steps), n))

    ratio_w = target_w / float(base_w)
    ratio_h = target_h / float(base_h)
    rungs: List[Tuple[int, int]] = []
    for i in range(1, n + 1):
        t = i / float(n)
        rungs.append((_snap8(base_w * (ratio_w ** t)), _snap8(base_h * (ratio_h ** t))))
    rungs[-1] = (target_w, target_h)  # land exactly on target

    dedup: List[Tuple[int, int]] = []
    for r in rungs:
        if not dedup or r != dedup[-1]:
            dedup.append(r)
    return dedup


def _coarse_to_fine(
    pipe,
    *,
    prepared_image: Image.Image,
    control_images: List[Image.Image],
    control_scales: List[float],
    prompt: str,
    negative_prompt: str,
    bg_prompt: Optional[str],
    bg_negative_prompt: Optional[str],
    face_bboxes_px: Optional[List[Tuple[int, int, int, int]]],
    target_w: int,
    target_h: int,
    num_inference_steps: int,
    guidance_scale: float,
    base_strength: float,
    refine_strength: float,
    base_long: int,
    tile_size: int,
    tile_overlap: int,
    generator: torch.Generator,
    clip_skip: int,
    image_embeds,
    ip_adapter_scale: float,
    detect_faces_on_base: bool = False,
) -> Image.Image:
    """
    Coarse-to-fine high-res generation — the "best of both worlds".

    PASS 1 (coarse): one SINGLE-PASS generation of the whole image at a
    model-native resolution (<= base_long). Because the model sees the entire
    image at once, the result is globally coherent in colour and structure
    and — being single-pass — phantom-free. InstantID conditions this pass
    exactly like a normal single-pass job (identity embedding at
    `ip_adapter_scale`, IdentityNet via its control map).

    The base is upscaled to the target resolution, then:

    PASS 2 (fine): image-space TILED img2img at LOW strength over the
    upscaled base. Tiles only add high-resolution detail and re-assert the
    pixel-art style; because their init is the coherent base and the denoise
    is gentle, they cannot drift in colour or invent faces. InstantID is
    routed PER TILE by tiled_controlnet_img2img: face tiles get the identity
    embedding at `ip_adapter_scale`, background tiles run at IP scale 0.
    You get tiled crispness with single-pass coherence.

    Finally (optional) the tile mosaic is colour-matched back to the coherent
    reference to erase any residual drift.

    Control maps are computed once at target resolution by the caller and are
    simply downscaled for each pass, so identity-kps/depth stay aligned
    throughout.

    When Config.USE_ESCALATION is True, PASS 2 is replaced by a geometric
    climb (base -> ... -> target): each rung upscales the previous coherent
    result by ~ESCALATION_FACTOR and refines it with a low-strength tiled
    pass, carrying more genuine detail upward while never leaving SDXL's
    comfort zone. With escalation off there is exactly one refine rung at the
    target, i.e. the plain two-pass behaviour.

    detect_faces_on_base: TEXT-TO-IMAGE support. In txt2img there is no
    input face, but the base pass may well GENERATE one (the prompt asked
    for it). When True and no face_bboxes_px were given, faces are detected
    on the base result and used for the refine rungs' regional routing —
    so the prompted portrait keeps its face tile while background tiles get
    the face-free bg_prompt.
    """
    # ---- Base size: preserve aspect, cap long edge at base_long, mult of 8.
    long_edge = max(target_w, target_h)
    scale = min(1.0, float(base_long) / float(long_edge))
    base_w = max(64, (int(round(target_w * scale)) // 8) * 8)
    base_h = max(64, (int(round(target_h * scale)) // 8) * 8)

    base_img = prepared_image.resize((base_w, base_h), Image.LANCZOS)
    base_controls = [c.resize((base_w, base_h), Image.LANCZOS) for c in control_images]

    # ---- PASS 1: coherent single-pass base ----
    # Full-image pass => identity applies globally, exactly like a normal
    # single-pass job (per-tile routing only makes sense once there ARE
    # tiles). The tiler restores this scale after each rung.
    if hasattr(pipe, "set_ip_adapter_scale"):
        pipe.set_ip_adapter_scale(float(ip_adapter_scale))
    base_seed = int(generator.initial_seed())
    g_base = torch.Generator(device="cuda").manual_seed(base_seed)
    with torch.inference_mode():
        base_res = pipe(
            image=base_img,
            control_image=base_controls,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=base_strength,
            controlnet_conditioning_scale=list(control_scales),
            clip_skip=clip_skip,
            generator=g_base,
            width=base_w,
            height=base_h,
            image_embeds=image_embeds,
        )
    base_out = base_res.images[0]
    del base_res
    print(f"[CoarseToFine] base pass done @ {base_w}x{base_h} (strength {base_strength:.2f})")
    torch.cuda.empty_cache()

    # ---- txt2img regional routing: faces exist only AFTER the base pass.
    if detect_faces_on_base and not face_bboxes_px and bg_prompt:
        try:
            base_faces = _detect_faces_insight(base_out)
            if base_faces:
                bboxes = _expand_bboxes(
                    _faces_to_bboxes(base_faces), FACE_BOX_DILATE, base_w, base_h
                )
                face_bboxes_px = _scale_bboxes(
                    bboxes, base_w, base_h, target_w, target_h
                )
                print(
                    f"[CoarseToFine] {len(face_bboxes_px)} face(s) detected on "
                    "the base — regional prompt routing active for refine rungs"
                )
        except Exception as e:
            print(f"[CoarseToFine] base face detection failed (non-fatal): {e}")

    # ---- Resolution rungs ----
    # Without escalation: a single rung at the target -> base is upscaled to
    # target and refined once (the original two-pass behaviour).
    # With escalation: a geometric climb base -> ... -> target, each rung a
    # low-strength tiled refine over the previous (coherent) rung, carrying
    # more genuine detail upward while never leaving SDXL's comfort zone.
    if USE_ESCALATION:
        rungs = _compute_escalation_rungs(
            base_w, base_h, target_w, target_h,
            ESCALATION_FACTOR, ESCALATION_MAX_STEPS,
        )
        print(f"[CoarseToFine] escalation ON — rungs: {rungs}")
    else:
        rungs = [(target_w, target_h)]

    current = base_out
    n_rungs = len(rungs)
    for ri, (rw, rh) in enumerate(rungs, 1):
        # Upscale the previous coherent result to this rung, and bring the
        # (target-res) control maps + face boxes down to the same size.
        up = current.resize((rw, rh), Image.LANCZOS)
        controls_r = [c.resize((rw, rh), Image.LANCZOS) for c in control_images]
        bboxes_r = _scale_bboxes(face_bboxes_px, target_w, target_h, rw, rh)
        # Anchor every rung's tiles to the ORIGINAL coherent base (resized to
        # this rung), not the previous rung — keeps tone globally consistent
        # without letting small shifts compound up the ladder. The detailed
        # init still comes from `up`, so detail compounds while tone stays put.
        ref_r = base_out.resize((rw, rh), Image.LANCZOS)

        current = tiled_controlnet_img2img(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=up,                            # init from the coherent rung
            control_images=controls_r,
            width=rw,
            height=rh,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=refine_strength,            # gentle: add detail, don't drift
            controlnet_conditioning_scales=control_scales,
            generator=generator,
            clip_skip=clip_skip,
            tile_size_px=tile_size,
            tile_overlap_px=tile_overlap,
            bg_prompt=bg_prompt,
            bg_negative_prompt=bg_negative_prompt,
            face_bboxes_px=bboxes_r,
            reference_image=ref_r,
            tile_color_match=TILE_COLOR_MATCH,
            # InstantID per-tile routing: identity only on face tiles.
            image_embeds=image_embeds,
            ip_adapter_scale=ip_adapter_scale,
        )
        torch.cuda.empty_cache()
        print(
            f"[CoarseToFine] refine rung {ri}/{n_rungs} @ {rw}x{rh} "
            f"(strength {refine_strength:.2f})"
        )

    refined = current

    # ---- Colour match the result back to a coherent reference ----
    if COLOR_MATCH:
        if COLOR_MATCH_REFERENCE == "base":
            ref = base_out.resize((target_w, target_h), Image.LANCZOS)
        else:
            ref = prepared_image
        refined = match_colors_lab(refined, ref, COLOR_MATCH_STRENGTH)
        print(
            f"[CoarseToFine] colour-matched to '{COLOR_MATCH_REFERENCE}' "
            f"(strength {COLOR_MATCH_STRENGTH:.2f})"
        )

    return refined


def generate_pixel_art(
    input_image: Optional[Image.Image] = None,
    additional_prompt: str = "",
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    seed: int = -1,
    lora_intensity: float = DEFAULT_LORA_INTENSITY,
    img2img_strength: float = IMG_STRENGTH,
    identity_preserve: bool = DEFAULT_FACE_PRESERVE_ENABLED,
    identitynet_strength: float = DEFAULT_IDENTITYNET_SCALE,
    ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE,
    resolution: int = DEFAULT_RESOLUTION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    use_tiled: bool = DEFAULT_USE_TILED,
    tile_size: int = TILE_SIZE,
    tile_overlap: int = TILE_OVERLAP,
) -> Tuple[Image.Image, int, Optional[str], str]:
    """
    Generate pixel art in one of two modes. Should be called within a GPU
    context (via the @spaces.GPU decorator).

    Jobs whose prepared canvas fits within BASE_PASS_LONG_EDGE run as a
    SINGLE pipeline pass. Larger jobs (or use_tiled=True) run the
    coarse-to-fine TILED path: a coherent single-pass base, then
    low-strength image-space tiled refinement rungs up to the target — see
    _coarse_to_fine. InstantID is preserved across both paths; in tiled
    mode the identity embedding is routed per tile (face tiles only).

    IMG2IMG (input_image given): the classic path. The input is resized to
    fill a `resolution`² pixel budget while preserving its own aspect ratio
    (`aspect_ratio` is ignored), depth + InstantID conditioning are built
    from it, and it is redrawn in the pixel-art style. Face identity (when
    enabled and a face is detected) is preserved DURING diffusion via
    InstantID: the IdentityNet ControlNet is conditioned on the real face's
    5-point keypoint map, and the IP-Adapter cross-attends the ArcFace
    identity embedding into the UNet.

    TEXT-TO-IMAGE (input_image is None): `additional_prompt` is MANDATORY
    and drives the content (ValueError if empty). There is no control image:
    the canvas is the `resolution`² pixel budget shaped by `aspect_ratio`,
    both ControlNets are fed inert black maps at conditioning scale 0, and
    the img2img pipeline denoises from (almost) pure noise at strength
    TXT2IMG_STRENGTH (1.0) — equivalent to txt2img. High-res txt2img always
    uses the coarse-to-fine path (independent high-strength tiles have no
    shared content to agree on when denoising from noise).

    Args:
        input_image: Input PIL Image, or None for text-to-image mode.
        additional_prompt: Style/content instructions. Optional in img2img
                           mode; REQUIRED in text-to-image mode.
        guidance_scale: Guidance scale for generation.
        num_inference_steps: Number of inference steps.
        seed: Random seed (-1 for random).
        identity_preserve: Enable InstantID identity conditioning (img2img).
        identitynet_strength: IdentityNet ControlNet conditioning scale
                              (facial structure/pose; 0-1.5).
        ip_adapter_scale: IP-Adapter scale (identity likeness; 0-1.5).
        resolution: Pixel-budget edge in px (<= MAX_RESOLUTION): the output
                    contains ~resolution² total pixels regardless of shape
                    (832 -> ~0.69 MP, like an 832x832 square).
        aspect_ratio: Canvas aspect ratio for text-to-image mode; one of
                      Config.ASPECT_RATIOS keys ("2:1" ... "1:2").
                      Ignored in img2img mode.
        use_tiled: Force the tiled high-res path on. Auto-enabled whenever
                   the prepared canvas exceeds BASE_PASS_LONG_EDGE in
                   either dimension.
        tile_size: Pixel tile size for tiled refinement (default 768).
        tile_overlap: Pixel overlap between adjacent tiles (default 192).

    Returns:
        Tuple of (generated image, seed used, face report or None, full prompt)
    """
    is_txt2img = input_image is None
    extra_prompt = (additional_prompt or "").strip()

    if is_txt2img and not extra_prompt:
        raise ValueError(
            "A prompt is required when generating without an input image."
        )

    # Clamp the requested resolution to the hard cap.
    resolution = max(512, min(int(resolution), MAX_RESOLUTION))

    # ---- Canvas / input preparation -----------------------------------
    if is_txt2img:
        # No control image exists in this mode: the canvas is the
        # resolution² pixel budget shaped by the aspect ratio, and the init
        # is a neutral grey whose content is irrelevant at strength 1.0.
        target_width, target_height = _txt2img_dims(resolution, aspect_ratio)
        prepared_image = Image.new(
            "RGB", (target_width, target_height), (127, 127, 127)
        )
        faces: List = []
        print(
            f"[Generator] TEXT-TO-IMAGE mode — {aspect_ratio} canvas, "
            "prompt-driven (no control image)"
        )
    else:
        # Fit the input into the same resolution² pixel budget, preserving
        # its own aspect ratio (the aspect_ratio arg is ignored). A 2:1
        # panorama and a 1:1 square therefore render with the same total
        # pixel count, VRAM use and runtime.
        src = input_image.convert("RGB")
        target_width, target_height = _budget_dims(
            resolution, src.width, src.height
        )
        prepared_image = src.resize(
            (target_width, target_height), Image.LANCZOS
        )

        # ---- Face detection gate ---------------------------------------
        # One insightface pass supplies EVERYTHING the pipeline needs:
        # high-confidence bboxes (prompt gate, depth boost, tile routing),
        # the primary face's 5-point kps (IdentityNet conditioning image),
        # and its ArcFace embedding (IP-Adapter identity injection). A false
        # positive here would corrupt the prompt AND inject a bogus
        # identity, so detections below FACE_DET_SCORE_MIN are discarded.
        faces = _detect_faces_insight(prepared_image)

    # Tiling gate: auto-enable whenever the prepared canvas exceeds the
    # single-pass comfort zone in either dimension. Up to
    # BASE_PASS_LONG_EDGE a single pass (with depth + InstantID pinning the
    # layout) is both faster and coherent, so the user toggle only matters
    # in that range.
    must_tile = (
        target_width > BASE_PASS_LONG_EDGE
        or target_height > BASE_PASS_LONG_EDGE
    )
    use_tiled = bool(use_tiled) or must_tile

    face_bboxes: List[Tuple[int, int, int, int]] = _faces_to_bboxes(faces)
    face_confirmed = len(face_bboxes) > 0
    primary_face = faces[0] if faces else None  # largest by bbox area

    if face_confirmed:
        print(
            f"[Generator] insightface confirmed {len(face_bboxes)} face(s), "
            f"primary det_score {float(primary_face.det_score):.2f}"
        )
    elif not is_txt2img:
        print("[Generator] No high-confidence face — treating as non-portrait")

    # InstantID activation: identity conditioning runs only when the user
    # asked for it AND a real face exists (never in txt2img mode, where no
    # input face can exist). Otherwise both identity levers are zeroed and
    # a zero embedding is passed (inert at scale 0).
    identity_on = bool(identity_preserve) and primary_face is not None
    identity_scale = float(identitynet_strength) if identity_on else 0.0
    effective_ip_scale = float(ip_adapter_scale) if identity_on else 0.0
    if primary_face is not None:
        image_embeds = primary_face.embedding
    else:
        image_embeds = np.zeros(512, dtype=np.float32)

    single_face = len(face_bboxes) == 1

    # ---- Prompts -------------------------------------------------------
    if is_txt2img:
        # The user's (mandatory) prompt defines the content, so it gets the
        # FULL trigger — including the LoRA token — and NO face gating: if
        # the user asks for a person, nothing should fight it.
        prompt = ", ".join([TRIGGER_WORD, extra_prompt])
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
        # Face-free variant for background tiles in the tiled refine: the
        # base pass may legitimately generate a face (the prompt asked for
        # one), but no OTHER tile should be told to draw a person too —
        # person nouns are scrubbed from the user's prompt for those tiles.
        bg_parts = [_style_trigger(False)]
        scrubbed_extra = _scrub_person_tokens(extra_prompt)
        if scrubbed_extra:
            bg_parts.append(scrubbed_extra)
        bg_prompt = ", ".join(bg_parts)
        bg_negative_prompt = DEFAULT_NEGATIVE_PROMPT + ", " + NOFACE_NEGATIVE_ADDON
    else:
        # When face_confirmed is False, the trigger drops "portrait" and no
        # face attributes are added — so SDXL won't be biased into
        # generating a face that wasn't there. When exactly one face is
        # confirmed, "solo" tokens are added — positive tokens act even at
        # weak LCM CFG, unlike the negative addons. In tiled mode, per-face
        # attribute descriptions are never injected: one face's description
        # leaking into several tiles is exactly what spawns duplicates.
        prompt, bg_prompt = _build_prompts(
            prepared_image, additional_prompt, face_confirmed,
            face_count=len(face_bboxes),
            preserve_identity=identity_preserve,
            primary_face=primary_face,
            allow_face_attributes=not use_tiled,
        )

        # Negative prompts.
        #   - no face confirmed: suppress faces everywhere (the portrait-
        #     biased LoRA must not invent one).
        #   - exactly one face confirmed: suppress DUPLICATE faces. Gated to
        #     single-face inputs so it never fights legitimate group photos.
        #   - background tiles (tiled mode): full face suppression — tiles
        #     outside every face box must not draw a person at all.
        # These are weak at LCM CFG ~1.2, which is why guidance is floored
        # below for face jobs.
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
        if not face_confirmed:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT + ", " + NOFACE_NEGATIVE_ADDON
            print("[Generator] No-face negative suppression active")
        elif single_face:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT + ", " + FACE_NEGATIVE_ADDON
            print("[Generator] Single-face anti-duplication negatives active")
        bg_negative_prompt = DEFAULT_NEGATIVE_PROMPT + ", " + NOFACE_NEGATIVE_ADDON

    print(f"[Generator] Prompt: {prompt[:120]}...")
    print(
        f"[Generator] Resolution {target_width}x{target_height} "
        f"(pixel budget {resolution}x{resolution}), "
        f"mode={'txt2img' if is_txt2img else 'img2img'}, tiled={use_tiled}"
    )

    # Guidance floor for face jobs (img2img only — face_confirmed is always
    # False in txt2img): at the LCM default (1.2) the negative prompt is
    # nearly inert, so the anti-duplication / face-suppression negatives
    # need real CFG to act. LCM handles up to ~2.0 fine.
    effective_guidance = float(guidance_scale)
    if face_confirmed and FACE_GUIDANCE_MIN > 0 and effective_guidance < FACE_GUIDANCE_MIN:
        effective_guidance = float(FACE_GUIDANCE_MIN)
        print(
            f"[Generator] Guidance raised {guidance_scale} -> "
            f"{effective_guidance} (face job: negatives need CFG to bite)"
        )

    # Create seed. The torch.Generator object is created inside the
    # attempt loop below so phantom-face retries can re-roll it.
    actual_seed = create_seed(seed)

    # Get the pipeline
    pipe = get_pipeline()

    # ---- Conditioning (control images + strengths) ----------------------
    if is_txt2img:
        # NO CONTROL IMAGE. The MultiControlNet still expects its
        # [identity, depth] pair, so feed inert all-black maps at
        # conditioning scale 0 (a guaranteed no-op) and denoise from pure
        # noise: strength 1.0 makes the img2img call equivalent to txt2img.
        kps_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        depth_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        control_scales = [0.0, 0.0]
        img_strength = float(TXT2IMG_STRENGTH)
        # No source image to stay close to — apply the base style
        # intensity as-is (no face/no-face multiplier).
        lora_mult = 1.0
        print(
            f"[Generator] txt2img: strength {img_strength:.2f}, "
            "ControlNets zeroed (no control image)"
        )
    else:
        zoe = get_zoe_detector()

        # Generate depth map (required by ControlNet). Zoe handles arbitrary
        # input sizes — we then resize the result to match the prepared image.
        depth_image = zoe(prepared_image)
        if depth_image.size != (target_width, target_height):
            depth_image = depth_image.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

        # Face-aware tuning. InstantID preserves identity during diffusion
        # (IdentityNet structure + IP-Adapter embedding), so we apply simple
        # per-context img2img multipliers to the caller's base strength:
        #   face    -> x IMG_FACE_MULTIPLIER   (1.0: full redraw, InstantID holds ID)
        #   no face -> x IMG_NOFACE_MULTIPLIER (0.5: stay close to the source)
        depth_strength = DEPTH_STRENGTH

        if face_confirmed:
            print("[Generator] Confirmed face: depth boost + full img2img (InstantID holds identity)")
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

        # InstantID structural conditioning: the 5-point keypoint map of the
        # primary face. This is the identity constraint depth can't provide.
        # When no face was confirmed the map is all-black and the IdentityNet
        # conditioning scale is 0 — a no-op.
        kps_image = _make_identity_kps(prepared_image, primary_face)
        if identity_on:
            print(
                f"[Generator] InstantID active — IdentityNet {identity_scale:.2f}, "
                f"IP-Adapter {effective_ip_scale:.2f}"
            )

        control_scales = [identity_scale, depth_strength]
        lora_mult = LORA_FACE_MULTIPLIER if face_confirmed else LORA_NOFACE_MULTIPLIER

    # Control-image list. ORDER MUST MATCH model.py: [identity, depth].
    control_images = [kps_image, depth_image]

    # Style intensity. `lora_intensity` is the API-exposed BASE value; in
    # img2img it is multiplied by a face/non-face factor (faces boosted
    # since InstantID holds their identity; non-faces attenuated to avoid
    # over-cooking); in txt2img it is applied as-is. Then clamp. The LoRA is
    # unfused so we set the scale live per request.
    lora_scale = max(0.0, min(LORA_INTENSITY_MAX, float(lora_intensity) * lora_mult))
    try:
        pipe.set_adapters(["retroart"], adapter_weights=[lora_scale])
        print(
            f"[Generator] LoRA: base={lora_intensity} x {lora_mult} "
            f"-> scale {lora_scale:.3f}"
        )
    except Exception as e:
        print(f"[Generator] Could not set LoRA scale ({e}); using loaded default")

    # Expanded face boxes drive the tiled regional-prompt routing: a face
    # straddling a tile boundary is still assigned to the tile that owns
    # its (expanded) center.
    expanded_bboxes = _expand_bboxes(
        face_bboxes, FACE_BOX_DILATE, target_width, target_height
    )

    # Free VRAM from preprocessing before the heavy diffusion step
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Run diffusion. Three paths:
    #   - tiled + TWO_PASS_REFINE (always for tiled txt2img): coherent
    #     single-pass base -> upscale -> low-strength image-space tiled
    #     refine (escalation rungs) -> colour match.
    #   - tiled only: single-pass image-space tiled img2img.
    #   - neither: one standard pipeline call.
    # In img2img mode this is wrapped in a phantom-face guard: if the output
    # contains MORE high-confidence faces than the input, re-roll the seed
    # and try again (the duplicate is a generation artifact, deterministic
    # for a given seed). Each retry costs a full generation — keep
    # PHANTOM_FACE_MAX_RETRIES small inside the ZeroGPU time window.
    # The guard is OFF in txt2img mode: there is no input face count to
    # compare against, and faces the user *prompted for* must never
    # trigger a re-roll.
    # ----------------------------------------------------------------
    input_face_count = len(face_bboxes)
    guard_active = (not is_txt2img) and PHANTOM_FACE_MAX_RETRIES > 0
    max_attempts = (1 + max(0, int(PHANTOM_FACE_MAX_RETRIES))) if guard_active else 1
    gen_notes: List[str] = []

    if hasattr(pipe, "set_ip_adapter_scale"):
        pipe.set_ip_adapter_scale(effective_ip_scale)

    output_image = None
    for attempt in range(max_attempts):
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)

        if use_tiled and (TWO_PASS_REFINE or is_txt2img):
            # Best of both worlds: a coherent single-pass base, upscaled,
            # then refined with low-strength image-space tiles (escalation
            # rungs). Colour matching is applied inside the helper.
            # txt2img ALWAYS takes this path when tiled: independent
            # high-strength tiles denoised from noise have no shared
            # content to agree on.
            output_image = _coarse_to_fine(
                pipe,
                prepared_image=prepared_image,
                control_images=control_images,
                control_scales=control_scales,
                prompt=prompt,
                negative_prompt=negative_prompt,
                bg_prompt=bg_prompt if (face_confirmed or is_txt2img) else None,
                bg_negative_prompt=(
                    bg_negative_prompt if (face_confirmed or is_txt2img) else None
                ),
                face_bboxes_px=expanded_bboxes if face_confirmed else None,
                target_w=target_width,
                target_h=target_height,
                num_inference_steps=num_inference_steps,
                guidance_scale=effective_guidance,
                base_strength=img_strength,
                refine_strength=REFINE_STRENGTH,
                base_long=BASE_PASS_LONG_EDGE,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                generator=generator,
                clip_skip=CLIP_SKIP,
                image_embeds=image_embeds,
                ip_adapter_scale=effective_ip_scale,
                detect_faces_on_base=is_txt2img,
            )
        elif use_tiled:
            # Single-pass tiled path (high strength, no base). Each tile is a
            # full img2img on real cropped pixels.
            output_image = tiled_controlnet_img2img(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=prepared_image,
                control_images=control_images,
                width=target_width,
                height=target_height,
                num_inference_steps=num_inference_steps,
                guidance_scale=effective_guidance,
                strength=img_strength,
                controlnet_conditioning_scales=control_scales,
                generator=generator,
                clip_skip=CLIP_SKIP,
                tile_size_px=tile_size,
                tile_overlap_px=tile_overlap,
                # Regional prompting: face-free tiles get the scrubbed
                # background prompt + full face suppression, so the
                # portrait prompt can't make every tile grow a face.
                bg_prompt=bg_prompt if face_confirmed else None,
                bg_negative_prompt=bg_negative_prompt if face_confirmed else None,
                face_bboxes_px=expanded_bboxes if face_confirmed else None,
                # No coherent pixel-art base in this path, so anchor tile tone
                # to the input's local tone — still a single coherent source,
                # so tiles stay mutually consistent and seam-free.
                reference_image=prepared_image,
                tile_color_match=TILE_COLOR_MATCH,
                # InstantID per-tile routing: identity only on face tiles.
                image_embeds=image_embeds,
                ip_adapter_scale=effective_ip_scale,
            )
            # No coherent base here, so the only global reference for colour
            # matching is the input. Off by default (reference="base"); set
            # COLOR_MATCH_REFERENCE="input" to unify tile tone toward the
            # source photo.
            if COLOR_MATCH and COLOR_MATCH_REFERENCE == "input":
                output_image = match_colors_lab(
                    output_image, prepared_image, COLOR_MATCH_STRENGTH
                )
        else:
            with torch.inference_mode():
                result = pipe(
                    image=prepared_image,
                    control_image=control_images,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=effective_guidance,
                    num_inference_steps=num_inference_steps,
                    strength=img_strength,
                    controlnet_conditioning_scale=control_scales,
                    clip_skip=CLIP_SKIP,
                    generator=generator,
                    width=target_width,
                    height=target_height,
                    image_embeds=image_embeds,
                )
            output_image = result.images[0]

        if not guard_active:
            break

        # Count high-confidence faces in the output. The bar
        # (PHANTOM_FACE_MIN_SCORE) is deliberately above the input gate:
        # SCRFD can hallucinate on stylised textures, and phantom faces
        # tend to be the realistic ones it does catch.
        out_count = len(
            _detect_faces_insight(output_image, min_score=PHANTOM_FACE_MIN_SCORE)
        )
        if out_count <= input_face_count:
            if attempt > 0:
                gen_notes.append(
                    f"Phantom-face guard: clean output after {attempt} "
                    f"retry(ies), final seed {actual_seed}."
                )
            break

        print(
            f"[Generator] Phantom-face guard: {out_count} face(s) in output "
            f"vs {input_face_count} in input (attempt {attempt + 1}/{max_attempts})"
        )
        if attempt < max_attempts - 1:
            actual_seed = create_seed(-1)
            print(f"[Generator] Re-rolling seed -> {actual_seed}")
        else:
            gen_notes.append(
                f"Phantom-face guard: output still has {out_count} face(s) vs "
                f"{input_face_count} in input after {PHANTOM_FACE_MAX_RETRIES} "
                f"retry(ies) — returning last result."
            )

    # NOTE: the output is returned at the generation resolution — resizing
    # back to the input size silently softened pixel-art crispness.

    if use_tiled:
        gen_notes.insert(
            0,
            "High-res tiled mode: "
            + (
                "coarse-to-fine (coherent base + escalation refine rungs)"
                if (TWO_PASS_REFINE or is_txt2img)
                else "single-pass tiles"
            )
            + f", tile {int(tile_size)}px / overlap {int(tile_overlap)}px.",
        )

    # Identity was preserved DURING diffusion (InstantID) — there is no
    # post-generation blend step anymore. Just report what ran.
    face_report = None
    if identity_on:
        face_report = (
            f"InstantID: {len(face_bboxes)} face(s) detected "
            f"(primary det_score {float(primary_face.det_score):.2f}); "
            f"IdentityNet scale {identity_scale:.2f}, "
            f"IP-Adapter scale {effective_ip_scale:.2f}"
            + (" — routed per tile (face tiles only)." if use_tiled else ".")
        )
    elif identity_preserve and not face_confirmed and not is_txt2img:
        face_report = (
            "InstantID: enabled but no high-confidence face detected — "
            "ran style-only (identity conditioning inert)."
        )

    if gen_notes:
        face_report = "\n".join(gen_notes + ([face_report] if face_report else []))

    return output_image, actual_seed, face_report, prompt
