"""
Pixagram Pixel Art Generator - Configuration
With Prompt-Based Face Preservation

License: All components use commercially-permissive licenses:
- OpenCV: BSD License
- MediaPipe: Apache 2.0
- Custom modules: Your proprietary IP
"""
import torch


class Config:
    # ============================================================
    # APP METADATA
    # ============================================================
    TITLE = "Pixagram.io - Image to Pixel Art"
    DESCRIPTION = "Transform any image into retro pixel art style with face preservation"
    ARTICLE = "..."
    VERSION = "2.0.0"

    # ============================================================
    # HARDWARE CONFIGURATION
    # ============================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    # ============================================================
    # BASE MODEL & LORA (from primerz/pixagram)
    # ============================================================
    REPO_ID = "primerz/pixagram"
    CHECKPOINT_FILENAME = "horizon.safetensors"
    LORA_FILENAME = "retroart.safetensors"
    LORA_STRENGTH = 1.25  # Fixed strength for fusion
    DEFAULT_GUIDANCE_SCALE = 1.2
    DEFAULT_NUM_INFERENCE_STEPS = 10
    DEFAULT_SEED = 42

    # Trigger Words for the LoRA
    STYLE_TRIGGER1 = "HD detailed pixel art artwork and high quality detailed art illustration in retroart style"
    STYLE_TRIGGER2 = "artwork, illustration, pixel art, retro game art portrait, art."

    # Combined trigger word used by the generation pipeline
    TRIGGER_WORD = STYLE_TRIGGER1 + ", " + STYLE_TRIGGER2

    # Default Negative Prompt
    DEFAULT_NEGATIVE_PROMPT = "Ugly, artifacts, blurry, disformed, photo-realistic, photo, photography, realistic, low-quality, text, white edges, white border, pixel art, squares, crisp-edge, squared."

    # ============================================================
    # CONTROLNET CONFIGURATION
    # ============================================================
    CN_ZOE_REPO = "diffusers/controlnet-zoe-depth-sdxl-1.0"

    # Preprocessor (Annotator) Repo
    ANNOTATOR_REPO = "lllyasviel/Annotators"

    # ============================================================
    # CAPTIONING MODEL
    # ============================================================
    CAPTIONER_REPO = "Salesforce/blip-image-captioning-base"

    # ============================================================
    # FACE ATTRIBUTE ANALYSIS (Commercial-Friendly)
    # ============================================================
    # All face analysis uses OpenCV (BSD) and MediaPipe (Apache 2.0).
    # No InsightFace, no ONNX recognition models.

    # Face detection settings
    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_DETECTION_PADDING = 0.3

    # Whether to auto-detect and enrich prompts with face attributes
    AUTO_FACEID = True

    # How strongly to weight face attribute descriptors in the prompt.
    DEFAULT_FACEID_STRENGTH = 0.85

    # Depth boost: multiply ControlNet depth in face region
    FACE_DEPTH_BOOST = 1.4

    # Reduce img2img strength when face detected (preserves more detail).
    # Raised from 0.12 to 0.22 — at 0.12 the effective strength stayed
    # at 0.53, which still gave the LoRA enough latitude to redraw
    # faces entirely. 0.22 brings strength to 0.43 which preserves
    # noticeably more of the original face structure.
    FACE_STRENGTH_REDUCTION = 0.22

    # Minimum similarity score (0-1) between input and output face
    FACE_SIMILARITY_THRESHOLD = 0.6

    # Blend strength for face region compositing (0-1).
    # Raised from 0.35 to 0.65 — at 0.35 the LoRA's stylization was
    # winning even on confirmed drifted faces. 0.65 gives noticeable
    # identity restoration without erasing the pixel-art treatment of
    # the rest of the head.
    FACE_BLEND_STRENGTH = 0.65

    # Face region expansion for blending (fraction of face size)
    FACE_BLEND_PADDING = 0.3

    # Whether to enable post-generation face blending
    FACE_BLEND_ENABLED = True

    # Whether face preservation is enabled by default in the UI
    DEFAULT_FACE_PRESERVE_ENABLED = True

    # Per-edge padding (PIXELS) added to the face bbox before masking.
    # Used to be misread as a multiplier — see _create_soft_face_mask.
    # Keep this small relative to face size; the Gaussian blur below
    # handles softening the edges.
    FACE_MASK_EXPAND = 16

    # Gaussian blur radius for face mask edges (pixels). Larger = softer
    # transition between the blended face and the surrounding pixel art.
    FACE_MASK_BLUR = 24

    # Number of retry attempts if face similarity is too low
    FACE_MAX_RETRIES = 3

    # Whether to use Poisson blending (seamless clone) vs alpha blend
    FACE_POISSON_BLEND = True

    # Minimum face size (pixels) to trigger preservation
    FACE_MIN_SIZE = 64

    # Face color correction strength (match original face colors)
    FACE_COLOR_CORRECTION = 0.5

    # ============================================================
    # GENERATION DEFAULTS
    # ============================================================
    CGF_SCALE = 1.2
    STEPS_NUMBER = 10
    IMG_STRENGTH = 0.65
    DEPTH_STRENGTH = 0.75
    CLIP_SKIP = 2

    # ============================================================
    # TILED MULTI-DIFFUSION (high-resolution generation)
    # ============================================================
    # When the prepared input is larger than the tile size, generation
    # is performed via overlapping-tile denoising (MultiDiffusion).
    # Each tile shares the same prompt and conditioning, and noise
    # predictions are blended in overlap regions every step.

    # Hard cap on the longer side of the prepared image (pixels).
    # 3840 = 4K UHD width. Set lower if VRAM is tight.
    MAX_RESOLUTION = 3840

    # Default "low-res" mode size — matches the previous ~1.6MP behavior.
    DEFAULT_RESOLUTION = 1280

    # Tile size in pixels (must be a multiple of 64 — VAE downsamples 8x
    # and the UNet needs latent dims divisible by 8).
    TILE_SIZE = 768

    # Pixel overlap between adjacent tiles.  ~25% of TILE_SIZE is a good
    # balance between coherence and speed.
    TILE_OVERLAP = 192

    # If True and the prepared image is larger than TILE_SIZE in either
    # dimension, use the tiled denoiser. If False, the standard pipeline
    # is used and the image is downscaled to fit in one canvas.
    DEFAULT_USE_TILED = False


print("[OK] Config loaded - Prompt-based face preservation (no InsightFace)")

# Auto-export all Config class attributes at module level
# so both `Config.X` and `from config import X` work.
import sys as _sys
_module = _sys.modules[__name__]
for _name in dir(Config):
    if not _name.startswith("_"):
        setattr(_module, _name, getattr(Config, _name))