"""
Pixagram Pixel Art Generator - Configuration
With InstantID Face Identity (IdentityNet + IP-Adapter)

License note: InstantID code/weights are Apache-2.0, but its face
detection + embedding stack uses InsightFace's antelopev2 models, which
are released for NON-COMMERCIAL research use. This deliberately reverses
this module's previous "no InsightFace" stance — clear commercial use
with counsel before shipping.
"""
import os
import torch

# /data is the HF Spaces persistent-storage mount — it only exists when
# persistent storage is enabled (never on ZeroGPU). Fall back to a local
# folder; models are then simply re-downloaded on each restart.
if os.path.isdir("/data") and os.access("/data", os.W_OK):
    _DATA_ROOT = "/data"
else:
    _DATA_ROOT = os.path.join(os.getcwd(), "data")
os.makedirs(_DATA_ROOT, exist_ok=True)


class Config:
    # ============================================================
    # APP METADATA
    # ============================================================
    TITLE = "Pixagram.com - Image to Pixel Art"
    DESCRIPTION = (
        "Transform any image into retro pixel art style with face preservation "
        "— or leave the image empty and generate pixel art from a text prompt."
    )
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
    LORA_STRENGTH = 1.25  # DEPRECATED: see DEFAULT_LORA_INTENSITY. No longer
                          # fused. Kept only for back-compat.

    # Runtime LoRA style intensity. The LoRA is loaded UNFUSED and the
    # generator sets its scale live per request. DEFAULT_LORA_INTENSITY is the
    # BASE value exposed through the API/UI; the generator multiplies it by a
    # face/non-face factor:
    #     effective = clamp(lora_intensity * (FACE_MULT if face else NOFACE_MULT))
    # Faces are boosted (InstantID's IdentityNet + IP-Adapter hold identity
    # during diffusion, so more style is safe and adds pixel-art punch);
    # non-faces are attenuated to avoid the over-cooked / artifacted look.
    DEFAULT_LORA_INTENSITY = 0.75
    LORA_FACE_MULTIPLIER = 1.2
    LORA_NOFACE_MULTIPLIER = 0.6
    # Hard clamp on the effective scale after multiplication.
    LORA_INTENSITY_MAX = 2.0

    # img2img strength multipliers. InstantID preserves identity DURING
    # diffusion (IdentityNet structure + IP-Adapter embedding), so faces no
    # longer need a starved strength to keep identity — they run at full
    # (x1.0). Non-faces have no identity lock, so we halve their redraw
    # (x0.5) to stay close to the source and avoid the over-transformed look.
    IMG_FACE_MULTIPLIER = 1.0
    IMG_NOFACE_MULTIPLIER = 0.5
    # Clamp range for the effective img2img strength after multiplication.
    IMG_STRENGTH_MIN = 0.1
    IMG_STRENGTH_MAX = 1.0

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

    # Extra negative-prompt tokens appended ONLY when insightface confirmed NO
    # face. Actively suppresses the portrait-biased retroart LoRA from
    # inventing a face in faceless images. NOTE: with LCM at guidance ~1.2
    # CFG is weak, so this is a *secondary* lever — the primary phantom-face
    # fix is keeping face nouns out of the POSITIVE prompt (see generator.py's
    # _scrub_person_tokens). Raising guidance for the no-face branch gives
    # this addon more teeth.
    NOFACE_NEGATIVE_ADDON = (
        "face, human face, person, people, portrait, eyes, facial features, "
        "man, woman, child, human, figure"
    )

    # Extra negative-prompt tokens appended when EXACTLY ONE face was
    # confirmed. Suppresses the duplicate-face failure mode where the
    # portrait-loaded prompt + portrait-biased LoRA paint a second face
    # onto hair/shoulders/background blobs. Gated to single-face inputs
    # so it never fights legitimate group photos. Weak at LCM CFG ~1.2,
    # which is why FACE_GUIDANCE_MIN below raises guidance on face jobs.
    FACE_NEGATIVE_ADDON = (
        "multiple faces, extra face, second face, duplicated face, "
        "second person, twins, clone, crowd"
    )

    # Positive-prompt tokens appended when exactly ONE face was confirmed.
    # Unlike the negative addon, positive tokens act even at weak CFG —
    # this is the cheap anti-duplication lever.
    SOLO_PROMPT_ADDON = ""

    # Guidance floor for confirmed-face jobs. At the 1.2 default the
    # negative prompt is nearly inert, so the anti-duplication negatives
    # need real CFG to act. LCM handles up to ~2.0 fine. Set 0 to disable.
    FACE_GUIDANCE_MIN = 0

    # ============================================================
    # CONTROLNET CONFIGURATION
    # ============================================================
    CN_ZOE_REPO = "diffusers/controlnet-zoe-depth-sdxl-1.0"

    # ============================================================
    # INSTANTID (face identity — replaces the masked-canny face lock
    # AND the FaceNet post-blend)
    # ============================================================
    # Identity is injected DURING diffusion by two coupled components:
    #   1. IdentityNet — a ControlNet conditioned on the 5-point facial
    #      keypoint map (structure/pose of the real face), and
    #   2. an IP-Adapter that cross-attends the ArcFace identity
    #      embedding into the UNet (appearance/likeness).
    # The masked canny forced literal photographic edges into the
    # stylization and fought the LoRA (its scale had been tuned down to
    # 0.1); IdentityNet's 5 keypoints are a far softer structural prior,
    # with likeness carried by the embedding instead of edges.
    INSTANTID_REPO = "InstantX/InstantID"
    ANTELOPE_REPO = "DIAMONIK7777/antelopev2"

    # IdentityNet ControlNet conditioning scale (facial structure).
    # Raise for stricter pose/feature placement, lower if faces come out
    # under-stylized.
    DEFAULT_IDENTITYNET_SCALE = 0.8

    # IP-Adapter scale (identity-embedding strength in cross-attention).
    # This is the likeness knob. Raise if the person isn't recognizable,
    # lower if outputs look too photographic for the pixel-art style.
    DEFAULT_IP_ADAPTER_SCALE = 0.8

    # Minimum SCRFD detection score to treat a detection as a real face.
    # Replaces the old MTCNN 0.985 bar — different scale entirely:
    # genuine frontal faces typically score 0.70-0.90 under SCRFD.
    FACE_DET_SCORE_MIN = 0.65

    # Bar for counting faces in the OUTPUT (phantom-face guard).
    PHANTOM_FACE_MIN_SCORE = 0.75

    # insightface model root (models land in <root>/models/antelopev2).
    DATA_ROOT = _DATA_ROOT
    INSIGHTFACE_ROOT = _DATA_ROOT

    # Per-edge expansion of the face bbox (fraction of face size). Drives
    # the depth-boost mask.
    FACE_BOX_DILATE = 0.25

    # Preprocessor (Annotator) Repo
    ANNOTATOR_REPO = "lllyasviel/Annotators"

    # ============================================================
    # CAPTIONING MODEL
    # ============================================================
    # BLIP auto-captioning is DISABLED by default. The caption ("a man
    # standing...", "a person...", "a close up of a woman") was the primary
    # driver of phantom / duplicated faces: it injected person-nouns into the
    # POSITIVE prompt, and the portrait-biased retroart LoRA turned them into
    # faces. This was worst in tiled mode, where every tile re-rendered that
    # same caption and grew its own face. With captioning off, generation is
    # driven purely by the LoRA style trigger word(s) + the img2img latents —
    # which is safer and is exactly what high-res tiling needs. Set to True
    # only if you specifically want content-aware prompts back.
    USE_CAPTION = False
    CAPTIONER_REPO = "Salesforce/blip-image-captioning-base"

    # ============================================================
    # FACE HANDLING (InstantID era)
    # ============================================================
    # Detection, keypoints, and identity embeddings all come from
    # insightface antelopev2 (SCRFD detector + ArcFace recognition).
    # The old OpenCV-Haar attribute heuristics and the facenet-pytorch
    # post-generation blend are gone — identity is preserved DURING
    # diffusion by IdentityNet + IP-Adapter (see the INSTANTID section).

    # Whether to enrich prompts with coarse face descriptors (from
    # insightface gender/age) when identity preservation is OFF.
    AUTO_FACEID = True

    # Depth boost: multiply ControlNet depth in face region
    FACE_DEPTH_BOOST = 1.25

    # Whether InstantID identity preservation is enabled by default in the UI
    DEFAULT_FACE_PRESERVE_ENABLED = True

    # Phantom-face guard: after generation, faces are re-counted in the
    # output (insightface @ PHANTOM_FACE_MIN_SCORE). If the output has
    # MORE high-confidence faces than the input, the job is re-run with
    # a new seed, up to this many extra attempts. Each retry costs a
    # full generation — keep it small inside the ZeroGPU time window.
    # With InstantID conditioning phantoms are much rarer; 0 disables.
    PHANTOM_FACE_MAX_RETRIES = 1

    # ============================================================
    # GENERATION DEFAULTS
    # ============================================================
    CGF_SCALE = 1.2
    STEPS_NUMBER = 10
    IMG_STRENGTH = 0.65
    DEPTH_STRENGTH = 0.75
    CLIP_SKIP = 2

    # ============================================================
    # RESOLUTION & ASPECT RATIO
    # ============================================================
    # `resolution` is a PIXEL BUDGET, not an edge length: the output always
    # contains ~resolution² total pixels, shaped by the aspect ratio
    # (dropdown in txt2img, the upload's own ratio in img2img). 832 means
    # ~832x832 = ~0.69 MP whatever the shape — e.g. 16:9 -> 1112x624,
    # 2:1 -> 1176x592 — so every ratio costs the same VRAM and runtime.
    #
    # SDXL only stays coherent near its ~1 MP training area. Below
    # BASE_PASS_LONG_EDGE the job runs as a SINGLE pipeline pass; above it,
    # the coarse-to-fine TILED path takes over (a coherent base pass, then
    # low-strength tiled refinement rungs up to the target — see the tiling
    # section below), so high budgets add real detail instead of the
    # duplicated subjects a big single pass produces.
    #
    # 2880² ≈ 8.3 MP — the same total pixel count as 4K UHD (3840x2160).
    # Raise further only with generous ZeroGPU quota: cost grows ~linearly
    # with total pixels across the escalation rungs.
    MAX_RESOLUTION = 2880

    # Default pixel-budget edge (px): ~832x832 total pixels per output.
    DEFAULT_RESOLUTION = 832

    # Aspect ratios selectable in TEXT-TO-IMAGE mode (no input image).
    # In img2img mode the input image's own aspect ratio is always kept.
    # Spans 2:1 (wide) to 1:2 (tall) through all the common ratios.
    ASPECT_RATIOS = {
        "2:1":  (2, 1),
        "16:9": (16, 9),
        "3:2":  (3, 2),
        "4:3":  (4, 3),
        "5:4":  (5, 4),
        "1:1":  (1, 1),
        "4:5":  (4, 5),
        "3:4":  (3, 4),
        "2:3":  (2, 3),
        "9:16": (9, 16),
        "1:2":  (1, 2),
    }
    DEFAULT_ASPECT_RATIO = "1:1"

    # Denoise strength used in TEXT-TO-IMAGE mode. At 1.0 the img2img
    # pipeline denoises from (almost) pure noise — the neutral init image
    # contributes ~nothing at the max timestep — so it behaves as txt2img.
    TXT2IMG_STRENGTH = 1.0

    # ============================================================
    # TILED HIGH-RESOLUTION GENERATION (image-space tiles)
    # ============================================================
    # Above the single-pass comfort zone, generation runs on overlapping
    # PIXEL tiles: each tile is a full img2img on real cropped pixels (with
    # its identity-kps + depth control crops), then tiles are feathered back
    # together. See tiled_pipeline.py. InstantID routes PER TILE: the
    # identity embedding is injected only into tiles that own a face; every
    # background tile runs at IP scale 0 (an identity embedding attended
    # into a faceless tile is phantom-face pressure).

    # Tile size in pixels (multiple of 64 — VAE downsamples 8x and the
    # UNet needs latent dims divisible by 8). SDXL is most coherent when
    # each tile stays near its native ~1024 window.
    TILE_SIZE = 768

    # Pixel overlap between adjacent tiles. ~25% of TILE_SIZE is a good
    # balance between seam quality and speed.
    TILE_OVERLAP = 192

    # If True, the tiled path can be forced on for jobs that would fit a
    # single pass. Tiling always auto-activates when the prepared image
    # exceeds BASE_PASS_LONG_EDGE in either dimension.
    DEFAULT_USE_TILED = False

    # ============================================================
    # HIGH-RES STRATEGY: COARSE-TO-FINE (best of both worlds)
    # ============================================================
    # Pure image-space tiling gives crisp per-tile detail but, because each
    # tile is an independent generation, tiles can drift slightly in
    # hue/exposure -> a faint patchwork on big flat areas. A big single pass
    # keeps global colour coherent but, past ~1 MP, SDXL's attention starts
    # "seeing" multiple scenes and duplicates subjects/faces.
    #
    # Coarse-to-fine takes the best of both:
    #   1. BASE PASS   - one single-pass generation of the WHOLE image at a
    #      model-native resolution. The model sees everything at once, so the
    #      result is globally coherent, correctly "understood", and phantom-
    #      free (single-pass already behaves with the trigger-only prompt).
    #   2. UPSCALE that base to the target resolution.
    #   3. REFINE PASS - image-space tiled img2img at LOW strength over the
    #      upscaled base. Tiles only ADD high-res detail; they stay anchored
    #      to the base's global colour and structure, so no patchwork and no
    #      new faces - but you still get the per-tile crispness of
    #      image-space tiling.
    # Applies only when tiling is active. Set False to use the single-pass
    # (high-strength) tiled path instead. NOTE: text-to-image jobs always
    # use coarse-to-fine when tiled — independent high-strength tiles have
    # no shared content to agree on in that mode.
    TWO_PASS_REFINE = True

    # Long edge (px) of the coherent base pass — and the auto-tiling
    # threshold. This pass exists to nail a globally COHERENT composition,
    # and SDXL only stays coherent near its ~1 MP training resolution.
    # Pushed too high in a single pass, SDXL's attention (calibrated to
    # ~1024) starts "seeing" multiple scenes and DUPLICATES subjects —
    # repeated objects, and exactly the extra/duplicated faces this whole
    # pipeline works to avoid. So the base deliberately stays in SDXL's
    # comfort zone; the *refine* rungs are where real resolution is added
    # (safely, because each tile is only ~TILE_SIZE and so stays
    # in-distribution).
    #
    # The depth ControlNet, the InstantID conditioning and the img2img init
    # all PIN the layout, which raises the safe ceiling above naked txt2img:
    #   ~1024  - safest, classic SDXL sweet spot
    #   ~1280  - usually safe with ControlNet (default here)
    #   ~1536  - often OK with strong conditioning; watch for repetition
    #   >=2048 - duplication / incoherence likely returns; not recommended
    BASE_PASS_LONG_EDGE = 1280

    # Denoise strength for the refine tiles. This is the PRIMARY detail
    # lever: effective per-tile steps ~= num_inference_steps *
    # REFINE_STRENGTH, so at 0.5 each tile actually redraws genuine
    # high-frequency detail at full resolution. The depth + InstantID
    # conditioning pins the layout and the colour matching below re-unifies
    # tile tone, so this stays safe from drift/patchwork. Drop to ~0.40 if
    # you see tile seams or hue patchwork on big flat areas; raise toward
    # ~0.60 for even more detail at the cost of more drift risk.
    REFINE_STRENGTH = 0.5

    # ---- Colour matching (insurance against tile drift) -----------------
    # After the refine pass, transfer the tile mosaic's colour statistics
    # back onto a coherent reference (Reinhard mean/std in LAB). With
    # two-pass this is a light touch; for pure tiling it is the main
    # patchwork fix. Reference "base" = the upscaled base pass (preserves
    # the pixel-art palette); "input" = the original image (pulls colours
    # toward the source photo).
    COLOR_MATCH = True
    COLOR_MATCH_STRENGTH = 0.5
    COLOR_MATCH_REFERENCE = "base"   # "base" | "input"

    # ---- Per-tile local colour anchoring (the seam fix) -----------------
    # The post-assembly COLOR_MATCH above matches the WHOLE mosaic to one
    # global reference, which CANNOT fix LOCAL tile-to-tile divergence
    # (tile A warm vs tile B cool both get the same global shift, so they
    # stay mismatched and the feather just ramps between them -> patchwork).
    # This instead matches EACH tile to the same region of the coherent
    # reference (the upscaled base; the input for single-pass tiling)
    # BEFORE stitching, so neighbouring tiles already agree in tone at
    # their seam. It is an affine per-channel LAB match, so it aligns tone
    # WITHOUT touching the tile's high-frequency detail.
    # 0 = off; ~0.8 = strong tone lock; 1.0 = full lock. Lower toward 0.6
    # if areas look tonally flattened; raise toward 1.0 if any seam shows.
    TILE_COLOR_MATCH = 0.8

    # ---- Progressive escalation (THE high-res detail path) --------------
    # Instead of one big jump (base -> LANCZOS-upscale straight to target ->
    # a single refine), climb the resolution in geometric rungs, running a
    # tiled refine at each. Every rung upscales the previous
    # (already-coherent) result by only ~ESCALATION_FACTOR and adds genuine
    # detail while staying in-distribution. A 1.5x upscale is far less soft
    # than a 3-4x one, so the refine actually RECOVERS detail and it
    # COMPOUNDS rung over rung — real detail at high budgets instead of a
    # stretched 1280px image. Requires TWO_PASS_REFINE=True.
    #
    # Cost: one full tiled pass PER rung — genuinely heavy at the top
    # budgets. app.py scales the @spaces.GPU duration with the job. If you
    # hit the ZeroGPU time/quota wall, target a ~2048 budget instead of the
    # max, or lower ESCALATION_MAX_STEPS.
    USE_ESCALATION = True

    # Approx. resolution multiplier per rung (long edge). ~1.5 keeps each
    # upscale small so the refine has an easy job; 2.0 = fewer, bigger jumps.
    ESCALATION_FACTOR = 1.5

    # Hard cap on the number of refine rungs (time/VRAM budget). The climb
    # always ends exactly at the target resolution, even if this cap forces
    # a larger final jump. Each rung reuses REFINE_STRENGTH.
    ESCALATION_MAX_STEPS = 4


print("[OK] Config loaded - InstantID face identity (IdentityNet + IP-Adapter, insightface antelopev2)")

# Auto-export all Config class attributes at module level
# so both `Config.X` and `from config import X` work.
import sys as _sys
_module = _sys.modules[__name__]
for _name in dir(Config):
    if not _name.startswith("_"):
        setattr(_module, _name, getattr(Config, _name))