"""
Pixagram Pixel Art Converter - Hugging Face Space Application

Transform regular images into pixel art using SDXL + Pixagram LoRA, with
InstantID face identity (IdentityNet + IP-Adapter) preserved during diffusion.
Optimized for ZeroGPU (Hugging Face Spaces serverless GPU).
"""

# boot_guard MUST be the very first import. It places the HF cache on /data
# when persistent storage exists, sweeps stale download locks, and provides
# the self-healing loaders model.py uses. gradio/spaces/diffusers all import
# huggingface_hub, which freezes its cache paths from the environment at
# import time — so this line has to run before any of them.
import boot_guard  # noqa: F401

import math

import spaces
import gradio as gr
from PIL import Image

# Pre-load models on startup (imported early to trigger preload)
from model import preload_pipeline
print("Ensuring model is pre-loaded...")
preload_pipeline()

from config import (
    TITLE,
    DESCRIPTION,
    ARTICLE,
    DEFAULT_GUIDANCE_SCALE,
    FACE_GUIDANCE_MIN,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_FACE_PRESERVE_ENABLED,
    DEFAULT_IDENTITYNET_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    MAX_RESOLUTION,
    DEFAULT_RESOLUTION,
    ASPECT_RATIOS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_LORA_INTENSITY,
    LORA_STYLES,
    DEFAULT_LORA_STYLE,
    IMG_STRENGTH,
    TILE_SIZE,
    TILE_OVERLAP,
    DEFAULT_USE_TILED,
    BASE_PASS_LONG_EDGE,
    USE_ESCALATION,
)
from generator import generate_pixel_art
from utils import create_seed


# ─────────────────────────────────────────────────────────────────────────────
# Pixagram look: near-total black chrome, rounded squares, rainbow reserved
# for emphasis (wordmark, divider, Generate button). The rainbow uses the
# PICO-8 palette — on-brand for a pixel-art tool.
# ─────────────────────────────────────────────────────────────────────────────

PIXA_RAINBOW = (
    "linear-gradient(90deg, #ff004d, #ffa300, #ffec27, "
    "#00e436, #29adff, #7e5cff, #ff77a8)"
)
# Loop variant (first color repeated at the end) so the animated slide on the
# Generate button tiles seamlessly.
PIXA_RAINBOW_LOOP = (
    "linear-gradient(90deg, #ff004d, #ffa300, #ffec27, "
    "#00e436, #29adff, #7e5cff, #ff77a8, #ff004d)"
)

# "Rounded square" radius scale: generous corners, never pill-shaped.
PIXA_RADIUS = gr.themes.Size(
    name="radius_pixa",
    xxs="6px", xs="8px", sm="10px", md="14px", lg="18px", xl="22px", xxl="28px",
)

# Every variable is pinned in BOTH light and dark mode, so the app stays black
# even if a visitor (or the HF embed) forces ?__theme=light.
pixa_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.zinc,
    radius_size=PIXA_RADIUS,
    font=[
        gr.themes.GoogleFont("Inter"),
        gr.themes.GoogleFont("Press Start 2P"),  # loaded for the wordmark
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
).set(
    # Canvas + surfaces (true black canvas, near-black raised blocks)
    body_background_fill="#000000",
    body_background_fill_dark="#000000",
    background_fill_primary="#0e0e11",
    background_fill_primary_dark="#0e0e11",
    background_fill_secondary="#131318",
    background_fill_secondary_dark="#131318",
    block_background_fill="#0e0e11",
    block_background_fill_dark="#0e0e11",
    panel_background_fill="#08080a",
    panel_background_fill_dark="#08080a",
    block_shadow="none",
    block_shadow_dark="none",
    # Borders
    border_color_primary="#212127",
    border_color_primary_dark="#212127",
    block_border_color="#212127",
    block_border_color_dark="#212127",
    # Text
    body_text_color="#ececf1",
    body_text_color_dark="#ececf1",
    body_text_color_subdued="#9a9aa8",
    body_text_color_subdued_dark="#9a9aa8",
    block_title_text_color="#ececf1",
    block_title_text_color_dark="#ececf1",
    block_label_background_fill="#0e0e11",
    block_label_background_fill_dark="#0e0e11",
    block_label_text_color="#9a9aa8",
    block_label_text_color_dark="#9a9aa8",
    # Inputs
    input_background_fill="#131318",
    input_background_fill_dark="#131318",
    input_border_color="#26262c",
    input_border_color_dark="#26262c",
    input_border_color_focus="#29adff",
    input_border_color_focus_dark="#29adff",
    # Accents (sliders, checkboxes, radio pills)
    color_accent_soft="#1a1a22",
    color_accent_soft_dark="#1a1a22",
    slider_color="#7e5cff",
    slider_color_dark="#7e5cff",
    checkbox_background_color="#131318",
    checkbox_background_color_dark="#131318",
    checkbox_background_color_selected="#7e5cff",
    checkbox_background_color_selected_dark="#7e5cff",
    checkbox_border_color="#2c2c34",
    checkbox_border_color_dark="#2c2c34",
    checkbox_label_background_fill="#131318",
    checkbox_label_background_fill_dark="#131318",
    checkbox_label_background_fill_hover="#1a1a22",
    checkbox_label_background_fill_hover_dark="#1a1a22",
    checkbox_label_text_color="#ececf1",
    checkbox_label_text_color_dark="#ececf1",
    # Buttons — primary carries the rainbow, secondary stays greyscale
    button_primary_background_fill=PIXA_RAINBOW,
    button_primary_background_fill_dark=PIXA_RAINBOW,
    button_primary_background_fill_hover=PIXA_RAINBOW,
    button_primary_background_fill_hover_dark=PIXA_RAINBOW,
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",
    button_secondary_background_fill="#15151b",
    button_secondary_background_fill_dark="#15151b",
    button_secondary_background_fill_hover="#1d1d24",
    button_secondary_background_fill_hover_dark="#1d1d24",
    button_secondary_text_color="#ececf1",
    button_secondary_text_color_dark="#ececf1",
    button_secondary_border_color="#26262c",
    button_secondary_border_color_dark="#26262c",
)

PIXA_CSS = """
:root {
  --pixa-rainbow: linear-gradient(90deg,#ff004d,#ffa300,#ffec27,#00e436,#29adff,#7e5cff,#ff77a8);
  --pixa-rainbow-loop: linear-gradient(90deg,#ff004d,#ffa300,#ffec27,#00e436,#29adff,#7e5cff,#ff77a8,#ff004d);
}

.gradio-container { background: #000000 !important; }

/* ── Header: rainbow wordmark + pixagram.com pill ─────────────────────── */
#pixa-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 10px 2px 12px; flex-wrap: wrap;
}
#pixa-header .pixa-wordmark {
  font-family: 'Press Start 2P', monospace;
  font-size: 18px; line-height: 1.4; letter-spacing: 1px;
  background: var(--pixa-rainbow);
  -webkit-background-clip: text; background-clip: text;
  color: transparent; text-decoration: none;
}
#pixa-header .pixa-cta {
  font-size: 13px; font-weight: 600; color: #ececf1; text-decoration: none;
  padding: 8px 16px; border-radius: 12px; border: 2px solid transparent;
  background: linear-gradient(#0e0e11,#0e0e11) padding-box,
              var(--pixa-rainbow) border-box;
}
#pixa-header .pixa-cta:hover {
  background: var(--pixa-rainbow) border-box;
  color: #000000;
}
.pixa-divider {
  height: 2px; border-radius: 2px;
  background: var(--pixa-rainbow);
  margin: 2px 0 10px;
}

/* ── Generate button: the one animated rainbow moment ─────────────────── */
#pixa-generate {
  background: var(--pixa-rainbow-loop) !important;
  background-size: 200% 100% !important;
  color: #000000 !important;
  font-weight: 700;
  border: none !important;
  border-radius: 16px !important;
}
@media (prefers-reduced-motion: no-preference) {
  #pixa-generate { animation: pixa-slide 8s linear infinite; }
}
@keyframes pixa-slide {
  from { background-position: 0% 50%; }
  to   { background-position: 200% 50%; }
}

/* ── Rounded squares on media ─────────────────────────────────────────── */
.gradio-container .image-container,
.gradio-container .image-container img,
.gradio-container .image-frame,
.gradio-container .image-frame img {
  border-radius: 14px;
}

/* ── Links, focus, scrollbars ─────────────────────────────────────────── */
.gradio-container .prose a { color: #29adff; }
.gradio-container .prose a:hover { color: #ff77a8; }
.gradio-container button:focus-visible {
  outline: 2px solid #29adff; outline-offset: 2px;
}
.gradio-container ::-webkit-scrollbar { width: 10px; height: 10px; }
.gradio-container ::-webkit-scrollbar-thumb {
  background: #26262c; border-radius: 8px;
}
.gradio-container ::-webkit-scrollbar-track { background: transparent; }

/* ── Footer ───────────────────────────────────────────────────────────── */
#pixa-footer {
  text-align: center; color: #9a9aa8; font-size: 13px; padding: 10px 0 4px;
}
#pixa-footer a { color: #29adff; text-decoration: none; }
#pixa-footer a:hover { color: #ff77a8; }
"""

PIXA_HEADER_HTML = """
<div id="pixa-header">
  <a class="pixa-wordmark" href="https://pixagram.com" target="_blank"
     rel="noopener">PIXAGRAM</a>
  <a class="pixa-cta" href="https://pixagram.com" target="_blank"
     rel="noopener">Post it on pixagram.com</a>
</div>
<div class="pixa-divider"></div>
"""

PIXA_FOOTER_HTML = """
<div class="pixa-divider"></div>
<div id="pixa-footer">
  Built by <a href="https://pixagram.com" target="_blank"
  rel="noopener">Pixagram</a>, the pixel-art social network —
  <a href="https://pixagram.com" target="_blank"
  rel="noopener">pixagram.com</a>
</div>
"""


def _gpu_duration(
    input_image,
    additional_prompt,
    guidance_scale,
    num_inference_steps,
    lora_intensity,
    lora_style,
    img2img_strength,
    seed,
    identity_preserve,
    identitynet_strength,
    ip_adapter_scale,
    resolution,
    aspect_ratio,
    use_tiled,
    tile_size,
    tile_overlap,
) -> int:
    """
    Per-job @spaces.GPU duration (ZeroGPU accepts a callable that receives
    the same arguments as the decorated function). Single-pass jobs keep the
    old 40 s hold; tiled jobs scale with the estimated tile count at the
    final rung (plus the escalation overhead), so a max-budget job isn't
    killed mid-ladder. ZeroGPU only bills the seconds actually used — the
    duration is a ceiling, not a cost.
    """
    try:
        res = max(512, int(resolution))
        t_size = max(64, int(tile_size))
        t_overlap = max(0, min(int(tile_overlap), t_size - 8))
    except (TypeError, ValueError):
        return 40

    tiled = bool(use_tiled) or res > BASE_PASS_LONG_EDGE
    if not tiled:
        return 40

    # Approximate final-rung canvas from the pixel budget + aspect ratio
    # (mirrors generator._txt2img_dims: w*h ~= res^2, w/h = ar). Needed
    # because the tiler now uses SQUARE tiles clamped to the SHORT edge —
    # wide/tall aspects get more, smaller tiles, and a square-canvas
    # estimate would undershoot the ceiling for them.
    try:
        arw, arh = ASPECT_RATIOS.get(str(aspect_ratio), (1, 1))
        ar = float(arw) / float(arh)
    except Exception:
        ar = 1.0
    w = res * math.sqrt(ar)
    h = res / math.sqrt(ar)

    # Mirror tiled_pipeline's square-tile grid math.
    t = max(64.0, min(float(t_size), w, h))
    ov = min(t - 8.0, t_overlap * t / float(t_size))
    stride = max(1.0, t - ov)
    nx = 1 + max(0, math.ceil((w - t) / stride))
    ny = 1 + max(0, math.ceil((h - t) / stride))
    n_tiles = nx * ny
    # Escalation runs ~1.8x the final rung's tile count across all rungs
    # (geometric series); ~2.5 s per LCM tile + 40 s base/setup headroom.
    est = 40 + int(n_tiles * (1.8 if USE_ESCALATION else 1.0) * 2.5)
    return max(60, min(300, est))


# ZeroGPU accepts a callable for `duration`: it receives the same args as
# the decorated function and returns the per-job hold in seconds. This
# was hardcoded to 25s while _gpu_duration sat unused above — so every
# high-budget tiled/escalation job was killed mid-run.
@spaces.GPU(duration=_gpu_duration)
def process_image(
    input_image,  # PIL.Image (gr.Image) — OPTIONAL
    additional_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    lora_intensity: float,
    lora_style: str,
    img2img_strength: float,
    seed: int,
    identity_preserve: bool,
    identitynet_strength: float,
    ip_adapter_scale: float,
    resolution: int,
    aspect_ratio: str,
    use_tiled: bool,
    tile_size: int,
    tile_overlap: int,
) -> tuple:
    """
    Generate pixel art. Decorated with @spaces.GPU for ZeroGPU support.

    Two modes:
      - IMG2IMG: an image is uploaded — it is converted to pixel art,
        preserving its own aspect ratio. Face identity (when enabled) is
        preserved during diffusion via InstantID — no post-blending.
      - TEXT-TO-IMAGE: no image is uploaded — the prompt is MANDATORY and
        the canvas size comes from `resolution` + `aspect_ratio`.

    Jobs above BASE_PASS_LONG_EDGE automatically switch to the tiled
    coarse-to-fine pipeline (coherent base pass, then tile refinement with
    per-tile InstantID routing); `use_tiled` forces it on for smaller jobs.

    Args:
        input_image: Input image from Gradio, or None for text-to-image
        additional_prompt: Style instructions (required without an image)
        guidance_scale: Guidance scale parameter
        num_inference_steps: Number of denoising steps
        lora_style: Which pixel-art LoRA to use ("retroart" or "vga") —
                    selects the weights AND the matching trigger prompt
        seed: Random seed (-1 for random)
        identity_preserve: Enable InstantID identity conditioning
        identitynet_strength: IdentityNet ControlNet scale (facial structure)
        ip_adapter_scale: IP-Adapter scale (identity likeness)
        resolution: Long-edge target resolution in pixels
        aspect_ratio: Canvas aspect ratio ("2:1" ... "1:2") — only used in
                      text-to-image mode; uploads keep their own ratio
        use_tiled: Force the tiled pipeline even below the auto-tile
                   threshold (BASE_PASS_LONG_EDGE)
        tile_size: Diffusion tile edge in pixels (tiled mode)
        tile_overlap: Blend overlap between adjacent tiles in pixels

    Returns:
        Tuple of (output image, seed info text)
    """
    if input_image is None and not (additional_prompt or "").strip():
        gr.Warning(
            "Write a prompt to generate without an image, "
            "or upload an image to convert!"
        )
        return None, "No image and no prompt provided"

    try:
        output_image, actual_seed, face_report, full_prompt = generate_pixel_art(
            input_image=input_image,
            additional_prompt=additional_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            lora_intensity=lora_intensity,
            lora_style=lora_style,
            img2img_strength=img2img_strength,
            seed=seed,
            identity_preserve=identity_preserve,
            identitynet_strength=identitynet_strength,
            ip_adapter_scale=ip_adapter_scale,
            resolution=int(resolution),
            aspect_ratio=aspect_ratio,
            use_tiled=bool(use_tiled),
            tile_size=int(tile_size),
            tile_overlap=int(tile_overlap),
        )
        
        # Include image dimensions in seed info
        seed_info = f"Seed: {actual_seed}\nOutput: {output_image.size[0]}x{output_image.size[1]}\nPrompt: {full_prompt}"
        
        if face_report:
            seed_info += f"\n\nFace Identity (InstantID):\n{face_report}"
        
        return output_image, seed_info
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        # gr.Error must be RAISED to surface in the UI — instantiating it
        # (the previous code) silently discards it.
        raise gr.Error(f"Generation failed: {str(e)}") from e
    finally:
        # Return freed host pages to the OS and release cached VRAM blocks
        # after every job, and log real vs cache-inflated memory. Keeps RSS
        # from ratcheting up across generations (glibc fragmentation).
        import boot_guard
        boot_guard.after_job_cleanup()


def randomize_seed():
    """Generate a random seed."""
    return create_seed(-1)


# Build the Gradio interface
with gr.Blocks(
    title=TITLE,
    # Gradio 6: `theme` moved from the Blocks constructor to launch().
    # PRIVACY: uploaded/generated images live only in Gradio's temp cache
    # (pinned to ephemeral /tmp by boot_guard, never /data). On top of that,
    # purge cached files older than 1h every 15 min, and Gradio empties the
    # cache entirely on restart. Nothing is ever written to persistent
    # storage — that holds model weights only.
    delete_cache=(900, 3600),
) as demo:
    
    gr.HTML(PIXA_HEADER_HTML)
    
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            input_image = gr.Image(
                label=(
                    "📷 Upload Image (optional) — leave empty to generate "
                    "from your prompt alone"
                ),
                type="pil",
                height=400,
                sources=["upload", "clipboard"],
            )
            
            additional_prompt = gr.Textbox(
                label="✨ Prompt (required when no image is uploaded)",
                placeholder=(
                    "With an image: optional extra style instructions. "
                    "Without an image: describe what to generate, e.g. "
                    "'a knight standing in front of a castle at sunset'..."
                ),
                lines=2,
            )

            lora_style = gr.Radio(
                label="🕹️ Pixel-Art Style (LoRA)",
                choices=list(LORA_STYLES.keys()),
                value=DEFAULT_LORA_STYLE,
                info=(
                    "retroart → retroart.safetensors ('retroart style' "
                    "trigger) · vga → vga.safetensors ('dosvga style' "
                    "trigger). The matching trigger prompt is applied "
                    "automatically."
                ),
            )

            aspect_ratio = gr.Dropdown(
                label="🖼️ Aspect Ratio (text-to-image only)",
                choices=list(ASPECT_RATIOS.keys()),
                value=DEFAULT_ASPECT_RATIO,
                info=(
                    "Only used when no image is uploaded. "
                    "Uploaded images always keep their own aspect ratio."
                ),
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=2.0,
                    value=DEFAULT_GUIDANCE_SCALE,
                    step=0.01,
                    info=(
                        "1.0 = fastest: CFG off, negative prompt inactive "
                        "(~1.8x faster). Higher = stronger prompt adherence "
                        "at 2x the compute. "
                        f"Auto-raised to {FACE_GUIDANCE_MIN} on face images so "
                        "anti-duplication negatives take effect."
                    ),
                )
                
                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=8,
                    maximum=50,
                    value=DEFAULT_NUM_INFERENCE_STEPS,
                    step=1,
                    info="More steps = higher quality but slower (8-12 is the LCM sweet spot)",
                )

                lora_intensity = gr.Slider(
                    label="🎨 Style Intensity (LoRA)",
                    minimum=0.0,
                    maximum=1.5,
                    value=DEFAULT_LORA_INTENSITY,
                    step=0.05,
                    info=(
                        "Pixel-art style strength. Applied x1.2 on faces and "
                        "x0.6 on images without a face."
                    ),
                )

                img2img_strength = gr.Slider(
                    label="🖌️ Image Strength (img2img)",
                    minimum=0.1,
                    maximum=1.0,
                    value=IMG_STRENGTH,
                    step=0.05,
                    info=(
                        "img2img. Applied x1.0 (no changes) on faces and "
                        "x0.5 on images without a face."
                    ),
                )

                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=DEFAULT_SEED,
                        precision=0,
                        info="-1 for random seed",
                    )
                    randomize_btn = gr.Button("🎲 Random", size="sm")

                resolution = gr.Slider(
                    label="🔍 Resolution (pixel budget)",
                    minimum=512,
                    maximum=MAX_RESOLUTION,
                    value=DEFAULT_RESOLUTION,
                    step=64,
                    info=(
                        "Total pixels ≈ value², whatever the shape: 832 → "
                        "~832x832 worth of pixels (16:9 becomes 1112x624). "
                        f"Above {BASE_PASS_LONG_EDGE} the tiled pipeline "
                        "kicks in automatically (coherent base pass + tile "
                        "refinement), so large values stay sharp AND "
                        "coherent."
                    ),
                )

            with gr.Accordion("🧩 High-Resolution Tiling", open=False):
                use_tiled = gr.Checkbox(
                    label="Force Tiled Refinement",
                    value=DEFAULT_USE_TILED,
                    info=(
                        f"Tiling auto-activates above {BASE_PASS_LONG_EDGE}px "
                        "— this forces it on for smaller jobs too (adds a "
                        "detail-refinement pass over the normal result)."
                    ),
                )
                tile_size = gr.Slider(
                    label="Tile Size",
                    minimum=512,
                    maximum=1024,
                    value=TILE_SIZE,
                    step=64,
                    info=(
                        "Diffusion window per tile. Bigger tiles = more "
                        "coherent but slower and more VRAM."
                    ),
                )
                tile_overlap = gr.Slider(
                    label="Tile Overlap",
                    minimum=64,
                    maximum=384,
                    value=TILE_OVERLAP,
                    step=32,
                    info=(
                        "Blend zone between adjacent tiles. More overlap = "
                        "fewer seams, more compute (~25% of tile size is a "
                        "good default)."
                    ),
                )
            
            with gr.Accordion("🧑 Face Identity (InstantID)", open=False):
                identity_preserve = gr.Checkbox(
                    label="Enable InstantID Identity Preservation",
                    value=DEFAULT_FACE_PRESERVE_ENABLED,
                    info=(
                        "Injects the real face's identity DURING diffusion: "
                        "IdentityNet (keypoint structure) + IP-Adapter "
                        "(ArcFace embedding). No post-blending."
                    ),
                )
                identitynet_strength = gr.Slider(
                    label="IdentityNet Strength (structure)",
                    minimum=0.0,
                    maximum=1.5,
                    value=DEFAULT_IDENTITYNET_SCALE,
                    step=0.05,
                    info=(
                        "Facial structure/pose lock. Raise for stricter "
                        "feature placement, lower if the face fights the "
                        "pixel-art style."
                    ),
                )
                ip_adapter_scale = gr.Slider(
                    label="IP-Adapter Strength (likeness)",
                    minimum=0.0,
                    maximum=1.5,
                    value=DEFAULT_IP_ADAPTER_SCALE,
                    step=0.05,
                    info=(
                        "Identity-embedding strength. Raise if the person "
                        "isn't recognizable, lower if the face comes out "
                        "too photographic."
                    ),
                )
            
            generate_btn = gr.Button(
                "🎨 Generate Pixel Art",
                variant="primary",
                size="lg",
                elem_id="pixa-generate",
            )
        
        with gr.Column(scale=1):
            # Output section
            output_image = gr.Image(
                label="🖼️ Pixel Art Result",
                type="pil",
                height=400,
            )
            
            seed_info = gr.Textbox(
                label="📋 Generation Info",
                lines=10,
                interactive=False,
            )
    
    # Example prompts
    gr.Markdown("### 💡 Example Style Instructions")
    gr.Examples(
        examples=[
            [""],
            ["16-bit retro game aesthetic"],
            ["vibrant colors, high contrast"],
            ["minimalist, limited color palette"],
            ["isometric view, game asset style"],
        ],
        inputs=[additional_prompt],
        label="Click to use",
    )
    
    gr.Markdown(ARTICLE)
    
    gr.HTML(PIXA_FOOTER_HTML)
    
    # Event handlers
    randomize_btn.click(
        fn=randomize_seed,
        outputs=[seed],
    )
    
    generate_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            additional_prompt,
            guidance_scale,
            num_inference_steps,
            lora_intensity,
            lora_style,
            img2img_strength,
            seed,
            identity_preserve,
            identitynet_strength,
            ip_adapter_scale,
            resolution,
            aspect_ratio,
            use_tiled,
            tile_size,
            tile_overlap,
        ],
        outputs=[output_image, seed_info],
    )


if __name__ == "__main__":
    demo.queue()
    # Gradio 6: `theme` and `css` are launch() parameters. `share=True`
    # dropped — it is unsupported (a no-op + warning) on Hugging Face Spaces.
    demo.launch(
        theme=pixa_theme,
        css=PIXA_CSS,
    )