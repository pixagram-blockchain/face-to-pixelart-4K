"""
Pixagram Pixel Art Converter - Hugging Face Space Application

Transform regular images into pixel art using SDXL + Pixagram LoRA, with
InstantID face identity (IdentityNet + IP-Adapter) preserved during diffusion.
Optimized for ZeroGPU (Hugging Face Spaces serverless GPU).
"""

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
    IMG_STRENGTH,
    TILE_SIZE,
    TILE_OVERLAP,
    DEFAULT_USE_TILED,
    BASE_PASS_LONG_EDGE,
    USE_ESCALATION,
)
from generator import generate_pixel_art
from utils import create_seed


def _gpu_duration(
    input_image,
    additional_prompt,
    guidance_scale,
    num_inference_steps,
    lora_intensity,
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

    # Tiles per axis at the final (budget-edge) rung.
    stride = max(1, t_size - t_overlap)
    per_axis = 1 + max(0, math.ceil((res - t_size) / stride))
    n_tiles = per_axis * per_axis
    # Escalation runs ~1.8x the final rung's tile count across all rungs
    # (geometric series); ~2.5 s per LCM tile + 40 s base/setup headroom.
    est = 40 + int(n_tiles * (1.8 if USE_ESCALATION else 1.0) * 2.5)
    return max(60, min(300, est))


@spaces.GPU(duration=_gpu_duration)
def process_image(
    input_image,  # PIL.Image (gr.Image) — OPTIONAL
    additional_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    lora_intensity: float,
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
        gr.Error(f"Generation failed: {str(e)}")
        return None, f"Error: {str(e)}"


def randomize_seed():
    """Generate a random seed."""
    return create_seed(-1)


# Build the Gradio interface
with gr.Blocks(
    title=TITLE,
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="pink",
    ),
) as demo:
    
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
                        "Higher values = stronger adherence to prompt. "
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
                    info="More steps = higher quality but slower (16 recommended for speed)",
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
    demo.launch(share=True)