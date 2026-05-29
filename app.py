"""
Pixagram Pixel Art Converter - Hugging Face Space Application

Transform regular images into pixel art using FLUX.1-Kontext with Pixagram LoRA.
Optimized for ZeroGPU (Hugging Face Spaces serverless GPU) with speed improvements.
"""

import spaces
import gradio as gr
from PIL import Image

# Install facenet-pytorch early (--no-deps avoids torch/Pillow version conflicts)
from face_preserve import _ensure_facenet_installed
print("Ensuring facenet-pytorch is installed...")
_ensure_facenet_installed()

# Pre-load models on startup (imported early to trigger preload)
from model import preload_pipeline
print("Ensuring model is pre-loaded...")
preload_pipeline()

from config import (
    TITLE,
    DESCRIPTION,
    ARTICLE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_FACE_PRESERVE_ENABLED,
    FACE_BLEND_STRENGTH,
    FACE_SIMILARITY_THRESHOLD,
    MAX_RESOLUTION,
    DEFAULT_RESOLUTION,
    TILE_SIZE,
    TILE_OVERLAP,
    DEFAULT_USE_TILED,
    DEFAULT_LORA_INTENSITY,
    IMG_STRENGTH,
)
from generator import generate_pixel_art
from utils import create_seed


@spaces.GPU(duration=120)
def process_image(
    input_image: Image.Image,
    additional_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    lora_intensity: float,
    img2img_strength: float,
    seed: int,
    face_preserve: bool,
    face_blend_strength: float,
    face_similarity_threshold: float,
    resolution: int,
    use_tiled: bool,
    tile_size: int,
    tile_overlap: int,
) -> tuple:
    """
    Process an image and convert it to pixel art.
    Decorated with @spaces.GPU for ZeroGPU support.
    Preserves original aspect ratio. Optionally preserves face identity.

    Args:
        input_image: Input image from Gradio
        additional_prompt: Additional style instructions
        guidance_scale: Guidance scale parameter
        num_inference_steps: Number of denoising steps
        seed: Random seed (-1 for random)
        face_preserve: Whether to enable face identity preservation
        face_blend_strength: Max face blend strength (0-1)
        face_similarity_threshold: Cosine similarity threshold
        resolution: Long-edge target resolution in pixels (up to 3840 / 4K)
        use_tiled: Force MultiDiffusion tiled denoising on
        tile_size: Tile size in pixels (multiple of 64)
        tile_overlap: Pixel overlap between adjacent tiles

    Returns:
        Tuple of (output image, seed info text)
    """
    if input_image is None:
        gr.Warning("Please upload an image first!")
        return None, "No image provided"

    try:
        output_image, actual_seed, face_report, full_prompt = generate_pixel_art(
            input_image=input_image,
            additional_prompt=additional_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            lora_intensity=lora_intensity,
            img2img_strength=img2img_strength,
            seed=seed,
            face_preserve=face_preserve,
            face_blend_strength=face_blend_strength,
            face_similarity_threshold=face_similarity_threshold,
            resolution=int(resolution),
            use_tiled=use_tiled,
            tile_size=int(tile_size),
            tile_overlap=int(tile_overlap),
        )
        
        # Include image dimensions in seed info
        seed_info = f"Seed: {actual_seed}\nOutput: {output_image.size[0]}x{output_image.size[1]}\nPrompt: {full_prompt}"
        
        if face_report:
            seed_info += f"\n\nFace Preservation:\n{face_report}"
        
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
                label="📷 Upload Image",
                type="pil",
                height=400,
            )
            
            additional_prompt = gr.Textbox(
                label="✨ Additional Style Instructions (optional)",
                placeholder="e.g., 16-bit retro game aesthetic, vibrant colors...",
                lines=2,
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=2.0,
                    value=DEFAULT_GUIDANCE_SCALE,
                    step=0.01,
                    info="Higher values = stronger adherence to prompt",
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

            with gr.Accordion("🔍 Resolution & Tiling (high-detail mode)", open=False):
                gr.Markdown(
                    "Generate at higher resolution using **MultiDiffusion** "
                    "tiled denoising. The image is split into overlapping tiles "
                    "that share prompt and ControlNet conditioning, then "
                    "noise predictions are blended in the overlap regions every "
                    "step. Slower than single-pass generation, but lets you go "
                    "well past SDXL's native 1024px."
                )
                resolution = gr.Slider(
                    label="Output resolution (long edge, px)",
                    minimum=768,
                    maximum=MAX_RESOLUTION,
                    value=DEFAULT_RESOLUTION,
                    step=64,
                    info=(
                        f"Up to {MAX_RESOLUTION}px (4K). Tiling auto-activates "
                        f"above {TILE_SIZE}px. Values are snapped to multiples of 64."
                    ),
                )
                use_tiled = gr.Checkbox(
                    label="Force MultiDiffusion tiled denoising",
                    value=DEFAULT_USE_TILED,
                    info="Tiling is always used above the tile size; this only "
                         "forces it on for in-range resolutions.",
                )
                tile_size = gr.Slider(
                    label="Tile size (px)",
                    minimum=512,
                    maximum=1024,
                    value=TILE_SIZE,
                    step=64,
                    info="SDXL works best around 768–1024 per tile.",
                )
                tile_overlap = gr.Slider(
                    label="Tile overlap (px)",
                    minimum=64,
                    maximum=384,
                    value=TILE_OVERLAP,
                    step=32,
                    info="~25% of tile size is a good default. More overlap = "
                         "smoother seams but more compute.",
                )
            
            with gr.Accordion("🧑 Face Preservation (MIT-licensed FaceNet)", open=False):
                face_preserve = gr.Checkbox(
                    label="Enable Face Identity Preservation",
                    value=DEFAULT_FACE_PRESERVE_ENABLED,
                    info="Detects faces and blends original identity back if it drifts during conversion",
                )
                face_blend_strength = gr.Slider(
                    label="Blend Strength",
                    minimum=0.0,
                    maximum=1.0,
                    value=FACE_BLEND_STRENGTH,
                    step=0.05,
                    info="Max blend of original face into output (higher = more identity, less pixel art style on face)",
                )
                face_similarity_threshold = gr.Slider(
                    label="Similarity Threshold",
                    minimum=0.2,
                    maximum=0.9,
                    value=FACE_SIMILARITY_THRESHOLD,
                    step=0.05,
                    info="Cosine similarity below which blending activates (lower = only fix severe drift)",
                )
            
            generate_btn = gr.Button(
                "🎨 Convert to Pixel Art",
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
            face_preserve,
            face_blend_strength,
            face_similarity_threshold,
            resolution,
            use_tiled,
            tile_size,
            tile_overlap,
        ],
        outputs=[output_image, seed_info],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
