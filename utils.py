"""
Utility functions for Pixagram Pixel Art Generator
With face detection utilities

All components use commercially-permissive licenses.
"""
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import random
from config import Config
import cv2
import numpy as np
from typing import Optional, Tuple, List

# Simple global caching for the captioner
captioner_processor = None
captioner_model = None


def preload_captioner():
    """
    Preload the BLIP captioning model at startup.
    Call this during model initialization to avoid the 990MB download
    hitting on the first generation.
    """
    global captioner_processor, captioner_model
    if captioner_model is None:
        print("  Loading Captioner (BLIP)...")
        captioner_processor = BlipProcessor.from_pretrained(Config.CAPTIONER_REPO)
        captioner_model = BlipForConditionalGeneration.from_pretrained(
            Config.CAPTIONER_REPO, torch_dtype=Config.DTYPE
        ).to(Config.DEVICE)
        print("  [OK] Captioner loaded")


def resize_image_to_target(image: Image.Image, long_side: int = 1280) -> Image.Image:
    """
    Resize so the longer side is ~`long_side` pixels, preserving aspect ratio.
    Both dimensions are snapped down to multiples of 64 so the SDXL UNet and
    the tile grid (multiples of TILE_SIZE/8 in latent space) line up cleanly.
    """
    image = image.convert("RGB")
    w, h = image.size
    aspect_ratio = w / h

    if w >= h:
        new_w = long_side
        new_h = int(round(long_side / aspect_ratio))
    else:
        new_h = long_side
        new_w = int(round(long_side * aspect_ratio))

    # Snap to multiples of 64
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)

    return image.resize((new_w, new_h), Image.LANCZOS)


def resize_image_to_1mp(image: Image.Image) -> Image.Image:
    """Back-compat: resize to ~1.6MP (1280 long side)."""
    return resize_image_to_target(image, long_side=1280)



def prepare_image(
    image: Image.Image,
    long_side: int = 1280,
) -> Tuple[Image.Image, int, int]:
    """
    Prepare an input image for the pipeline.
    Resizes so the longer side is `long_side` pixels, preserving aspect
    ratio, with dimensions divisible by 64.

    Args:
        image: PIL image to resize.
        long_side: Target longer-edge length in pixels.

    Returns:
        Tuple of (resized image, target_width, target_height).
    """
    resized = resize_image_to_target(image, long_side=long_side)
    w, h = resized.size
    return resized, w, h


def create_seed(seed: int = -1) -> int:
    """
    Create a generation seed.
    If seed is -1, generate a random seed; otherwise return seed as-is.
    """
    if seed < 0:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def format_prompt(trigger_word: str, additional_prompt: str = "") -> str:
    """
    Combine the trigger word with optional additional style instructions.
    """
    prompt = trigger_word
    if additional_prompt and additional_prompt.strip():
        prompt += f", {additional_prompt.strip()}"
    return prompt


def get_caption(image: Image.Image) -> str:
    """Generates a caption for the image using BLIP."""
    global captioner_processor, captioner_model

    if captioner_model is None:
        preload_captioner()

    inputs = captioner_processor(image, return_tensors="pt").to(Config.DEVICE)
    out = captioner_model.generate(**inputs, max_new_tokens=25)
    caption = captioner_processor.decode(out[0], skip_special_tokens=True)
    return caption


# ============================================================
# FACE UTILITIES (Commercial-Friendly)
# ============================================================

def detect_faces_opencv(
    image: Image.Image,
    min_size: Tuple[int, int] = (30, 30)
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces using OpenCV Haar Cascades. License: BSD
    """
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        gray = image_np
    else:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
    )

    return [tuple(f) for f in faces]


def has_face(image: Image.Image) -> bool:
    """Quick check if image contains a face."""
    faces = detect_faces_opencv(image)
    return len(faces) > 0


def match_colors_lab(
    src: Image.Image,
    ref: Image.Image,
    strength: float = 1.0,
) -> Image.Image:
    """
    Reinhard colour transfer: rescale `src` so its per-channel mean/std in LAB
    match `ref`, then blend by `strength` (0 = unchanged, 1 = full match).

    Used to remove hue/exposure drift between independently-generated tiles by
    pulling the stitched mosaic back toward a globally-coherent reference
    (e.g. the upscaled base pass). Operating in LAB corrects luminance and
    colour without destroying detail. OpenCV (BSD) + NumPy — no new deps.

    Args:
        src: image to recolour (the tile mosaic).
        ref: reference whose colour statistics to match.
        strength: 0..1 blend between the original and the fully-matched result.

    Returns:
        Recoloured PIL image, same size as `src`.
    """
    strength = float(max(0.0, min(1.0, strength)))
    if strength == 0.0:
        return src

    src_rgb = np.asarray(src.convert("RGB"))
    ref_rgb = np.asarray(ref.convert("RGB").resize(src.size, Image.LANCZOS))

    s = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    out = s.copy()
    for c in range(3):
        s_mean, s_std = float(s[..., c].mean()), float(s[..., c].std())
        r_mean, r_std = float(r[..., c].mean()), float(r[..., c].std())
        if s_std < 1e-5:
            # Flat channel (e.g. a uniform sky tile): can't rescale by a std
            # ratio, but still shift the mean so the tone matches the ref.
            out[..., c] = s[..., c] - s_mean + r_mean
        else:
            out[..., c] = (s[..., c] - s_mean) * (r_std / s_std) + r_mean

    out = np.clip(out, 0, 255)
    matched = np.clip(s * (1.0 - strength) + out * strength, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def visualize_face_detection(
    image: Image.Image,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Image.Image:
    """Draw bounding boxes around detected faces."""
    image_np = np.array(image.copy())
    faces = detect_faces_opencv(image)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), color, thickness)

    return Image.fromarray(image_np)


print("[OK] Utils loaded (with face utilities)")