"""
Model loading for Pixagram Pixel Art Generator
With Prompt-Based Face Preservation

This module loads:
1. SDXL Pipeline with ControlNet
2. Pixel Art LoRA
3. Face Attribute Analyzer (OpenCV + MediaPipe, no InsightFace)
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler
)

# ---- mediapipe shim ------------------------------------------------
# controlnet_aux unconditionally imports mp.solutions.drawing_utils in
# its __init__.py (for MediapipeFaceDetector).  New mediapipe (>=0.10.18)
# removed mp.solutions entirely, so that import crashes even though we
# never use the mediapipe_face annotator — we only need ZoeDetector.
#
# Fix: if mp.solutions is missing, inject a harmless stub module so the
# import chain doesn't blow up.  The stub is never actually called.
# ---------------------------------------------------------------------
import importlib, sys, types

def _patch_mediapipe_solutions():
    """Add a dummy mp.solutions tree if the real one is missing."""
    try:
        mp = importlib.import_module("mediapipe")
    except ImportError:
        return                       # mediapipe not installed at all — nothing to patch

    if hasattr(mp, "solutions"):
        return                       # old mediapipe — no patch needed

    # A stub module that returns a no-op for any attribute access.
    # This lets controlnet_aux do things like:
    #   mp_drawing = mp.solutions.drawing_utils
    #   FACEMESH_TESSELATION = mp.solutions.face_mesh.FACEMESH_TESSELATION
    # without crashing.  None of these are ever actually *called* since
    # we never use MediapipeFaceDetector at runtime.
    class _StubModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            # Explicit metadata so sys.modules iteration never gets a
            # stub where it expects str/None (Gradio reload, importlib).
            self.__file__ = None
            self.__path__ = []
            self.__spec__ = None
            self.__loader__ = None
            self.__all__ = []

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _StubModule(f"{self.__name__}.{name}")

        def __call__(self, *args, **kwargs):
            return self

        def __iter__(self):
            return iter([])

        def __bool__(self):
            return False

    solutions = _StubModule("mediapipe.solutions")
    solutions.__package__ = "mediapipe.solutions"

    for sub in (
        "drawing_utils",
        "drawing_styles",
        "face_detection",
        "face_mesh",
        "hands",
        "pose",
    ):
        mod = _StubModule(f"mediapipe.solutions.{sub}")
        setattr(solutions, sub, mod)
        sys.modules[f"mediapipe.solutions.{sub}"] = mod

    mp.solutions = solutions
    sys.modules["mediapipe.solutions"] = solutions

_patch_mediapipe_solutions()
# ---- end shim ------------------------------------------------------

from controlnet_aux import ZoeDetector
from huggingface_hub import hf_hub_download
from config import Config
from faceid import FaceAnalyzer
from utils import preload_captioner


class ModelHandler:
    """
    Handles loading and management of all models.

    Components:
    - SDXL Pipeline with ControlNet
    - Zoe Depth Detector
    - LoRA Weights
    - Face Attribute Analyzer (Commercial)
    """

    def __init__(self):
        self.pipeline = None
        self.zoe_detector = None
        self.face_analyzer = None

    def load_models(self):
        print("=" * 60)
        print("Loading Pixagram Pixel Art Models")
        print("  (Prompt-Based Face Preservation)")
        print("=" * 60)

        # 1. Load Zoe Depth Detector
        print("\n[1/4] Loading Zoe Depth Detector...")
        try:
            self.zoe_detector = ZoeDetector.from_pretrained(Config.ANNOTATOR_REPO)
            self.zoe_detector.to(Config.DEVICE)
            print("  [OK] ZoeDetector loaded")
        except Exception as e:
            print(f"  [ERROR] Failed to load ZoeDetector: {e}")
            raise

        # 2. Load ControlNets: Depth (structure/pose) + Canny (face lock)
        print("\n[2/4] Loading ControlNets (Zoe Depth + Canny)...")
        depth_controlnet = ControlNetModel.from_pretrained(
            Config.CN_ZOE_REPO,
            torch_dtype=Config.DTYPE
        )
        print("  [OK] ControlNet Depth loaded")
        canny_controlnet = ControlNetModel.from_pretrained(
            Config.CN_CANNY_REPO,
            torch_dtype=Config.DTYPE
        )
        print("  [OK] ControlNet Canny loaded (face structure lock)")

        # IMPORTANT: order is [depth, canny] and MUST match the order of
        # the control-image list and conditioning-scale list passed by
        # generator.py / tiled_pipeline.py.
        controlnets = [depth_controlnet, canny_controlnet]

        # 3. Load Base Pipeline with checkpoint
        print("\n[3/4] Loading SDXL Pipeline...")
        checkpoint_path = hf_hub_download(
            repo_id=Config.REPO_ID,
            filename=Config.CHECKPOINT_FILENAME
        )
        print(f"  [OK] Checkpoint downloaded: {Config.CHECKPOINT_FILENAME}")

        self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            checkpoint_path,
            controlnet=controlnets,
            torch_dtype=Config.DTYPE,
            use_safetensors=True
        )
        print("  [OK] Pipeline loaded (MultiControlNet: depth + canny)")

        # Load LoRA
        print("\n  Loading LoRA weights...")
        lora_path = hf_hub_download(
            repo_id=Config.REPO_ID,
            filename=Config.LORA_FILENAME
        )
        self.pipeline.load_lora_weights(lora_path, adapter_name="retroart")
        # IMPORTANT: do NOT fuse. Fusing bakes a single fixed scale (1.25)
        # into the UNet, which made the style far too strong — over-cooking
        # non-portraits and over-stylising faces, with no way to differentiate.
        # Keeping it as a live PEFT adapter lets generator.py set the style
        # strength per request via set_adapters(). We seed a sensible default
        # here; generator overrides it each call.
        self.pipeline.set_adapters(
            ["retroart"], adapter_weights=[Config.DEFAULT_LORA_INTENSITY]
        )
        print(
            f"  [OK] LoRA loaded (adapter 'retroart', unfused, "
            f"default scale {Config.DEFAULT_LORA_INTENSITY}): {Config.LORA_FILENAME}"
        )

        # Setup scheduler
        print("\n  Configuring LCM Scheduler...")
        self.pipeline.scheduler = LCMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        print("  [OK] LCM Scheduler configured")

        # Move pipeline to device and optimize
        print(f"\n  Moving pipeline to {Config.DEVICE}...")
        self.pipeline.to(Config.DEVICE)
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_attention_slicing()

        if Config.DEVICE == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("  [OK] xformers enabled")
            except Exception as e:
                print(f"  [INFO] xformers not available: {e}")

        # 4. Load Face Attribute Analyzer
        # Lightweight: just OpenCV + MediaPipe, no GPU models
        print("\n[4/4] Loading Face Attribute Analyzer...")
        try:
            self.face_analyzer = FaceAnalyzer(
                min_confidence=Config.FACE_DETECTION_CONFIDENCE
            )
            print("  [OK] Face Analyzer loaded")
            print("       - Face Detection: OpenCV/MediaPipe (BSD/Apache 2.0)")
            print("       - Attribute Analysis: MediaPipe FaceMesh (Apache 2.0)")
            print("       - Color Analysis: OpenCV (BSD)")
        except Exception as e:
            print(f"  [WARN] Face Analyzer failed to load: {e}")
            print("         Face preservation will be disabled.")
            self.face_analyzer = None

        # Preload BLIP captioner
        print("\nPreloading captioner...")
        preload_captioner()

        print("\n" + "=" * 60)
        print("Model loading complete!")
        print("  - Pixel Art Style: ✓")
        print("  - Depth Control: ✓")
        print("  - Face Attributes: ✓" if self.face_analyzer else "  - Face Attributes: ✗")
        print("  - License: All commercial-friendly")
        print("=" * 60 + "\n")

    def check_face(self, image) -> bool:
        """Check if image contains a detectable face."""
        if self.face_analyzer is None:
            return False
        return self.face_analyzer.has_face(image)


print("[OK] Model handler ready (prompt-based face preservation)")

# Module-level cached handler so get_pipeline() can return it after preload
_handler: ModelHandler | None = None


def preload_pipeline():
    """
    Convenience function: create ModelHandler, load all models, and cache it.
    Used by app.py to initialize everything on startup.
    """
    global _handler
    if _handler is None:
        _handler = ModelHandler()
        _handler.load_models()
    return _handler


def get_pipeline():
    """
    Return the loaded diffusion pipeline, loading models first if needed.
    Used by generator.py at generation time.
    """
    global _handler
    if _handler is None:
        preload_pipeline()
    return _handler.pipeline


def get_zoe_detector():
    """Return the loaded ZoeDetector for depth map generation."""
    global _handler
    if _handler is None:
        preload_pipeline()
    return _handler.zoe_detector


def get_face_analyzer():
    """Return the loaded FaceAnalyzer (or None if unavailable)."""
    global _handler
    if _handler is None:
        preload_pipeline()
    return _handler.face_analyzer