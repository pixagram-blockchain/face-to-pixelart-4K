"""
Model loading for Pixagram Pixel Art Generator
With InstantID Face Identity (IdentityNet + IP-Adapter)

This module loads:
1. SDXL Pipeline (InstantID img2img variant) with MultiControlNet
   [IdentityNet, Zoe Depth] from the single-file checkpoint
2. The InstantID IP-Adapter (identity embedding -> UNet cross-attention)
3. Pixel Art LoRA (unfused PEFT adapter, live per-request scale)
4. insightface antelopev2 (SCRFD detection + ArcFace embeddings + kps)

LICENSE NOTE: insightface antelopev2 model weights are released for
NON-COMMERCIAL research use (InstantID code itself is Apache-2.0).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from diffusers import ControlNetModel, LCMScheduler

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

    class _StubModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
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
from huggingface_hub import hf_hub_download, snapshot_download
from config import Config
from utils import preload_captioner
from pipeline_stable_diffusion_xl_instantid_img2img import (
    StableDiffusionXLInstantIDImg2ImgPipeline,
)


class ModelHandler:
    """
    Handles loading and management of all models.

    Components:
    - SDXL InstantID img2img Pipeline with MultiControlNet [identity, depth]
    - InstantID IP-Adapter
    - Zoe Depth Detector
    - LoRA Weights (unfused PEFT adapter)
    - insightface antelopev2 FaceAnalysis (detection + kps + embeddings)
    """

    def __init__(self):
        self.pipeline = None
        self.zoe_detector = None
        self.face_app = None

    def load_models(self):
        print("=" * 60)
        print("Loading Pixagram Pixel Art Models")
        print("  (InstantID face identity: IdentityNet + IP-Adapter)")
        print("=" * 60)

        # 1. Load Zoe Depth Detector
        print("\n[1/5] Loading Zoe Depth Detector...")
        try:
            self.zoe_detector = ZoeDetector.from_pretrained(Config.ANNOTATOR_REPO)
            self.zoe_detector.to(Config.DEVICE)
            print("  [OK] ZoeDetector loaded")
        except Exception as e:
            print(f"  [ERROR] Failed to load ZoeDetector: {e}")
            raise

        # 2. Load ControlNets: IdentityNet (face identity/structure) + Depth
        print("\n[2/5] Loading ControlNets (InstantID IdentityNet + Zoe Depth)...")
        identity_controlnet = ControlNetModel.from_pretrained(
            Config.INSTANTID_REPO,
            subfolder="ControlNetModel",
            torch_dtype=Config.DTYPE,
        )
        print("  [OK] IdentityNet loaded (InstantID face structure/identity)")
        depth_controlnet = ControlNetModel.from_pretrained(
            Config.CN_ZOE_REPO,
            torch_dtype=Config.DTYPE,
        )
        print("  [OK] ControlNet Depth loaded")

        # IMPORTANT: order is [identity, depth] and MUST match the order of
        # the control-image list and conditioning-scale list passed by
        # generator.py / tiled_pipeline.py.
        controlnets = [identity_controlnet, depth_controlnet]

        # 3. Load Base Pipeline with checkpoint
        print("\n[3/5] Loading SDXL InstantID Pipeline...")
        checkpoint_path = hf_hub_download(
            repo_id=Config.REPO_ID,
            filename=Config.CHECKPOINT_FILENAME
        )
        print(f"  [OK] Checkpoint downloaded: {Config.CHECKPOINT_FILENAME}")

        self.pipeline = StableDiffusionXLInstantIDImg2ImgPipeline.from_single_file(
            checkpoint_path,
            controlnet=controlnets,
            torch_dtype=Config.DTYPE,
            use_safetensors=True
        )
        print("  [OK] Pipeline loaded (MultiControlNet: identity + depth)")
        print(
            f"  [INFO] VAE in use: force_upcast="
            f"{getattr(self.pipeline.vae.config, 'force_upcast', None)} "
            f"dtype={self.pipeline.vae.dtype}"
        )

        # InstantID IP-Adapter: projects the ArcFace embedding into
        # UNet cross-attention tokens (the likeness component).
        print("\n  Loading InstantID IP-Adapter...")
        ip_adapter_path = hf_hub_download(
            repo_id=Config.INSTANTID_REPO,
            filename="ip-adapter.bin",
        )
        self.pipeline.load_ip_adapter_instantid(ip_adapter_path)
        self.pipeline.set_ip_adapter_scale(Config.DEFAULT_IP_ADAPTER_SCALE)
        print(
            f"  [OK] IP-Adapter loaded (default scale "
            f"{Config.DEFAULT_IP_ADAPTER_SCALE})"
        )

        # Load LoRA
        print("\n  Loading LoRA weights...")
        lora_path = hf_hub_download(
            repo_id=Config.REPO_ID,
            filename=Config.LORA_FILENAME
        )
        self.pipeline.load_lora_weights(lora_path, adapter_name="retroart")
        # IMPORTANT: do NOT fuse. Fusing bakes a single fixed scale into the
        # UNet with no per-request differentiation. Keeping it as a live PEFT
        # adapter lets generator.py set the style strength per request via
        # set_adapters(). PEFT wraps the attention Linear layers, so it
        # coexists cleanly with the InstantID IPAttnProcessors (which call
        # those same wrapped Linears).
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
        # NOTE: xformers and attention slicing are deliberately NOT enabled.
        # enable_xformers_memory_efficient_attention() REPLACES the UNet's
        # attention processors — which would silently rip out the InstantID
        # IPAttnProcessors installed by load_ip_adapter_instantid() and kill
        # identity injection. VAE slicing/tiling above don't touch the UNet.

        # 4. Load insightface antelopev2 (detection + kps + embeddings)
        print("\n[4/5] Loading insightface antelopev2...")
        try:
            from insightface.app import FaceAnalysis

            snapshot_download(
                repo_id=Config.ANTELOPE_REPO,
                local_dir=os.path.join(
                    Config.INSIGHTFACE_ROOT, "models", "antelopev2"
                ),
            )
            self.face_app = FaceAnalysis(
                name="antelopev2",
                root=Config.INSIGHTFACE_ROOT,
                providers=["CPUExecutionProvider"],
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("  [OK] antelopev2 ready (SCRFD det + ArcFace embeddings)")
            print("       LICENSE: non-commercial research use (InsightFace)")
        except Exception as e:
            print(f"  [WARN] insightface failed to load: {e}")
            print("         InstantID identity will be disabled (style-only).")
            self.face_app = None

        # 5. Optional BLIP captioner. OFF by default (Config.USE_CAPTION):
        # the caption was the main source of phantom faces; skipping it also
        # avoids a ~1GB download on startup.
        if Config.USE_CAPTION:
            print("\n[5/5] Preloading captioner...")
            preload_captioner()
        else:
            print(
                "\n[5/5] BLIP captioning disabled (USE_CAPTION=False) — "
                "generation uses trigger word(s) only."
            )

        print("\n" + "=" * 60)
        print("Model loading complete!")
        print("  - Pixel Art Style: ✓")
        print("  - Depth Control: ✓")
        print("  - InstantID Identity: ✓" if self.face_app else "  - InstantID Identity: ✗ (no insightface)")
        print("  - NOTE: antelopev2 weights are non-commercial (InsightFace)")
        print("=" * 60 + "\n")

    def check_face(self, image) -> bool:
        """Check if image contains a detectable face."""
        if self.face_app is None:
            return False
        try:
            import cv2
            import numpy as np
            bgr = cv2.cvtColor(
                np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR
            )
            return len(self.face_app.get(bgr)) > 0
        except Exception:
            return False


print("[OK] Model handler ready (InstantID face identity)")

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


def get_face_app():
    """Return the insightface FaceAnalysis app (or None if unavailable)."""
    global _handler
    if _handler is None:
        preload_pipeline()
    return _handler.face_app
