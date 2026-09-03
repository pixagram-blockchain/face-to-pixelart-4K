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

# boot_guard MUST come before torch/diffusers/controlnet_aux: it places the
# HF cache (on /data when persistent storage exists) before huggingface_hub
# freezes its paths, and provides the self-healing loaders used below.
# app.py imports it even earlier (before gradio/spaces); this line covers
# importing model.py standalone.
import boot_guard

import torch
from PIL import Image
from diffusers import ControlNetModel, LCMScheduler

# TF32 for whatever still runs in fp32 (the Zoe depth model, any residual
# fp32 matmuls): free speedup on Ampere+ GPUs with no quality impact at this
# precision level. No-op on CPU / pre-Ampere.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Tiles are uniform squares and rung shapes repeat, so cudnn autotuning
# pays for itself after the first occurrence of each shape.
torch.backends.cudnn.benchmark = True

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
        # Global handles on the ControlNets. These are the SAME module objects
        # the pipeline's MultiControlNet wraps (pipeline.controlnet.nets), not
        # copies — no extra VRAM. Order matches control_images /
        # controlnet_conditioning_scale everywhere: [identity, depth].
        self.identity_controlnet = None
        self.depth_controlnet = None
        self.controlnets = None

    def load_models(self):
        print("=" * 60)
        print("Loading Pixagram Pixel Art Models")
        print("  (InstantID face identity: IdentityNet + IP-Adapter)")
        print("=" * 60)

        # Dirty-disk hygiene: report free space and clear stale download
        # locks / partial files left by an interrupted previous boot. Those
        # leftovers are exactly what makes a plain Restart fail while a
        # Factory reboot (clean disk) works.
        boot_guard.report_disk()
        boot_guard.sweep_stale_files()

        # 1. Load Zoe Depth Detector
        print("\n[1/5] Loading Zoe Depth Detector...")
        try:
            # controlnet_aux's from_pretrained doesn't accept force_download,
            # so repair escalates straight to purging the annotator repo.
            self.zoe_detector = boot_guard.run_with_cache_repair(
                ZoeDetector.from_pretrained,
                Config.ANNOTATOR_REPO,
                label="ZoeDetector",
                repair_repos=(Config.ANNOTATOR_REPO,),
                supports_force_download=False,
            )
            self.zoe_detector.to(Config.DEVICE)
            boot_guard.release_host_memory()  # Zoe's fp32 weights now live on GPU
            boot_guard.report_ram("after Zoe")
            print("  [OK] ZoeDetector loaded")
        except Exception as e:
            print(f"  [ERROR] Failed to load ZoeDetector: {e}")
            raise

        # 2. Load ControlNets: IdentityNet (face identity/structure) + Depth
        print("\n[2/5] Loading ControlNets (InstantID IdentityNet + Zoe Depth)...")
        identity_controlnet = boot_guard.run_with_cache_repair(
            ControlNetModel.from_pretrained,
            Config.INSTANTID_REPO,
            label="IdentityNet ControlNet",
            repair_repos=(Config.INSTANTID_REPO,),
            supports_force_download=True,
            subfolder="ControlNetModel",
            torch_dtype=Config.DTYPE,
        )
        print("  [OK] IdentityNet loaded (InstantID face structure/identity)")
        # HOST-RAM: ship each ControlNet to the GPU the moment it loads.
        # They used to sit in host RAM (~5 GB together) through the whole
        # pipeline load — which is exactly when the RAM peak happens.
        identity_controlnet = identity_controlnet.to(Config.DEVICE)
        boot_guard.release_host_memory()
        depth_controlnet = boot_guard.run_with_cache_repair(
            ControlNetModel.from_pretrained,
            Config.CN_ZOE_REPO,
            label="Zoe depth ControlNet",
            repair_repos=(Config.CN_ZOE_REPO,),
            supports_force_download=True,
            torch_dtype=Config.DTYPE,
        )
        print("  [OK] ControlNet Depth loaded")
        depth_controlnet = depth_controlnet.to(Config.DEVICE)
        boot_guard.release_host_memory()
        boot_guard.report_ram("after ControlNets")

        # IMPORTANT: order is [identity, depth] and MUST match the order of
        # the control-image list and conditioning-scale list passed by
        # generator.py / tiled_pipeline.py.
        controlnets = [identity_controlnet, depth_controlnet]
        self.identity_controlnet = identity_controlnet
        self.depth_controlnet = depth_controlnet
        self.controlnets = controlnets

        # 3. Load Base Pipeline
        print("\n[3/5] Loading SDXL InstantID Pipeline...")
        # HOST-RAM: prefer a diffusers-layout export (fp16 shards, loaded
        # component by component with low_cpu_mem_usage) over parsing the
        # 7GB single-file checkpoint, whose whole-dict + converted-copies
        # peak is what exhausts the L40S tier's RAM. The resolver returns a
        # complete local export, a Hub repo holding one, or builds the export
        # once (staged, low-RAM) when DIFFUSERS_LAYOUT_REPO / persistent
        # storage allow it. Env var wins over Config so the switch needs no
        # code change. See convert_to_diffusers.py.
        from convert_to_diffusers import resolve_layout_source

        layout_source = resolve_layout_source(
            os.environ.get("DIFFUSERS_LAYOUT_REPO", "").strip()
            or Config.DIFFUSERS_LAYOUT_REPO
        )
        if layout_source:
            is_local = os.path.isdir(layout_source)
            self.pipeline = boot_guard.run_with_cache_repair(
                StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained,
                layout_source,
                label="SDXL pipeline (diffusers layout)",
                repair_repos=() if is_local else (layout_source,),
                supports_force_download=not is_local,
                controlnet=controlnets,
                torch_dtype=Config.DTYPE,
                variant="fp16",
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            print(f"  [OK] Pipeline loaded from diffusers layout: {layout_source}")
        else:
            print(
                "  [WARN] No diffusers-layout export available — using the "
                "RAM-hungry from_single_file path. To fix: set the Space "
                "Variable DIFFUSERS_LAYOUT_REPO (e.g. primerz/pixagram-horizon) "
                "plus a write-capable HF_TOKEN Secret and reboot once."
            )

            # Download + parse are repaired TOGETHER: a truncated cached
            # checkpoint makes from_single_file fail, and only re-downloading
            # the file fixes it — so the retry has to cover both steps.
            def _download_and_parse_checkpoint(force_download: bool = False):
                checkpoint_path = hf_hub_download(
                    repo_id=Config.REPO_ID,
                    filename=Config.CHECKPOINT_FILENAME,
                    force_download=force_download,
                )
                print(f"  [OK] Checkpoint ready: {Config.CHECKPOINT_FILENAME}")
                return StableDiffusionXLInstantIDImg2ImgPipeline.from_single_file(
                    checkpoint_path,
                    controlnet=controlnets,
                    torch_dtype=Config.DTYPE,
                    use_safetensors=True,
                )

            self.pipeline = boot_guard.run_with_cache_repair(
                _download_and_parse_checkpoint,
                label=f"SDXL pipeline ({Config.CHECKPOINT_FILENAME})",
                repair_repos=(Config.REPO_ID,),
                supports_force_download=True,
            )
        boot_guard.release_host_memory()  # drop the parsed state dict(s)

        # HOST-RAM: move the pipeline to the GPU NOW, before the IP-Adapter,
        # LoRAs and scheduler are attached (they all work on a CUDA pipeline:
        # load_ip_adapter_instantid targets self.unet.device, PEFT LoRAs load
        # onto whatever device the Linear layers are on). This frees the
        # ~7 GB fp16 pipeline from host RAM right after it is built instead
        # of holding it through every subsequent load.
        print(f"  Moving pipeline to {Config.DEVICE}...")
        self.pipeline.to(Config.DEVICE)
        boot_guard.release_host_memory()
        print("  [OK] Pipeline loaded (MultiControlNet: identity + depth)")

        # Swap in the fp16-fix VAE (force_upcast=False). The stock SDXL VAE
        # upcasts the ENTIRE VAE fp16 -> fp32 on every encode/decode and
        # casts it back afterwards — the tiled path paid that dtype churn
        # PER TILE, on top of running the VAE in fp32. fp16-fix is
        # numerically safe in fp16: same latent space, no churn, ~half the
        # VAE cost and memory.
        if Config.USE_FP16_FIX_VAE and Config.DTYPE == torch.float16:
            from diffusers import AutoencoderKL
            self.pipeline.vae = boot_guard.run_with_cache_repair(
                AutoencoderKL.from_pretrained,
                Config.VAE_REPO,
                label="fp16-fix VAE",
                repair_repos=(Config.VAE_REPO,),
                supports_force_download=True,
                torch_dtype=Config.DTYPE,
            ).to(Config.DEVICE)  # pipeline already lives on the GPU
            boot_guard.release_host_memory()
            print(f"  [OK] VAE swapped: {Config.VAE_REPO} (fp16, no upcast)")
        print(
            f"  [INFO] VAE in use: force_upcast="
            f"{getattr(self.pipeline.vae.config, 'force_upcast', None)} "
            f"dtype={self.pipeline.vae.dtype}"
        )

        # InstantID IP-Adapter: projects the ArcFace embedding into
        # UNet cross-attention tokens (the likeness component).
        print("\n  Loading InstantID IP-Adapter...")
        ip_adapter_path = boot_guard.run_with_cache_repair(
            hf_hub_download,
            label="InstantID ip-adapter.bin",
            repair_repos=(Config.INSTANTID_REPO,),
            supports_force_download=True,
            repo_id=Config.INSTANTID_REPO,
            filename="ip-adapter.bin",
        )
        self.pipeline.load_ip_adapter_instantid(ip_adapter_path)
        self.pipeline.set_ip_adapter_scale(Config.DEFAULT_IP_ADAPTER_SCALE)
        print(
            f"  [OK] IP-Adapter loaded (default scale "
            f"{Config.DEFAULT_IP_ADAPTER_SCALE})"
        )

        # Load LoRA styles (every selectable adapter, loaded unfused).
        # IMPORTANT: do NOT fuse. Fusing bakes a single fixed scale into the
        # UNet with no per-request differentiation. Keeping them as live PEFT
        # adapters lets generator.py pick the style AND its strength per
        # request via set_adapters(). PEFT wraps the attention Linear layers,
        # so it coexists cleanly with the InstantID IPAttnProcessors (which
        # call those same wrapped Linears).
        print("\n  Loading LoRA weights...")
        self.loaded_lora_styles = []
        for style, spec in Config.LORA_STYLES.items():
            try:
                # repair_repos deliberately EMPTY: these files share
                # Config.REPO_ID with the 7GB checkpoint, and a repo purge
                # for a small LoRA would force a needless multi-GB
                # re-download. force_download retries cover file corruption.
                lora_path = boot_guard.run_with_cache_repair(
                    hf_hub_download,
                    label=f"LoRA {spec['weight_name']}",
                    repair_repos=(),
                    supports_force_download=True,
                    repo_id=Config.REPO_ID,
                    filename=spec["weight_name"],
                )
                self.pipeline.load_lora_weights(lora_path, adapter_name=style)
                self.loaded_lora_styles.append(style)
                print(
                    f"  [OK] LoRA '{style}' loaded (unfused): "
                    f"{spec['weight_name']}"
                )
            except Exception as e:
                print(
                    f"  [WARN] LoRA '{style}' NOT loaded "
                    f"({spec['weight_name']} in {Config.REPO_ID}): {e}"
                )
        if not self.loaded_lora_styles:
            raise RuntimeError(
                "No LoRA style could be loaded — check Config.LORA_STYLES "
                f"weight files in {Config.REPO_ID}."
            )
        # Activate the default style (or the first loaded one). Only ONE
        # adapter is active at a time; set_adapters() replaces the active
        # set, so the non-selected style contributes nothing.
        active = (
            Config.DEFAULT_LORA_STYLE
            if Config.DEFAULT_LORA_STYLE in self.loaded_lora_styles
            else self.loaded_lora_styles[0]
        )
        self.pipeline.set_adapters(
            [active], adapter_weights=[Config.DEFAULT_LORA_INTENSITY]
        )
        print(
            f"  [OK] Active LoRA: '{active}' (unfused, "
            f"default scale {Config.DEFAULT_LORA_INTENSITY}); "
            f"available: {self.loaded_lora_styles}"
        )

        # Setup scheduler
        print("\n  Configuring LCM Scheduler...")
        self.pipeline.scheduler = LCMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        print("  [OK] LCM Scheduler configured")

        # Pipeline was moved to the device right after it was built (see
        # above); this is a no-op safety net for anything attached since.
        self.pipeline.to(Config.DEVICE)
        boot_guard.release_host_memory()
        boot_guard.report_ram("after pipeline on GPU")

        # channels_last for the conv-heavy modules: faster convolutions on
        # Ampere/Ada in fp16, layout-only (numerically equivalent). The
        # attention processors are unaffected (they work on 3D tensors).
        if Config.USE_CHANNELS_LAST and Config.DEVICE == "cuda":
            self.pipeline.unet.to(memory_format=torch.channels_last)
            self.pipeline.vae.to(memory_format=torch.channels_last)
            for _net in getattr(
                self.pipeline.controlnet, "nets", [self.pipeline.controlnet]
            ):
                _net.to(memory_format=torch.channels_last)
            print("  [OK] channels_last enabled (unet + controlnets + vae)")

        self.pipeline.enable_vae_slicing()
        # NOTE: enable_vae_tiling() is deliberately OFF now. The full canvas
        # is never decoded in one go (the base pass tops out at
        # BASE_PASS_LONG_EDGE and refine tiles are stitched in PIXEL space),
        # while the VAE tiler's 512px default threshold was chopping every
        # 768px tile decode into overlapping windows — pure overhead plus a
        # slight blur risk inside tiles.
        # NOTE: xformers and attention slicing are deliberately NOT enabled.
        # enable_xformers_memory_efficient_attention() REPLACES the UNet's
        # attention processors — which would silently rip out the InstantID
        # IPAttnProcessors installed by load_ip_adapter_instantid() and kill
        # identity injection (fuse_qkv_projections() must never be called
        # for the same reason). The custom processors now use PyTorch SDPA
        # internally, so xformers would be a downgrade anyway. VAE slicing
        # doesn't touch the UNet.

        # 4. Load insightface antelopev2 (detection + kps + embeddings)
        print("\n[4/5] Loading insightface antelopev2...")
        try:
            from insightface.app import FaceAnalysis

            boot_guard.run_with_cache_repair(
                snapshot_download,
                label="insightface antelopev2",
                repair_repos=(Config.ANTELOPE_REPO,),
                supports_force_download=True,
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
        boot_guard.report_ram("boot complete")
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


def get_controlnets():
    """
    Return [identity_controlnet, depth_controlnet] — the live ControlNetModel
    objects wrapped by the pipeline's MultiControlNet (same objects as
    pipeline.controlnet.nets, same order as the control_images /
    controlnet_conditioning_scale lists). Loads models first if needed.
    """
    global _handler
    if _handler is None:
        preload_pipeline()
    return _handler.controlnets


def get_identity_controlnet():
    """Return the InstantID IdentityNet ControlNetModel (global handle)."""
    global _handler
    if _handler is None:
        preload_pipeline()
    return _handler.identity_controlnet


def get_depth_controlnet():
    """Return the Zoe depth ControlNetModel (global handle)."""
    global _handler
    if _handler is None:
        preload_pipeline()
    return _handler.depth_controlnet


def get_loaded_lora_styles():
    """Return the list of LoRA style keys that actually loaded at startup."""
    global _handler
    if _handler is None:
        preload_pipeline()
    return list(_handler.loaded_lora_styles)
