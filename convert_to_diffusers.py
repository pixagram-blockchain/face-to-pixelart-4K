"""
convert_to_diffusers.py — export horizon.safetensors to a diffusers-layout
(fp16 shards), and resolve where model.py should load the pipeline from.

Why this matters for host RAM
-----------------------------
`Pipeline.from_single_file()` on the 7 GB checkpoint loads the whole state
dict into RAM, builds converted copies of every component and only then
instantiates the models — a ~30 GB+ peak that kills the Space on the
L40S tier (62 GB, with the ControlNets/Zoe/IP-Adapter already resident).

`from_pretrained(<diffusers layout>, variant="fp16", low_cpu_mem_usage=True)`
loads shard by shard straight into the models instead: peak ≈ model size.

The conversion itself is STAGED so it also fits a modest box: UNet alone,
then VAE alone, then the pipeline with those two passed in (so
from_single_file only builds the text encoders). Peak ≈ 15–20 GB.

Two ways to use it
------------------
1. Automatic (recommended): set the Space Variable
       DIFFUSERS_LAYOUT_REPO = primerz/pixagram-horizon
   and a write-capable HF_TOKEN Secret. On the next boot model.py calls
   resolve_layout_source(): if the repo already holds the export it is used;
   otherwise the export is built once, pushed to that repo, and used. Every
   later boot (on any hardware) loads from the repo.

2. Manual:
       python convert_to_diffusers.py                        # ./horizon-diffusers
       python convert_to_diffusers.py --push primerz/pixagram-horizon

The ControlNets, LoRAs and the fp16-fix VAE are NOT part of the export —
they load separately at runtime exactly as before (the VAE is swapped in
after load, so it doesn't matter which VAE ships in the export).
"""

import argparse
import os
import shutil
import traceback

import torch

import boot_guard

# File whose presence proves an export (local or on the Hub) is complete.
MARKER_FILE = "unet/diffusion_pytorch_model.fp16.safetensors"
_PARTIAL_SUFFIX = ".partial"


# ─────────────────────────────────────────────────────────────────────────────
# Export location / completeness
# ─────────────────────────────────────────────────────────────────────────────

def local_export_dir() -> str:
    """Where the export lives on disk (on /data when persistent storage is
    mounted, so it survives restarts; otherwise a local folder)."""
    from config import Config

    return os.path.join(Config.DATA_ROOT, "horizon-diffusers")


def export_is_complete(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "model_index.json")) and os.path.isfile(
        os.path.join(path, MARKER_FILE)
    )


def hub_repo_ready(repo_id: str) -> bool:
    """True if repo_id already contains a complete fp16 export."""
    if not repo_id:
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        return api.file_exists(repo_id, "model_index.json") and api.file_exists(
            repo_id, MARKER_FILE
        )
    except Exception as exc:  # private repo without token, network, typo…
        print(f"[convert] could not check {repo_id}: {type(exc).__name__}: {exc}")
        return False


def _have_hub_token() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


# ─────────────────────────────────────────────────────────────────────────────
# Staged conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(ckpt_path: str, out_dir: str) -> str:
    """Build the diffusers layout at out_dir (atomically: written to a
    .partial folder first, renamed on success). Returns out_dir."""
    from diffusers import (
        AutoencoderKL,
        StableDiffusionXLImg2ImgPipeline,
        UNet2DConditionModel,
    )

    boot_guard.report_disk()

    # Stage 1 — UNet only. from_single_file reads the whole 7 GB dict, keeps
    # just the UNet (5 GB fp16) and drops the rest when it returns.
    print("[convert] stage 1/3: UNet ...")
    unet = UNet2DConditionModel.from_single_file(ckpt_path, torch_dtype=torch.float16)
    boot_guard.release_host_memory()
    boot_guard.report_ram("convert: after UNet")

    # Stage 2 — VAE only (small, but keeps the pipeline stage lean).
    print("[convert] stage 2/3: VAE ...")
    vae = AutoencoderKL.from_single_file(ckpt_path, torch_dtype=torch.float16)
    boot_guard.release_host_memory()
    boot_guard.report_ram("convert: after VAE")

    # Stage 3 — pipeline with unet/vae PASSED IN, so from_single_file only
    # has to build the two text encoders + tokenizers + scheduler.
    print("[convert] stage 3/3: text encoders, tokenizers, scheduler ...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        ckpt_path,
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    boot_guard.release_host_memory()
    boot_guard.report_ram("convert: after text encoders")

    partial = out_dir + _PARTIAL_SUFFIX
    shutil.rmtree(partial, ignore_errors=True)
    print(f"[convert] saving fp16 layout to {out_dir} ...")
    pipe.save_pretrained(partial, safe_serialization=True, variant="fp16")
    del pipe, unet, vae
    boot_guard.release_host_memory()

    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(partial, out_dir)
    if not export_is_complete(out_dir):
        raise RuntimeError(f"export at {out_dir} is incomplete ({MARKER_FILE} missing)")
    print("[convert] export complete")
    boot_guard.report_disk()
    return out_dir


def push(out_dir: str, repo_id: str, private: bool = True) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    print(f"[convert] uploading {out_dir} -> {repo_id} ({'private' if private else 'public'}) ...")
    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="diffusers-layout fp16 export of horizon.safetensors",
    )
    print(f"[convert] pushed to {repo_id}")


def _download_checkpoint() -> str:
    from config import Config
    from huggingface_hub import hf_hub_download

    return boot_guard.run_with_cache_repair(
        hf_hub_download,
        label=f"checkpoint {Config.CHECKPOINT_FILENAME}",
        repair_repos=(Config.REPO_ID,),
        supports_force_download=True,
        repo_id=Config.REPO_ID,
        filename=Config.CHECKPOINT_FILENAME,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Boot-time resolver used by model.py
# ─────────────────────────────────────────────────────────────────────────────

def resolve_layout_source(repo_id: str = "") -> str:
    """
    Return something `from_pretrained` can load the SDXL pipeline from:
    a complete local export dir, or a Hub repo holding one. Returns "" when
    nothing usable exists and nothing could be built — model.py then falls
    back to the RAM-hungry single-file path.

    Order:
      1. complete local export on disk            -> use it (fastest)
      2. repo_id already holds the export         -> use it
      3. conversion allowed?                      -> build locally, push if
                                                     repo_id + token, use it
    Conversion is attempted when a repo_id is given (so the result is kept
    on the Hub) or when persistent /data storage exists (so the local export
    survives). Set CONVERT_TO_DIFFUSERS=0 to forbid it.
    """
    repo_id = (repo_id or "").strip()
    local = local_export_dir()

    if export_is_complete(local):
        print(f"[convert] using local diffusers export: {local}")
        return local
    if hub_repo_ready(repo_id):
        print(f"[convert] using diffusers export from the Hub: {repo_id}")
        return repo_id

    if os.environ.get("CONVERT_TO_DIFFUSERS", "1").strip() == "0":
        print("[convert] conversion disabled (CONVERT_TO_DIFFUSERS=0)")
        return ""
    if not repo_id and not boot_guard.PERSISTENT:
        print(
            "[convert] no DIFFUSERS_LAYOUT_REPO and no persistent storage — "
            "skipping conversion (it would be redone on every boot)"
        )
        return ""

    print("[convert] no diffusers export found — building it now (one-off, ~10-15 min)")
    try:
        ckpt = _download_checkpoint()
        convert(ckpt, local)
        if repo_id:
            if _have_hub_token():
                private = os.environ.get("DIFFUSERS_LAYOUT_PRIVATE", "1").strip() != "0"
                push(local, repo_id, private=private)
            else:
                print(
                    f"[convert] WARNING: no HF token — export kept locally only, "
                    f"not pushed to {repo_id}. Add a write-capable HF_TOKEN Secret."
                )
        return local
    except Exception:
        print("[convert] conversion FAILED — falling back to single-file loading")
        traceback.print_exc()
        shutil.rmtree(local + _PARTIAL_SUFFIX, ignore_errors=True)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI (manual use)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="./horizon-diffusers",
        help="Local output directory for the diffusers layout.",
    )
    parser.add_argument(
        "--push", default="",
        help="Optional hub repo id to push to (e.g. primerz/pixagram-horizon). "
             "Requires a logged-in token with write access.",
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Create the pushed repo as public (default: private).",
    )
    args = parser.parse_args()

    print("[1/3] Downloading checkpoint ...")
    ckpt = _download_checkpoint()
    print("[2/3] Converting (staged, low-RAM) ...")
    convert(ckpt, os.path.abspath(args.out))
    if args.push:
        print("[3/3] Pushing ...")
        push(os.path.abspath(args.out), args.push, private=not args.public)
        print(f"Done. Set Space Variable DIFFUSERS_LAYOUT_REPO = {args.push}")
    else:
        print(
            "[3/3] Done. Upload the folder to a hub repo, then set the Space "
            "Variable DIFFUSERS_LAYOUT_REPO to that repo id."
        )


if __name__ == "__main__":
    main()
