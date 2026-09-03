"""
boot_guard — makes Space startup survive dirty disk state.

Why this module exists
----------------------
A plain "Restart" on Hugging Face Spaces reuses the already-built image and
can inherit whatever the previous run left on disk: a Hub cache with stale
``*.lock`` / ``*.incomplete`` files from an interrupted download, a truncated
safetensors blob, or a nearly-full root filesystem. Any of those makes the
un-guarded ``from_pretrained`` / ``hf_hub_download`` calls in model.py crash
at boot — which is exactly the "works after Factory reboot, dies on Restart"
pattern. This module gives the Space three defenses:

1. **Cache placement** (at import time): if persistent storage (``/data``) is
   mounted and writable, route ``HF_HOME`` and ``TMPDIR`` there — unless they
   are already set as Space Variables in Settings (env always wins). With the
   cache on ``/data``, restarts stop re-downloading ~15 GB of weights at all.

2. **Stale-state sweep**: delete leftover ``*.lock`` / ``*.incomplete`` files
   in the Hub cache before loading anything.

3. **Self-healing loaders**: ``run_with_cache_repair()`` wraps a loading call
   and, on failure, escalates — sweep locks and retry (with
   ``force_download=True`` where supported, which re-fetches just the needed
   files and overwrites corrupted blobs), then as a last resort purge the
   affected repo from the cache and download it fresh. What used to need a
   manual Factory reboot now repairs itself on the next Restart.

IMPORT ORDER MATTERS: import this module BEFORE gradio / spaces / diffusers /
transformers / controlnet_aux. They all import ``huggingface_hub``, which
freezes its cache paths from the environment at import time — so the env
setup below has to run first. app.py imports boot_guard as its first import.

This module must never be able to crash a boot itself: every action is
wrapped so that, at worst, it silently does nothing.
"""

import errno
import os
import shutil
import time
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cache placement — runs at import time, before huggingface_hub is loaded
# ─────────────────────────────────────────────────────────────────────────────

def _writable_dir(path: str) -> bool:
    try:
        return os.path.isdir(path) and os.access(path, os.W_OK)
    except OSError:
        return False


PERSISTENT = _writable_dir("/data")

if PERSISTENT:
    # setdefault: a Space Variable set in Settings > Variables always wins.
    # ONLY the model cache goes to /data. Temp data — which includes
    # Gradio's cache of uploaded and generated images — must stay on the
    # ephemeral disk, so a user photo never touches persistent storage.
    os.environ.setdefault("HF_HOME", "/data/.huggingface")

# Pin Gradio's cache to an ephemeral path explicitly (it derives its
# default from tempfile.gettempdir(), i.e. from TMPDIR).
os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/gradio")

try:
    os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)
    if "HF_HOME" in os.environ:
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)
except OSError:
    pass

# Migration: an earlier revision of this module routed TMPDIR to /data/tmp,
# which let Gradio cache user images on persistent storage. Remove any such
# leftovers and never recreate the directory.
if PERSISTENT and os.path.isdir("/data/tmp"):
    shutil.rmtree("/data/tmp", ignore_errors=True)
    print("[boot_guard] removed /data/tmp (user images do not belong on persistent storage)")


def hub_cache_dir() -> str:
    """Resolve the Hub cache dir the way huggingface_hub will, without
    importing it (importing it here would freeze paths too early)."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    hf_home = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    return os.path.join(hf_home, "hub")


print(
    f"[boot_guard] HF cache: {hub_cache_dir()} "
    f"({'persistent /data' if PERSISTENT else 'EPHEMERAL — re-downloads on every boot'})"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Disk hygiene helpers
# ─────────────────────────────────────────────────────────────────────────────

def free_gb(path: str = "/") -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1.0


def _fmt_free(path: str) -> str:
    gb = free_gb(path)
    if gb < 0:
        return "unknown"
    # Network-backed mounts (like /data) often report absurd free space
    # (exabytes). Beyond any plausible quota the number carries no signal.
    if gb > 10_000:
        return "not meaningful (network mount)"
    return f"{gb:.1f} GB"


def report_disk() -> None:
    try:
        msg = f"[boot_guard] free disk: / = {_fmt_free('/')}"
        if PERSISTENT:
            msg += f", /data = {_fmt_free('/data')}"
        print(msg)
    except Exception:
        pass


def sweep_stale_files() -> int:
    """Delete ``*.lock`` and ``*.incomplete`` files anywhere under the Hub
    cache. These are left behind by interrupted downloads and are the classic
    reason a Restart fails where a Factory reboot works."""
    removed = 0
    root = hub_cache_dir()
    try:
        if not os.path.isdir(root):
            return 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith(".lock") or fname.endswith(".incomplete"):
                    try:
                        os.remove(os.path.join(dirpath, fname))
                        removed += 1
                    except OSError:
                        pass
        if removed:
            print(f"[boot_guard] removed {removed} stale lock/partial file(s) in {root}")
    except Exception:
        pass
    return removed


def release_host_memory() -> None:
    """Give freed host RAM back to the OS. After a multi-GB state dict is
    dropped, glibc's allocator keeps the pages mapped (fragmentation), so
    the container's RSS — the number the Spaces memory limit looks at — stays
    high. gc + malloc_trim(0) collapses it. Harmless when nothing is free."""
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def rss_gb() -> float:
    """Resident set size of THIS process (real host RAM in use)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6  # kB -> GB
    except OSError:
        pass
    return -1.0


def _cgroup_read(name: str) -> int:
    for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        try:
            with open(os.path.join(base, name)) as fh:
                v = fh.read().strip()
                return -1 if v == "max" else int(v)
        except (OSError, ValueError):
            continue
    return -1


def report_ram(tag: str = "") -> None:
    """One log line that separates REAL memory use from cache inflation:
    process RSS vs the container counter (which includes reclaimable file
    cache from reading/writing weights — the number the Spaces metrics
    graph shows, and the one that looks scary without being dangerous)."""
    try:
        parts = [f"process RSS {rss_gb():.1f} GB"]
        current, limit = _cgroup_read("memory.current"), _cgroup_read("memory.max")
        if current >= 0:
            msg = f"container {current / 1e9:.1f} GB"
            if limit > 0:
                msg += f" of {limit / 1e9:.0f} GB"
            file_cache = -1
            for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
                try:
                    with open(os.path.join(base, "memory.stat")) as fh:
                        for line in fh:
                            if line.startswith("file "):
                                file_cache = int(line.split()[1]); break
                    break
                except OSError:
                    continue
            if file_cache >= 0:
                msg += f" (file cache {file_cache / 1e9:.1f} GB — reclaimable)"
            parts.append(msg)
        print(f"[boot_guard] RAM {tag}: " + " | ".join(parts))
    except Exception:
        pass


def after_job_cleanup(tag: str = "after job") -> None:
    """Call after each generation: return freed host pages to the OS and
    release cached VRAM blocks, then log where memory actually stands."""
    release_host_memory()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    report_ram(tag)


def clear_tmp() -> None:
    """Empty $TMPDIR and gradio's temp dir (safe at boot — nothing of ours is
    live there yet). Frees space when the disk filled up."""
    targets = {
        os.environ.get("TMPDIR", "/tmp"),
        os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio"),
        "/tmp/gradio",
    }
    for target in targets:
        try:
            if not os.path.isdir(target):
                continue
            for entry in os.listdir(target):
                path = os.path.join(target, entry)
                try:
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                except OSError:
                    pass
            print(f"[boot_guard] cleared temp dir: {target}")
        except Exception:
            pass


def purge_cached_repo(repo_id: str) -> bool:
    """Remove every cached copy of ``repo_id`` from the Hub cache so the next
    download starts fresh. Last-resort repair for a corrupted cache entry."""
    ok = False
    try:
        from huggingface_hub import scan_cache_dir  # safe: we're mid-boot now

        info = scan_cache_dir(hub_cache_dir())
        hashes = [
            rev.commit_hash
            for repo in info.repos
            if repo.repo_id == repo_id
            for rev in repo.revisions
        ]
        if hashes:
            info.delete_revisions(*hashes).execute()
            print(f"[boot_guard] purged cached {repo_id} ({len(hashes)} revision(s))")
            ok = True
    except Exception as exc:
        print(f"[boot_guard] scan_cache_dir purge failed for {repo_id}: {exc}")
    if not ok:
        # Fallback: remove the repo folder directly (models--org--name layout).
        folder = os.path.join(hub_cache_dir(), "models--" + repo_id.replace("/", "--"))
        try:
            if os.path.isdir(folder):
                shutil.rmtree(folder, ignore_errors=True)
                print(f"[boot_guard] removed {folder}")
                ok = True
        except Exception:
            pass
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 3. Self-healing call wrapper
# ─────────────────────────────────────────────────────────────────────────────

_CORRUPTION_MARKERS = (
    "headertoolarge",
    "error while deserializing header",
    "safetensorerror",
    "metadataincompletebuffer",
    "eoferror",
    "unexpected end of",
    "unexpected eof",
    "corrupt",
    "consistency check failed",
    "incomplete",
    "checksum",
    "invalid load key",
    "pytorchstreamreader failed",
    "no such file or directory",           # broken symlink inside the cache
    "does not appear to have a file named",  # partial snapshot in the cache
)


def _is_enospc(exc: BaseException) -> bool:
    seen = set()
    node = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, OSError) and node.errno == errno.ENOSPC:
            return True
        node = node.__cause__ or node.__context__
    return "no space left on device" in f"{exc!r}".lower()


def _looks_like_corruption(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _CORRUPTION_MARKERS)


def run_with_cache_repair(
    fn,
    *args,
    label: str = "",
    repair_repos=(),
    supports_force_download: bool = False,
    retries: int = 2,
    retry_delay: float = 4.0,
    **kwargs,
):
    """
    Call ``fn(*args, **kwargs)`` with dirty-disk self-repair between attempts.

    Attempt 1  plain call (normal fast path — cache hits cost nothing).
    Attempt 2  sweep stale locks/partials (and clear temp if the disk is
               full), then retry — with ``force_download=True`` when
               ``supports_force_download`` is set, which re-fetches exactly
               the files ``fn`` needs and overwrites any corrupted blob.
    Attempt 3  last resort: purge ``repair_repos`` from the Hub cache
               entirely (only when the error looks like corruption, a broken
               cache, or a full disk — never for plain code bugs), then retry
               with a fresh download.

    ``repair_repos`` should be left EMPTY for files that share a repo with
    much larger artifacts (e.g. a small LoRA living next to a 7 GB
    checkpoint), so a purge can't trigger a needless multi-GB re-download.
    """
    name = label or getattr(fn, "__name__", str(fn))
    last_exc = None
    for attempt in range(retries + 1):
        call_kwargs = dict(kwargs)
        try:
            if attempt >= 1 and supports_force_download:
                call_kwargs["force_download"] = True
            result = fn(*args, **call_kwargs)
            if attempt:
                print(f"[boot_guard] {name}: recovered on attempt {attempt + 1}")
            return result
        except Exception as exc:  # noqa: BLE001 — we re-raise after retries
            last_exc = exc
            if attempt >= retries:
                break
            print(
                f"[boot_guard] {name}: attempt {attempt + 1} failed "
                f"({type(exc).__name__}: {exc}) — repairing and retrying"
            )
            traceback.print_exc()
            sweep_stale_files()
            if _is_enospc(exc):
                clear_tmp()
                report_disk()
            # Before the FINAL attempt, nuke the affected repos from the
            # cache — but only when the failure plausibly lives on disk.
            if (
                attempt + 1 == retries
                and repair_repos
                and (
                    _looks_like_corruption(exc)
                    or _is_enospc(exc)
                    or isinstance(exc, OSError)
                )
            ):
                for repo_id in repair_repos:
                    purge_cached_repo(repo_id)
            time.sleep(retry_delay * (attempt + 1))
    raise last_exc
