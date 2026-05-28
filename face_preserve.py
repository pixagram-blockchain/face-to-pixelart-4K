"""
Face Identity Preservation for Pixagram Pixel Art Converter.

Uses facenet-pytorch (MIT License) for face detection and embedding extraction.
Commercially safe — no InsightFace dependency.

Strategy:
  1. Detect faces in input image using MTCNN
  2. Extract 512-dim face embeddings using InceptionResnetV1 (VGGFace2)
  3. After pixel art generation, compare embeddings (cosine similarity)
  4. If identity drifted beyond threshold, blend original face features
     back into the output using a soft Gaussian mask — preserving pixel art
     style while restoring identity cues
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple, List
from dataclasses import dataclass

from config import (
    FACE_SIMILARITY_THRESHOLD,
    FACE_BLEND_STRENGTH,
    FACE_MASK_EXPAND,
    FACE_MASK_BLUR,
)

# ---- Match-validity thresholds ------------------------------------
# Below this MTCNN detection probability, a "face" is considered a
# false positive and discarded entirely.
MIN_DETECTION_PROB = 0.95

# Two faces (one original, one generated) are treated as the SAME
# person only when cosine similarity is at least this high. Below this
# they're considered different identities and no blend happens.
#
# Raised from 0.40 to 0.50. FaceNet/VGGFace2 same-person pairs are
# typically 0.55+; 0.40 admitted pairs that were probably *different*
# people, which is exactly how a stranger's face could end up smeared
# onto the wrong region of the output.
MIN_SAME_PERSON_SIM = 0.50

# Spatial sanity: matched bboxes must overlap by at least this IoU.
# Stops a face in the corner of the original being "matched" to a
# face in the center of the generated image.
#
# Raised from 0.20 to 0.40. A real same-person face barely moves
# between input and pixel-art output (img2img preserves coarse
# structure). IoU of 0.20 meant two boxes sharing only a corner could
# be paired — that's how phantom-face blends got positioned wrong.
MIN_BBOX_IOU = 0.40

# Max area ratio between matched bboxes. If the generated face is more
# than this many times smaller / larger than the original, treat as a
# spurious detection and skip blending.
#
# Tightened from 3.0 to 2.0. img2img doesn't dramatically resize faces;
# a 3x area difference almost certainly means we matched the wrong
# detection.
MAX_AREA_RATIO = 2.0

# Lazy-loaded globals (loaded on first use, kept on CPU until needed)
_mtcnn = None
_facenet = None
_facenet_installed = False


def _ensure_facenet_installed():
    """
    Install facenet-pytorch with --no-deps at runtime.
    
    facenet-pytorch pins overly strict torch/Pillow versions
    (torch<2.3, Pillow<10.3) that conflict with the HF Spaces
    ZeroGPU environment. The library itself works fine with newer
    versions, so we skip dependency resolution.
    """
    global _facenet_installed
    if _facenet_installed:
        return

    try:
        import facenet_pytorch  # noqa: F401
        _facenet_installed = True
        return
    except ImportError:
        pass

    import subprocess
    import sys

    print("[FacePreserve] Installing facenet-pytorch (--no-deps to avoid version conflicts)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "facenet-pytorch>=2.5.3",
        "--no-deps",
        "--quiet",
    ])
    _facenet_installed = True
    print("[FacePreserve] facenet-pytorch installed successfully.")


@dataclass
class FaceInfo:
    """Detected face with bounding box and embedding."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: torch.Tensor           # 512-dim normalized embedding
    cropped: torch.Tensor              # Cropped face tensor
    det_prob: float = 1.0              # MTCNN detection probability


def _load_models():
    """Lazy-load MTCNN and InceptionResnetV1 to CPU."""
    global _mtcnn, _facenet

    if _mtcnn is not None and _facenet is not None:
        return

    _ensure_facenet_installed()
    from facenet_pytorch import MTCNN, InceptionResnetV1

    print("[FacePreserve] Loading MTCNN face detector (MIT license)...")
    _mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=True,
        post_process=True,       # Normalize for InceptionResnetV1
        device="cpu",
    )

    print("[FacePreserve] Loading InceptionResnetV1 (VGGFace2, MIT license)...")
    _facenet = InceptionResnetV1(pretrained="vggface2").eval()

    print("[FacePreserve] Face models loaded successfully.")


def _ensure_on_device(device: str = "cuda"):
    """Move face models to the specified device if needed."""
    global _mtcnn, _facenet
    _load_models()

    target = torch.device(device)
    current_device = next(_facenet.parameters()).device
    if current_device != target:
        print(f"[FacePreserve] Moving face models to {device}...")
        _facenet = _facenet.to(target)

        # MTCNN has internal networks (pnet, rnet, onet) that must be
        # moved individually — just setting .device is not enough.
        _mtcnn.device = target
        for net_name in ("pnet", "rnet", "onet"):
            net = getattr(_mtcnn, net_name, None)
            if net is not None:
                net.to(target)


def detect_faces(
    image: Image.Image,
    device: str = "cuda",
    min_prob: float = MIN_DETECTION_PROB,
) -> List[FaceInfo]:
    """
    Detect faces and extract embeddings from a PIL Image.

    Low-confidence detections (MTCNN probability < `min_prob`) are
    discarded — MTCNN can hallucinate "faces" in stylized pixel-art
    output (windows, abstract shapes, etc.) and we don't want those
    polluting downstream identity matching.

    Args:
        image: Input PIL Image (RGB)
        device: Device for inference
        min_prob: Minimum MTCNN detection probability to keep a face.

    Returns:
        List of FaceInfo with bboxes, embeddings, and detection probs.
    """
    _ensure_on_device(device)

    # MTCNN detection
    boxes, probs = _mtcnn.detect(image)

    if boxes is None or len(boxes) == 0:
        return []

    # Filter low-confidence detections before doing the expensive
    # alignment+embedding work.
    keep_idx = [
        i for i, p in enumerate(probs)
        if p is not None and float(p) >= min_prob
    ]
    if not keep_idx:
        return []

    # Get aligned/cropped face tensors (this re-runs MTCNN internally
    # to align all detected faces — we'll filter by index afterwards).
    faces_cropped = _mtcnn(image)
    if faces_cropped is None:
        return []

    # If single face, unsqueeze
    if faces_cropped.dim() == 3:
        faces_cropped = faces_cropped.unsqueeze(0)

    # Subset by keep_idx (guarded against length mismatch — MTCNN's two
    # call paths can disagree on count for borderline detections).
    if faces_cropped.shape[0] == len(probs):
        faces_cropped = faces_cropped[keep_idx]
        kept_boxes = [boxes[i] for i in keep_idx]
        kept_probs = [float(probs[i]) for i in keep_idx]
    else:
        # Lengths disagree. The old fallback silently let low-confidence
        # detections through with prob=1.0, which is exactly how phantom
        # faces got blended onto pixel-art textures. Instead, we re-run
        # detection on each cropped face: if the second pass also clears
        # the bar we keep it, otherwise we drop it.
        #
        # In practice the mismatch is rare and the loss of a true face
        # here is far less harmful than the false positive we used to
        # admit.
        print(
            f"[FacePreserve] count mismatch: cropped={faces_cropped.shape[0]} "
            f"vs probs={len(probs)} — refusing low-confidence fallback."
        )
        return []

    faces_cropped = faces_cropped.to(device)

    # Extract embeddings
    with torch.inference_mode():
        embeddings = _facenet(faces_cropped)

    # Normalize embeddings for cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    results = []
    for box, emb, crop, prob in zip(kept_boxes, embeddings, faces_cropped, kept_probs):
        bbox = tuple(int(c) for c in box)
        results.append(FaceInfo(
            bbox=bbox,
            embedding=emb.cpu(),
            cropped=crop.cpu(),
            det_prob=prob,
        ))

    return results


def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two face embeddings.

    Returns:
        Similarity score in [-1, 1]; higher = more similar.
        Typical same-person threshold is ~0.6-0.7.
    """
    return torch.dot(emb1.flatten(), emb2.flatten()).item()


def _create_soft_face_mask(
    image_size: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    expand_px: int = FACE_MASK_EXPAND,
    blur_radius: float = FACE_MASK_BLUR,
) -> Image.Image:
    """
    Create a soft Gaussian-blurred mask for the face region.

    Args:
        image_size: (width, height) of the full image
        bbox: Face bounding box (x1, y1, x2, y2)
        expand_px: Number of pixels to expand the bbox on each side.
                   PREVIOUSLY this was misinterpreted as a multiplier
                   (FACE_MASK_EXPAND=20 meaning "20x bigger bbox"),
                   which silently masked the entire image and caused
                   the original to bleed over wide regions of the
                   generated output. Now correctly interpreted as a
                   per-edge pixel padding.
        blur_radius: Gaussian blur radius for soft edges

    Returns:
        Grayscale PIL Image mask (white = face region)
    """
    w, h = image_size
    x1, y1, x2, y2 = bbox

    # Expand the bounding box by `expand_px` on every side
    x1 = max(0, int(x1) - int(expand_px))
    y1 = max(0, int(y1) - int(expand_px))
    x2 = min(w, int(x2) + int(expand_px))
    y2 = min(h, int(y2) + int(expand_px))

    # Create mask
    mask = Image.new("L", image_size, 0)
    mask_np = np.array(mask)
    mask_np[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_np)

    # Apply Gaussian blur for soft edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask


def _bbox_iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _is_valid_match(
    orig_face: "FaceInfo",
    gen_face: "FaceInfo",
    similarity: float,
) -> Tuple[bool, str]:
    """
    Decide whether (orig_face, gen_face) is a real same-person match.

    Returns (ok, reason). All three conditions must hold:
      1. Embedding cosine similarity above MIN_SAME_PERSON_SIM
      2. Bounding boxes overlap (IoU >= MIN_BBOX_IOU)
      3. Bbox area ratio within MAX_AREA_RATIO

    Rejecting on any of these stops us from blending a phantom face
    onto the wrong region of the output.
    """
    if similarity < MIN_SAME_PERSON_SIM:
        return False, f"different identity (sim={similarity:.2f})"

    iou = _bbox_iou(orig_face.bbox, gen_face.bbox)
    if iou < MIN_BBOX_IOU:
        return False, f"misaligned (iou={iou:.2f})"

    a_orig = _bbox_area(orig_face.bbox)
    a_gen = _bbox_area(gen_face.bbox)
    if a_orig == 0 or a_gen == 0:
        return False, "zero-area bbox"
    ratio = max(a_orig, a_gen) / min(a_orig, a_gen)
    if ratio > MAX_AREA_RATIO:
        return False, f"area mismatch ({ratio:.1f}x)"

    return True, "ok"


def _greedy_assign(
    original_faces: List["FaceInfo"],
    generated_faces: List["FaceInfo"],
) -> List[Tuple[int, Optional[int], float]]:
    """
    Greedy 1-to-1 assignment of original→generated faces.

    Iterates over all (orig_i, gen_j) pairs sorted by descending
    similarity, and binds each face on at most one side. Stops a
    single generated detection from being claimed by every original.

    Returns a list of (orig_idx, gen_idx_or_None, similarity) for
    each original face.
    """
    if not original_faces or not generated_faces:
        return [(i, None, -1.0) for i in range(len(original_faces))]

    # Score every pair.
    pairs: List[Tuple[float, int, int]] = []
    for oi, of in enumerate(original_faces):
        for gi, gf in enumerate(generated_faces):
            sim = compute_similarity(of.embedding, gf.embedding)
            pairs.append((sim, oi, gi))
    pairs.sort(key=lambda p: p[0], reverse=True)

    orig_to_gen: dict = {}
    used_gen = set()
    for sim, oi, gi in pairs:
        if oi in orig_to_gen or gi in used_gen:
            continue
        orig_to_gen[oi] = (gi, sim)
        used_gen.add(gi)

    out: List[Tuple[int, Optional[int], float]] = []
    for oi in range(len(original_faces)):
        if oi in orig_to_gen:
            gi, sim = orig_to_gen[oi]
            out.append((oi, gi, sim))
        else:
            out.append((oi, None, -1.0))
    return out


def blend_face_identity(
    original: Image.Image,
    generated: Image.Image,
    original_faces: List[FaceInfo],
    generated_faces: List[FaceInfo],
    blend_strength: float = FACE_BLEND_STRENGTH,
    similarity_threshold: float = FACE_SIMILARITY_THRESHOLD,
) -> Tuple[Image.Image, List[dict]]:
    """
    Blend original face features into the generated image where identity has drifted.

    The previous implementation paired every original face with the
    *most similar* generated detection, even when that "similarity" was
    extremely low. That caused two failure modes:

      a) The generated output had no real face but MTCNN found a
         false-positive (a window, a stylised shape). The unrelated
         original then got smeared onto that spot — a face appearing
         where there isn't one.
      b) Multiple originals all matched the same gen face, so the same
         region got blended twice with different content.

    The current implementation:
      1. Uses greedy 1-to-1 assignment by similarity.
      2. Requires the match to pass _is_valid_match (embedding similarity
         AND spatial overlap AND comparable size). If any check fails
         the pair is dropped — no blend.
      3. Only blends when same-person AND identity has drifted
         (similarity < similarity_threshold).
      4. Uses the *intersection* of the two bboxes for the blend mask
         so we never paint outside where a face actually is.

    Args:
        original: Original input image
        generated: Generated pixel art image
        original_faces: Faces detected in the original
        generated_faces: Faces detected in the generated output
        blend_strength: How strongly to blend (0 = no blend, 1 = full original face)
        similarity_threshold: Below this similarity (but above same-person floor),
                              identity is considered "drifted" and we blend.

    Returns:
        Tuple of (blended image, list of per-face similarity info dicts)
    """
    if not original_faces:
        return generated, []

    # Ensure images are the same size
    if original.size != generated.size:
        original = original.resize(generated.size, Image.Resampling.LANCZOS)

    result = generated.copy()
    face_reports: List[dict] = []

    assignments = _greedy_assign(original_faces, generated_faces)

    for orig_idx, gen_idx, similarity in assignments:
        orig_face = original_faces[orig_idx]
        report = {
            "original_bbox": orig_face.bbox,
            "similarity": round(similarity, 4) if gen_idx is not None else None,
            "blended": False,
            "reason": "",
        }

        # No partner in the generated image at all.
        if gen_idx is None:
            report["reason"] = "no candidate in output"
            face_reports.append(report)
            continue

        gen_face = generated_faces[gen_idx]
        ok, reason = _is_valid_match(orig_face, gen_face, similarity)
        if not ok:
            report["reason"] = reason
            face_reports.append(report)
            continue

        # Identity might still be intact — don't blend if similarity is fine.
        if similarity >= similarity_threshold:
            report["reason"] = "identity preserved"
            face_reports.append(report)
            continue

        # ---- Build mask over the INTERSECTION of orig and gen bboxes ----
        # This is the only region we can be confident is "the face" in both
        # images. Painting outside it risks bleeding the original face onto
        # background.
        ox1, oy1, ox2, oy2 = orig_face.bbox
        gx1, gy1, gx2, gy2 = gen_face.bbox
        ix1, iy1 = max(ox1, gx1), max(oy1, gy1)
        ix2, iy2 = min(ox2, gx2), min(oy2, gy2)
        if ix2 <= ix1 or iy2 <= iy1:
            # Defensive: IoU guard above should already catch this.
            report["reason"] = "empty intersection"
            face_reports.append(report)
            continue

        mask = _create_soft_face_mask(
            result.size,
            (ix1, iy1, ix2, iy2),
        )

        # Scale blend strength: more blending when similarity is lower.
        # Above similarity_threshold this is 0 (handled by early-return).
        drift = max(0.0, similarity_threshold - similarity)
        effective_strength = min(
            1.0, blend_strength * (drift / max(similarity_threshold, 0.01))
        )

        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np *= effective_strength

        gen_np = np.array(result).astype(np.float32)
        orig_np = np.array(original).astype(np.float32)
        mask_3ch = np.stack([mask_np] * 3, axis=-1)
        blended_np = gen_np * (1.0 - mask_3ch) + orig_np * mask_3ch
        result = Image.fromarray(blended_np.clip(0, 255).astype(np.uint8))

        report["blended"] = True
        report["effective_strength"] = round(effective_strength, 4)
        report["reason"] = "identity drift corrected"
        face_reports.append(report)

    return result, face_reports


def preserve_face_identity(
    original_image: Image.Image,
    generated_image: Image.Image,
    blend_strength: float = FACE_BLEND_STRENGTH,
    similarity_threshold: float = FACE_SIMILARITY_THRESHOLD,
    device: str = "cuda",
) -> Tuple[Image.Image, str]:
    """
    High-level API: detect faces in both images, compare, and blend if needed.

    Detection thresholds are *asymmetric* on purpose:
      - Input image:  prob >= 0.97 (high bar — if we're not confident there's
                      a real face here, the whole preservation path is a no-op).
      - Output image: prob >= 0.985 (even higher — MTCNN hallucinates faces in
                      stylised pixel-art textures; we'd rather miss a true
                      output face than paint a phantom one).

    Args:
        original_image: The input image before pixel art conversion
        generated_image: The pixel art output
        blend_strength: Max blend strength (0-1)
        similarity_threshold: Cosine similarity threshold for triggering blend
        device: Compute device

    Returns:
        Tuple of (output image, human-readable report string)
    """
    # Asymmetric thresholds — see docstring.
    INPUT_MIN_PROB = 0.97
    OUTPUT_MIN_PROB = 0.985

    # Detect faces in the input first. If there are none we have *nothing
    # to preserve*; bail before we ever ask MTCNN about the stylised output.
    orig_faces = detect_faces(
        original_image, device=device, min_prob=INPUT_MIN_PROB
    )
    if not orig_faces:
        return generated_image, (
            "No high-confidence faces in input — skipping face preservation."
        )

    gen_faces = detect_faces(
        generated_image, device=device, min_prob=OUTPUT_MIN_PROB
    )

    if not gen_faces:
        # The pixel-art output legitimately may not contain a MTCNN-detectable
        # face even when the input did. We do NOT try to compensate by blending
        # the input face onto an arbitrary spot — that was the old failure
        # mode where faces materialized inside non-face regions.
        return generated_image, (
            f"Found {len(orig_faces)} face(s) in input but none in output "
            f"at high confidence — leaving generated image untouched."
        )

    # Blend where identity drifted
    result, reports = blend_face_identity(
        original=original_image,
        generated=generated_image,
        original_faces=orig_faces,
        generated_faces=gen_faces,
        blend_strength=blend_strength,
        similarity_threshold=similarity_threshold,
    )

    # Build report
    lines = [f"Faces detected: {len(orig_faces)} input, {len(gen_faces)} output"]
    for i, r in enumerate(reports):
        sim = r.get("similarity")
        if sim is None:
            sim_str = "n/a"
        else:
            sim_str = f"{sim * 100:.1f}%"

        status = "BLENDED" if r["blended"] else "SKIPPED"
        line = f"  Face {i+1}: similarity={sim_str} [{status}]"
        if r["blended"]:
            line += f" (strength={r['effective_strength']:.2f})"
        reason = r.get("reason")
        if reason:
            line += f" — {reason}"
        lines.append(line)

    return result, "\n".join(lines)
