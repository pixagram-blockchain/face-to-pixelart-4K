"""
Face Attribute Analyzer for Pixagram
=====================================

Pure OpenCV implementation. No MediaPipe, no InsightFace.
Detects facial attributes and returns a prompt string.

All components: OpenCV (BSD), NumPy (BSD).

Copyright (c) 2024 Pixagram SA
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass


# ============================================================
# Data
# ============================================================

@dataclass
class FaceAttributes:
    """Detected facial attributes for prompt enrichment."""
    age_group: str = ""
    gender_hint: str = ""
    hair_length: str = ""
    hair_color: str = ""
    eye_color: str = ""
    skin_tone: str = ""
    has_glasses: bool = False
    face_shape: str = ""
    expression: str = ""
    confidence: float = 0.0

    def to_prompt_tokens(self) -> List[str]:
        tokens = []
        if self.gender_hint and self.age_group:
            tokens.append(f"{self.age_group} {self.gender_hint}")
        elif self.gender_hint:
            tokens.append(self.gender_hint)
        elif self.age_group:
            tokens.append(f"{self.age_group} person")

        if self.hair_color and self.hair_length:
            if self.hair_length == "bald":
                tokens.append("bald")
            else:
                tokens.append(f"{self.hair_color} {self.hair_length} hair")
        elif self.hair_color:
            tokens.append(f"{self.hair_color} hair")
        elif self.hair_length:
            tokens.append(f"{self.hair_length} hair" if self.hair_length != "bald" else "bald")

        if self.skin_tone:
            tokens.append(f"{self.skin_tone} skin")
        if self.eye_color:
            tokens.append(f"{self.eye_color} eyes")
        if self.has_glasses:
            tokens.append("wearing glasses")
        if self.face_shape:
            tokens.append(f"{self.face_shape} face")
        if self.expression:
            tokens.append(f"{self.expression} expression")
        return tokens

    def to_prompt_string(self) -> str:
        tokens = self.to_prompt_tokens()
        return ", ".join(tokens) if tokens else ""


# ============================================================
# OpenCV cascade paths
# ============================================================

_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"
_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"
_EYE_GLASSES = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
_SMILE = cv2.data.haarcascades + "haarcascade_smile.xml"


# ============================================================
# Color helpers
# ============================================================

def _dominant_color(pixels: np.ndarray, k: int = 3) -> Tuple[int, int, int]:
    if len(pixels) < 10:
        return tuple(np.median(pixels, axis=0).astype(int))
    pixels_f32 = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = min(k, len(pixels))
    _, labels, centers = cv2.kmeans(
        pixels_f32, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    dominant_idx = np.argmax(np.bincount(labels.flatten()))
    return tuple(centers[dominant_idx].astype(int))


def _classify_hair_color(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    bri = (r + g + b) / 3.0
    if bri > 210:
        return "white"
    if bri > 170:
        return "blonde" if (r > g > b and r - b > 30) else "gray"
    if bri > 130:
        if r > g and r > b and r - b > 25:
            return "blonde"
        if r > 100 and g < 90 and b < 80:
            return "red"
        return "brown"
    if bri > 70:
        if r > g and r > b and r - g > 20:
            return "red" if r > 120 else "brown"
        return "brown"
    return "dark"


def _classify_skin_tone(rgb: Tuple[int, int, int]) -> str:
    bri = sum(rgb) / 3.0
    if bri > 190:
        return "light"
    if bri > 150:
        return "fair"
    if bri > 120:
        return "medium"
    if bri > 85:
        return "tan"
    return "dark"


# ============================================================
# Face Analyzer — pure OpenCV
# ============================================================

class FaceAnalyzer:
    """
    Face attribute analyzer using only OpenCV (BSD).

    Detects face with Haar cascades, then extracts attributes
    from pixel analysis within the face region.
    """

    def __init__(self, min_confidence: float = 0.5):
        print("\nInitializing FaceAnalyzer (OpenCV only)...")
        self._face_cascade = cv2.CascadeClassifier(_FRONTAL)
        self._profile_cascade = cv2.CascadeClassifier(_PROFILE)
        self._eye_cascade = cv2.CascadeClassifier(_EYE)
        self._eye_glasses_cascade = cv2.CascadeClassifier(_EYE_GLASSES)
        self._smile_cascade = cv2.CascadeClassifier(_SMILE)
        print("  [OK] FaceAnalyzer ready (OpenCV Haar, BSD)\n")

    def _detect_faces(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        faces = list(self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        ))
        if not faces:
            faces = list(self._profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            ))
        return [tuple(f) for f in faces]

    def has_face(self, image: Union[Image.Image, np.ndarray]) -> bool:
        gray = self._to_gray(image)
        return len(self._detect_faces(gray)) > 0

    def analyze(self, image: Image.Image) -> Optional[FaceAttributes]:
        """Analyze face and return attributes. None if no face."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = image_rgb.shape[:2]

        faces = self._detect_faces(gray)
        if not faces:
            return None

        # Largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        attrs = FaceAttributes(confidence=0.7)

        face_rgb = image_rgb[y:y+fh, x:x+fw]
        face_gray = gray[y:y+fh, x:x+fw]

        if face_rgb.size == 0:
            return None

        # --- Skin tone (center of face) ---
        cy, cx = fh // 2, fw // 2
        r = min(fh, fw) // 6
        skin_patch = face_rgb[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
        if skin_patch.size > 0:
            skin_rgb = _dominant_color(skin_patch.reshape(-1, 3), k=2)
            attrs.skin_tone = _classify_skin_tone(skin_rgb)

        # --- Hair color (region above face) ---
        hair_y_end = max(0, y - 5)
        hair_y_start = max(0, y - int(fh * 0.5))
        if hair_y_end > hair_y_start:
            hair_region = image_rgb[hair_y_start:hair_y_end, x:x+fw]
            if hair_region.size > 0:
                hair_rgb = _dominant_color(hair_region.reshape(-1, 3), k=3)
                attrs.hair_color = _classify_hair_color(hair_rgb)

        # --- Hair length (check below jawline) ---
        attrs.hair_length = self._estimate_hair_length(image_rgb, x, y, fw, fh, h, w)

        # --- Glasses (compare eye detectors) ---
        attrs.has_glasses = self._detect_glasses(face_gray)

        # --- Expression (smile detector) ---
        attrs.expression = self._detect_expression(face_gray, fh)

        # --- Face shape (aspect ratio) ---
        ratio = fh / max(fw, 1)
        if ratio > 1.35:
            attrs.face_shape = "long"
        elif ratio > 1.15:
            attrs.face_shape = "oval"
        elif ratio > 0.95:
            attrs.face_shape = "square"
        else:
            attrs.face_shape = "round"

        # --- Age (skin texture) ---
        attrs.age_group = self._estimate_age(face_gray, attrs)

        # --- Gender hint (heuristics) ---
        attrs.gender_hint = self._guess_gender(face_gray, fw, fh, attrs)

        return attrs

    def _estimate_hair_length(self, image_rgb, x, y, fw, fh, h, w) -> str:
        """Check how far hair-like pixels extend below the jawline."""
        jaw_y = y + fh
        check_end = min(h, jaw_y + int(fh * 0.8))
        if check_end <= jaw_y:
            return "short"

        cx = x + fw // 2
        x1 = max(0, cx - fw // 3)
        x2 = min(w, cx + fw // 3)

        below = image_rgb[jaw_y:check_end, x1:x2]
        if below.size == 0:
            return "short"

        rows = below.shape[0]
        hair_rows = 0
        for ri in range(rows):
            row = below[ri]
            std = np.std(row, axis=0).mean()
            bri = np.mean(row)
            if 10 < std < 80 and 20 < bri < 220:
                hair_rows += 1
            else:
                break

        extent = hair_rows / max(rows, 1)

        # Check if possibly bald (bright/uniform above forehead)
        if extent < 0.05:
            above_y_start = max(0, y - int(fh * 0.3))
            above = image_rgb[above_y_start:max(0, y - 5), x:x+fw]
            if above.size > 0 and np.std(above) < 15 and np.mean(above) > 180:
                return "bald"
            return "short"
        if extent < 0.3:
            return "medium"
        return "long"

    def _detect_glasses(self, face_gray: np.ndarray) -> bool:
        """
        Detect glasses by comparing eye-with-glasses vs plain eye cascade.
        If glasses cascade finds eyes but plain doesn't, likely glasses.
        Also check for strong horizontal edges in the eye-bridge region.
        """
        fh, fw = face_gray.shape[:2]

        # Only search in the upper half of face (eye region)
        eye_region = face_gray[0:fh//2, :]

        plain_eyes = self._eye_cascade.detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        glasses_eyes = self._eye_glasses_cascade.detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        # Glasses often block plain eye detection but not the glasses cascade
        if len(glasses_eyes) >= 2 and len(plain_eyes) == 0:
            return True

        # Also check edge density in the nose-bridge area
        bridge_y = fh // 4
        bridge_h = fh // 8
        bridge_x = fw // 4
        bridge_w = fw // 2
        bridge = face_gray[bridge_y:bridge_y+bridge_h, bridge_x:bridge_x+bridge_w]
        if bridge.size > 0:
            edges = cv2.Canny(bridge, 50, 150)
            edge_density = np.sum(edges > 0) / max(edges.size, 1)
            if edge_density > 0.2:
                return True

        return False

    def _detect_expression(self, face_gray: np.ndarray, fh: int) -> str:
        """Detect smile using Haar cascade on lower face."""
        # Only search in the lower half (mouth region)
        lower_face = face_gray[fh//2:, :]

        smiles = self._smile_cascade.detectMultiScale(
            lower_face, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
        )
        if len(smiles) > 0:
            return "smiling"
        return "neutral"

    def _estimate_age(self, face_gray: np.ndarray, attrs: FaceAttributes) -> str:
        """Estimate age from skin texture (wrinkle proxy)."""
        fh, fw = face_gray.shape[:2]

        # Sample forehead region
        forehead = face_gray[0:fh//4, fw//4:3*fw//4]
        if forehead.size == 0:
            return "adult"

        laplacian_var = cv2.Laplacian(forehead, cv2.CV_64F).var()
        edge_density = np.mean(np.abs(cv2.Sobel(forehead, cv2.CV_64F, 0, 1, ksize=3)))

        gray_hair = attrs.hair_color in ("gray", "white")

        if gray_hair and edge_density > 15:
            return "elderly"
        if gray_hair or (edge_density > 12 and laplacian_var > 100):
            return "mature"
        if laplacian_var < 30 and edge_density < 6:
            return "young"
        return "adult"

    def _guess_gender(
        self, face_gray: np.ndarray, fw: int, fh: int, attrs: FaceAttributes
    ) -> str:
        """Rough gender guess from face proportions + detected features."""
        score = 0

        # Wider face tends masculine
        if fw / max(fh, 1) > 0.85:
            score += 1

        # Jaw sharpness (edge density in lower third)
        jaw_region = face_gray[2*fh//3:, :]
        if jaw_region.size > 0:
            jaw_edges = cv2.Canny(jaw_region, 50, 150)
            jaw_edge_density = np.sum(jaw_edges > 0) / max(jaw_edges.size, 1)
            if jaw_edge_density > 0.15:
                score += 1

        # Eyebrow thickness (edge density in upper region)
        brow_region = face_gray[fh//6:fh//4, fw//6:5*fw//6]
        if brow_region.size > 0:
            brow_edges = cv2.Canny(brow_region, 30, 100)
            brow_density = np.sum(brow_edges > 0) / max(brow_edges.size, 1)
            if brow_density > 0.2:
                score += 1

        # Hair length as signal
        if attrs.hair_length in ("long", "medium"):
            score -= 1

        return "man" if score >= 2 else "woman"

    def get_prompt_enrichment(self, image: Image.Image) -> str:
        attrs = self.analyze(image)
        return attrs.to_prompt_string() if attrs else ""

    @staticmethod
    def _to_gray(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


print("[OK] Face attribute analyzer loaded (pure OpenCV)")