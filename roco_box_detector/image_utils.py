"""Image utility functions: Chinese-path reading, preprocessing, resizing."""

import cv2
import numpy as np
import os
import sys


def _get_base_dir() -> str:
    """Return the base directory for resolving relative paths.
    When frozen (PyInstaller), prefer the exe's directory so users can
    edit config / add templates without repacking. Falls back to _MEIPASS."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resolve_path(relative_path: str) -> str:
    """Resolve a path that may be relative to the project root.
    Checks: ① as-is  ② exe dir  ③ _MEIPASS (PyInstaller bundled)."""
    if os.path.exists(relative_path):
        return relative_path
    base = _get_base_dir()
    resolved = os.path.join(base, relative_path)
    if os.path.exists(resolved):
        return resolved
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', '')
        if meipass:
            alt = os.path.join(meipass, relative_path)
            if os.path.exists(alt):
                return alt
    return resolved


def imread_chinese(path: str) -> np.ndarray | None:
    """Read an image from a path that may contain Chinese characters."""
    path = resolve_path(path)
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def preprocess_image(
    image: np.ndarray,
    use_grayscale: bool = True,
    use_canny: bool = False,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """Unified preprocessing for both ROI frames and templates."""
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if use_canny:
        return cv2.Canny(gray, canny_low, canny_high)
    if use_grayscale:
        return gray
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def resize_by_width(image: np.ndarray, target_width: int) -> np.ndarray | None:
    """Resize image to target width while maintaining aspect ratio."""
    if image is None or target_width <= 0:
        return image
    h, w = image.shape[:2]
    if w == target_width:
        return image
    ratio = target_width / w
    target_h = max(1, int(h * ratio))
    return cv2.resize(image, (target_width, target_h), interpolation=cv2.INTER_LINEAR)


def make_gaussian_mask(w: int, h: int, sigma: float | None = None) -> np.ndarray:
    """Generate a 2D Gaussian weight mask centered on (w/2, h/2).

    sigma defaults to min(w, h) / 4, so the centre ~1/4 area has weight >= 0.6.
    Returns uint8 array scaled to 0-255, same size as the template.
    """
    if sigma is None:
        sigma = min(w, h) / 4.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    dist_sq = ((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma * sigma)
    g = np.exp(-dist_sq)
    g = (g / g.max() * 255).astype(np.uint8)
    return g


def safe_resize_template(template: np.ndarray, scale: float) -> np.ndarray | None:
    """Resize template by scale factor. Returns None if the result would be larger than a sensible limit."""
    if template is None:
        return None
    h, w = template.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def safe_resize_mask(mask: np.ndarray, scale: float) -> np.ndarray | None:
    """Resize a Gaussian mask by scale factor, preserving smooth gradients."""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
