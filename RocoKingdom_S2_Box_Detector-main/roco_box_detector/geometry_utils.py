"""Coordinate mapping, Sub-ROI calculation, boundary clipping."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SubRoiConfig:
    x_ratio: float
    y_ratio: float
    w_ratio: float
    h_ratio: float


def compute_sub_roi(
    anchor_box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    sub_roi_config: SubRoiConfig,
) -> Tuple[int, int, int, int]:
    """
    Compute the middle-layer side Sub-ROI from the anchor box.

    anchor_box: (x, y, w, h) relative to the ROI
    image_shape: (height, width) of the ROI image
    sub_roi_config: ratio values for computing sub-region

    Returns (x, y, w, h) clamped to image bounds.
    """
    anchor_x, anchor_y, anchor_w, anchor_h = anchor_box
    img_h, img_w = image_shape[:2]

    sub_x = int(anchor_x + anchor_w * sub_roi_config.x_ratio)
    sub_y = int(anchor_y + anchor_h * sub_roi_config.y_ratio)
    sub_w = max(1, int(anchor_w * sub_roi_config.w_ratio))
    sub_h = max(1, int(anchor_h * sub_roi_config.h_ratio))

    return clamp_box((sub_x, sub_y, sub_w, sub_h), img_w, img_h)


def clamp_box(
    box: Tuple[int, int, int, int],
    max_w: int,
    max_h: int,
) -> Tuple[int, int, int, int]:
    """Clamp a bounding box (x, y, w, h) to fit within (max_w, max_h)."""
    x, y, w, h = box
    x = max(0, min(x, max_w - 1))
    y = max(0, min(y, max_h - 1))
    w = max(1, min(w, max_w - x))
    h = max(1, min(h, max_h - y))
    return (x, y, w, h)


def scale_box(box: Tuple[int, int, int, int], factor: float) -> Tuple[int, int, int, int]:
    """Scale a box by factor (e.g. map from normalized ROI back to original ROI)."""
    x, y, w, h = box
    return (
        int(round(x * factor)),
        int(round(y * factor)),
        max(1, int(round(w * factor))),
        max(1, int(round(h * factor))),
    )
