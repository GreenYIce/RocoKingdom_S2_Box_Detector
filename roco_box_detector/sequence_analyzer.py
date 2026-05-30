"""Multi-frame sequence analysis: stability filtering, sharpness, voting."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import time

import cv2
import numpy as np

from image_utils import (preprocess_image, resize_by_width,
                         safe_resize_template, safe_resize_mask)
from template_cache import TemplateCache, TemplateGroup, TemplateItem


@dataclass
class SequenceFrame:
    index: int
    timestamp: float
    roi_image: np.ndarray
    sub_roi_image: np.ndarray
    anchor_box: Tuple[int, int, int, int]
    sub_roi_box: Tuple[int, int, int, int]
    sharpness: float = 0.0
    avg_similarity: float = 0.0
    sub_roi_image_2: Optional[np.ndarray] = None
    sub_roi_box_2: Optional[Tuple[int, int, int, int]] = None

    def __hash__(self):
        return hash((self.index, self.timestamp))


@dataclass
class FrameMatchResult:
    frame_index: int
    label: Optional[str]
    matched: bool
    score: float
    template_path: Optional[str]
    scale: Optional[float]
    box_in_sub_roi: Optional[Tuple[int, int, int, int]]
    box_in_roi: Optional[Tuple[int, int, int, int]]
    roi_name: str = "roi1"


@dataclass
class SequenceDetectionResult:
    matched: bool
    label: Optional[str]
    final_score: float
    vote_count: int
    total_selected_frames: int
    selected_frame_indices: List[int]
    frame_results: List[FrameMatchResult] = field(default_factory=list)
    best_frame_index: Optional[int] = None
    # Dual-ROI support
    label_1: Optional[str] = None
    label_2: Optional[str] = None
    final_score_1: float = 0.0
    final_score_2: float = 0.0
    vote_count_1: int = 0
    vote_count_2: int = 0


# ── sharpness ────────────────────────────────────────────────────────

def calculate_sharpness(image: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    if image is None or image.size == 0:
        return 0.0
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── similarity ───────────────────────────────────────────────────────

def _resize_for_similarity(image: np.ndarray, target_w: int) -> np.ndarray:
    """Resize to a small fixed width for fast similarity comparison."""
    h, w = image.shape[:2]
    if w <= 0:
        return image
    ratio = target_w / w
    target_h = max(1, int(h * ratio))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray.astype(np.float32) / 255.0


def compute_pairwise_similarity(frame_a: SequenceFrame, frame_b: SequenceFrame, resize_w: int = 96) -> float:
    """Compute similarity [0, 1] between two frames' sub_roi regions."""
    ga = _resize_for_similarity(frame_a.sub_roi_image, resize_w)
    gb = _resize_for_similarity(frame_b.sub_roi_image, resize_w)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    fa = _to_gray_float(ga)
    fb = _to_gray_float(gb)
    mse = float(np.mean((fa - fb) ** 2))
    return 1.0 - min(mse, 1.0)


def compute_avg_similarities(frames: List[SequenceFrame], resize_w: int = 96) -> None:
    """Compute each frame's average similarity to all others. Mutates frames in-place."""
    n = len(frames)
    if n < 2:
        for f in frames:
            f.avg_similarity = 1.0
        return
    for i, fi in enumerate(frames):
        total = 0.0
        for j, fj in enumerate(frames):
            if i == j:
                continue
            total += compute_pairwise_similarity(fi, fj, resize_w)
        fi.avg_similarity = total / (n - 1)


# ── selection ────────────────────────────────────────────────────────

def select_stable_frames(
    frames: List[SequenceFrame],
    selected_count: int,
    resize_w: int = 96,
    use_similarity: bool = True,
    use_sharpness: bool = True,
    similarity_threshold: float = 0.0,
) -> List[SequenceFrame]:
    """
    Pick the best frames from a sequence.

    Strategy:
      1. Compute each frame's average similarity to all others.
      2. Optionally filter frames below similarity_threshold.
      3. Sort by similarity (desc), then by sharpness (desc) as tiebreaker.
      4. Return top selected_count frames.
    """
    if len(frames) <= selected_count:
        return list(frames)

    if use_similarity:
        compute_avg_similarities(frames, resize_w)

        # Filter low-similarity outliers
        if similarity_threshold > 0:
            candidates = [f for f in frames if f.avg_similarity >= similarity_threshold]
            if len(candidates) < selected_count:
                candidates = sorted(frames, key=lambda f: f.avg_similarity, reverse=True)
                candidates = candidates[:selected_count]
        else:
            candidates = list(frames)

        scored: List[Tuple[SequenceFrame, float]] = []
        for f in candidates:
            sim = f.avg_similarity
            shp = f.sharpness if use_sharpness else 0.0
            score = sim + min(shp / 10000.0, 0.01)
            scored.append((f, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:selected_count]]

    # Similarity disabled: pick by sharpness, fallback to time order
    if use_sharpness:
        sorted_frames = sorted(frames, key=lambda f: f.sharpness, reverse=True)
        return sorted_frames[:selected_count]

    # Neither: take the last N frames (most recent first)
    return list(reversed(frames[-selected_count:]))


# ── per-frame pattern matching ───────────────────────────────────────

def match_single_frame_to_patterns(
    sub_roi_image: np.ndarray,
    sub_roi_box: Tuple[int, int, int, int],
    frame_index: int,
    pattern_groups: Dict[str, TemplateGroup],
    roi_name: str = "roi1",
    norm_width: int = 0,
) -> FrameMatchResult:
    """
    Match a single sub_roi frame against all pattern groups.
    If norm_width > 0, the sub_roi is normalized to that width first,
    and box coordinates are scaled back to original sub_roi space.
    """
    best_result = FrameMatchResult(
        frame_index=frame_index,
        label=None,
        matched=False,
        score=0.0,
        template_path=None,
        scale=None,
        box_in_sub_roi=None,
        box_in_roi=None,
        roi_name=roi_name,
    )
    if sub_roi_image is None or sub_roi_image.size == 0:
        return best_result
    sx, sy, sw, sh = sub_roi_box

    # Normalize for resolution-independent matching
    pat_scale = 1.0
    if norm_width > 0 and sub_roi_image.shape[1] > 0:
        pat_scale = norm_width / sub_roi_image.shape[1]
        sub_roi_image = resize_by_width(sub_roi_image, norm_width)

    for pname, pgroup in pattern_groups.items():
        if not pgroup.items:
            continue

        sub_processed = preprocess_image(
            sub_roi_image,
            use_grayscale=pgroup.use_grayscale,
        )
        if sub_processed is None or sub_processed.size == 0:
            continue

        best_score = -1.0
        best_box = None
        best_tmpl_path = None
        best_scale = 1.0
        img_h, img_w = sub_processed.shape[:2]

        for tmpl in pgroup.items:
            tmpl_img = _select_template_variant(tmpl, pgroup.use_grayscale)
            if tmpl_img is None:
                continue
            for scale in np.linspace(pgroup.scale_min, pgroup.scale_max, pgroup.scale_steps):
                scaled = safe_resize_template(tmpl_img, scale)
                if scaled is None:
                    continue
                th, tw = scaled.shape[:2]
                if th > img_h or tw > img_w:
                    continue
                try:
                    if tmpl.mask is not None:
                        scaled_mask = safe_resize_mask(tmpl.mask, scale)
                        result = cv2.matchTemplate(
                            sub_processed, scaled, cv2.TM_CCOEFF_NORMED,
                            mask=scaled_mask)
                    else:
                        result = cv2.matchTemplate(
                            sub_processed, scaled, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                except cv2.error:
                    continue
                if max_val > best_score:
                    best_score = max_val
                    best_box = (max_loc[0], max_loc[1], tw, th)
                    best_tmpl_path = tmpl.path
                    best_scale = scale

        if best_box is not None:
            matched = best_score >= pgroup.threshold
            lx, ly, lw, lh = best_box
            if pat_scale != 1.0:
                inv = 1.0 / pat_scale
                lx, ly, lw, lh = (int(lx * inv), int(ly * inv),
                                   int(lw * inv), int(lh * inv))
            result = FrameMatchResult(
                frame_index=frame_index,
                label=pname,
                matched=matched,
                score=float(best_score),
                template_path=best_tmpl_path,
                scale=float(best_scale),
                box_in_sub_roi=(lx, ly, lw, lh),
                box_in_roi=(sx + lx, sy + ly, lw, lh),
                roi_name=roi_name,
            )
            if result.score > best_result.score:
                best_result = result

    return best_result


def _select_template_variant(
    tmpl: TemplateItem, use_grayscale: bool,
) -> Optional[np.ndarray]:
    if use_grayscale:
        return tmpl.image_gray
    return tmpl.image_color


# ── voting ───────────────────────────────────────────────────────────

def vote_frame_results(
    frame_results: List[FrameMatchResult],
    min_vote_count: int = 3,
    min_average_score: float = 0.72,
) -> SequenceDetectionResult:
    """
    Aggregate per-frame results via voting.

    Returns the best SequenceDetectionResult. If no label meets the thresholds,
    returns matched=False.
    """
    matched_results = [r for r in frame_results if r.matched and r.label]

    if not matched_results:
        return SequenceDetectionResult(
            matched=False,
            label=None,
            final_score=0.0,
            vote_count=0,
            total_selected_frames=len(frame_results),
            selected_frame_indices=[],
            frame_results=list(frame_results),
            best_frame_index=None,
        )

    # Group by label
    by_label: Dict[str, List[FrameMatchResult]] = {}
    for r in matched_results:
        by_label.setdefault(r.label, []).append(r)

    best: Optional[SequenceDetectionResult] = None

    for label, results in by_label.items():
        vote_count = len(results)
        avg_score = sum(r.score for r in results) / vote_count

        if vote_count >= min_vote_count and avg_score >= min_average_score:
            if best is None or vote_count > best.vote_count or (
                vote_count == best.vote_count and avg_score > best.final_score
            ):
                best_frame = max(results, key=lambda r: r.score)
                best = SequenceDetectionResult(
                    matched=True,
                    label=label,
                    final_score=avg_score,
                    vote_count=vote_count,
                    total_selected_frames=len(frame_results),
                    selected_frame_indices=[r.frame_index for r in results],
                    frame_results=list(frame_results),
                    best_frame_index=best_frame.frame_index,
                )

    if best is None:
        # Find the closest near-miss for debug
        best_label = max(by_label.items(), key=lambda kv: (len(kv[1]), sum(r.score for r in kv[1]) / len(kv[1])))
        results_list = best_label[1]
        best = SequenceDetectionResult(
            matched=False,
            label=best_label[0],
            final_score=sum(r.score for r in results_list) / len(results_list),
            vote_count=len(results_list),
            total_selected_frames=len(frame_results),
            selected_frame_indices=[r.frame_index for r in results_list],
            frame_results=list(frame_results),
            best_frame_index=max(results_list, key=lambda r: r.score).frame_index,
        )

    return best
