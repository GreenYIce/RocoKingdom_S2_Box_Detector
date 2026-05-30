"""Core cascade detection: anchor -> Sub-ROI -> pattern, with optional multi-frame sampling."""

import os
import threading
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import mss

from image_utils import (preprocess_image, resize_by_width,
                         safe_resize_template, safe_resize_mask, resolve_path)
from geometry_utils import SubRoiConfig, compute_sub_roi, scale_box
from template_cache import TemplateCache, TemplateItem
from debug_utils import DebugDrawer, ThrottledLogger
from sequence_analyzer import (
    SequenceFrame,
    FrameMatchResult,
    SequenceDetectionResult,
    calculate_sharpness,
    match_single_frame_to_patterns,
    vote_frame_results,
)


@dataclass
class MatchResult:
    stage: str
    label: str
    matched: bool
    score: float
    box: Optional[Tuple[int, int, int, int]]
    template_path: Optional[str]
    scale: Optional[float]


@dataclass
class CascadeDetectionResult:
    matched: bool
    label: Optional[str]
    anchor_result: Optional[MatchResult]
    pattern_result: Optional[MatchResult]
    sub_roi_box: Optional[Tuple[int, int, int, int]]
    anchor_box: Optional[Tuple[int, int, int, int]] = None  # original-ROI coords
    debug_frame: Optional[np.ndarray] = None
    sub_roi_box_2: Optional[Tuple[int, int, int, int]] = None
    status: str = ""
    mode: str = "SEQ"
    sequence_result: Optional[SequenceDetectionResult] = None
    match_votes: str = ""  # only meaningful in SEQ mode
    sub_roi1_image: Optional[np.ndarray] = None  # for screenshot preview
    sub_roi2_image: Optional[np.ndarray] = None


# ── Template matching ────────────────────────────────────────────────

def match_templates_multiscale(
    image: np.ndarray,
    template_items: List[TemplateItem],
    threshold: float,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    use_grayscale: bool = True,
    label: str = "unknown",
    coarse_threshold: Optional[float] = None,
    early_exit_score: float = 0.0,
) -> MatchResult:
    """
    Multi-template, multi-scale template matching.

    If coarse_threshold is set (>0), a single middle-scale pass runs first.
    When no template reaches coarse_threshold at that scale, the function
    returns early without the full multi-scale search — ~90% faster on empty frames.

    Important:
    - Always records the real global best score.
    - matched is decided only at the end by comparing best_score >= threshold.
    - Even when not matched, best_box / best_score can be used for debug.
    """
    best_score = -1.0
    best_box = None
    best_tmpl_path = None
    best_scale = 1.0

    if image is None or image.size == 0:
        return MatchResult(stage=label, label=label, matched=False,
                           score=0.0, box=None, template_path=None, scale=None)

    if not template_items:
        return MatchResult(stage=label, label=label, matched=False,
                           score=0.0, box=None, template_path=None, scale=None)

    img_h, img_w = image.shape[:2]

    # ── Phase 1: coarse single-scale check ──
    if coarse_threshold is not None and coarse_threshold > 0 and scale_steps > 1:
        mid_scale = (scale_min + scale_max) / 2.0
        best_coarse = -1.0
        for tmpl in template_items:
            tmpl_img = _select_template_variant(tmpl, use_grayscale)
            if tmpl_img is None or tmpl_img.size == 0:
                continue
            scaled = safe_resize_template(tmpl_img, mid_scale)
            if scaled is None or scaled.size == 0:
                continue
            th, tw = scaled.shape[:2]
            if th > img_h or tw > img_w:
                continue
            try:
                result = cv2.matchTemplate(image, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_coarse:
                    best_coarse = max_val
            except cv2.error:
                continue
        if best_coarse < coarse_threshold:
            return MatchResult(stage=label, label=label, matched=False,
                               score=float(best_coarse), box=None,
                               template_path=None, scale=None)

    # ── Phase 2: full multi-scale search (prefer cached variants) ──

    cached = any(tmpl.scaled_variants for tmpl in template_items)

    if cached:
        for tmpl in template_items:
            for sv in tmpl.scaled_variants:
                if sv.width > img_w or sv.height > img_h:
                    continue
                try:
                    if sv.scaled_mask is not None:
                        result = cv2.matchTemplate(
                            image, sv.scaled_gray, cv2.TM_CCOEFF_NORMED,
                            mask=sv.scaled_mask)
                    else:
                        result = cv2.matchTemplate(
                            image, sv.scaled_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                except cv2.error:
                    continue
                if max_val > best_score:
                    best_score = max_val
                    best_box = (max_loc[0], max_loc[1], sv.width, sv.height)
                    best_tmpl_path = tmpl.path
                    best_scale = sv.scale
                    if early_exit_score > 0 and best_score >= early_exit_score:
                        return MatchResult(
                            stage=label, label=label, matched=True,
                            score=float(best_score), box=best_box,
                            template_path=best_tmpl_path,
                            scale=float(best_scale))
    else:
        # Fallback: runtime resize (should not normally be reached)
        for tmpl in template_items:
            tmpl_img = _select_template_variant(tmpl, use_grayscale)
            if tmpl_img is None or tmpl_img.size == 0:
                continue
            for scale in np.linspace(scale_min, scale_max, scale_steps):
                scaled = safe_resize_template(tmpl_img, scale)
                if scaled is None or scaled.size == 0:
                    continue
                th, tw = scaled.shape[:2]
                if th > img_h or tw > img_w:
                    continue
                try:
                    if tmpl.mask is not None:
                        scaled_mask = safe_resize_mask(tmpl.mask, scale)
                        result = cv2.matchTemplate(
                            image, scaled, cv2.TM_CCOEFF_NORMED, mask=scaled_mask)
                    else:
                        result = cv2.matchTemplate(
                            image, scaled, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                except cv2.error:
                    continue
                if max_val > best_score:
                    best_score = max_val
                    best_box = (max_loc[0], max_loc[1], tw, th)
                    best_tmpl_path = tmpl.path
                    best_scale = scale
                    if early_exit_score > 0 and best_score >= early_exit_score:
                        return MatchResult(
                            stage=label, label=label, matched=True,
                            score=float(best_score), box=best_box,
                            template_path=best_tmpl_path,
                            scale=float(best_scale))

    matched = best_score >= threshold and best_box is not None

    return MatchResult(
        stage=label, label=label, matched=matched,
        score=float(best_score if best_score >= 0 else 0.0),
        box=best_box, template_path=best_tmpl_path,
        scale=float(best_scale) if best_box is not None else None,
    )


def _select_template_variant(
    tmpl: TemplateItem,
    use_grayscale: bool,
) -> Optional[np.ndarray]:
    if use_grayscale:
        return tmpl.image_gray
    return tmpl.image_color


# ── Detector ─────────────────────────────────────────────────────────

class CascadeDetector(threading.Thread):
    """Background detector thread."""

    def __init__(
        self,
        config: dict,
        cache: TemplateCache,
        debug_drawer: DebugDrawer,
        on_result: Callable[[CascadeDetectionResult], None],
    ):
        super().__init__(daemon=True)

        self.config = config
        self.cache = cache
        self.debug_drawer = debug_drawer
        self.on_result = on_result

        self.roi: Optional[Dict[str, int]] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._fps = 10
        self._norm_width = 800
        self._pat_norm_width = 0
        self._anchor_skip = 0
        self._log_every = 10
        self._show_preview = False
        self._debug_overlay_enabled = False  # transparent box overlay
        self._screenshot_only = False
        self._latest_sub_roi1: Optional[np.ndarray] = None
        self._latest_sub_roi2: Optional[np.ndarray] = None
        self._preview_window_open = False
        self._preview_fps_limit = 15   # max preview refresh rate
        self._last_preview_time = 0.0
        # Per-frame debug state (updated by detection methods, drawn by unified preview)
        self._dbg_roi: Optional[np.ndarray] = None
        self._dbg_anchor_box: Optional[Tuple[int, int, int, int]] = None
        self._dbg_sub_roi_box: Optional[Tuple[int, int, int, int]] = None
        self._dbg_sub_roi_box_2: Optional[Tuple[int, int, int, int]] = None
        self._dbg_pattern_box: Optional[Tuple[int, int, int, int]] = None
        self._dbg_pattern_box_2: Optional[Tuple[int, int, int, int]] = None
        self._dbg_pattern_best_box: Optional[Tuple[int, int, int, int]] = None
        self._dbg_pattern_best_box_2: Optional[Tuple[int, int, int, int]] = None
        self._dbg_anchor_score: float = 0.0
        self._dbg_pattern_score: float = 0.0
        self._dbg_pattern_score_2: float = 0.0
        self._dbg_pattern_label: Optional[str] = None
        self._dbg_pattern_label_2: Optional[str] = None
        self._dbg_status_text: str = ""
        self._last_cycle_ms: float = 0.0

        self._logger = ThrottledLogger(every_n=self._log_every)
        self._frame_idx = 0

        # Reusable mss instance, created in run()
        self._sct: Optional[mss.mss] = None

        # Sequence state
        self._sampling_state = "idle"  # "idle" | "sampling" | "cooldown"
        self._locked_anchor_box: Optional[Tuple[int, int, int, int]] = None
        self._locked_sub_roi: Optional[Tuple[int, int, int, int]] = None
        self._locked_sub_roi_2: Optional[Tuple[int, int, int, int]] = None
        self._sampling_frames: List[SequenceFrame] = []
        # Incremental pattern results for SEQ early vote.
        # Each sampled frame is matched exactly once, then reused by finalize.
        self._sampling_match_results_1: List[FrameMatchResult] = []
        self._sampling_match_results_2: List[FrameMatchResult] = []
        self._sampling_seq_idx = 0
        self._sampling_start_time = 0.0
        self._sampling_next_time = 0.0
        self._anchor_lost_count = 0
        self._cooldown_until = 0.0
        self._seq_triggered = False

        # Sequence config defaults
        self._sequence_enabled = True
        self._seq_sample_delay = 0.5
        self._seq_sample_interval = 0.1
        self._seq_max_frames = 5
        self._min_frames_for_vote = 3
        self._seq_selected_count = 5
        self._seq_min_votes = 2
        self._seq_min_avg = 0.72
        self._seq_sim_thresh = 0.88
        self._seq_resize_w = 96
        self._seq_use_sharpness = True
        self._seq_use_similarity = False
        self._seq_allow_lost = 2
        self._cooldown_seconds = 1.5
        self._require_roi2 = False
        self._roi2_enabled = False

        self._seq_output_dir = ""
        self._seq_selected_dir = ""

        self._load_all_config(config)

    # ── config ───────────────────────────────────────────────────────

    def _load_all_config(self, config: dict) -> None:
        """Shared config loader used by __init__ and update_config."""
        self.config = config

        rt = config.get("runtime", {})
        self._fps = max(1, int(rt.get("capture_fps", 10)))
        self._norm_width = int(rt.get("normalize_roi_width", 800))
        self._pat_norm_width = int(rt.get("pattern_normalize_width", 0))
        self._anchor_skip = max(0, int(rt.get("anchor_skip_frames", 0)))
        self._log_every = max(1, int(rt.get("log_every_n_frames", 10)))
        self._logger.every_n = self._log_every

        dbg = config.get("debug", {})
        self._show_preview = bool(dbg.get("show_preview_window", False))

        out_dir = resolve_path(dbg.get("debug_output_dir", "debug_output"))
        self._seq_output_dir = os.path.join(out_dir, "sequence_raw")
        self._seq_selected_dir = os.path.join(out_dir, "sequence_selected")

        seq = config.get("sequence", {})
        self._sequence_enabled = True  # always SEQ — ignore config
        self._seq_sample_delay = float(seq.get("sample_delay_seconds", 0.5))
        self._seq_sample_interval = float(
            seq.get("sample_interval_seconds")
            or (1.0 / max(1, seq.get("sample_fps", 10)))
        )
        self._seq_max_frames = max(1, int(seq.get("max_frames", 5)))
        self._min_frames_for_vote = max(1, int(seq.get("min_frames_for_vote", 3)))
        self._seq_selected_count = max(1, int(seq.get("selected_frame_count", 5)))
        self._seq_min_votes = max(1, int(seq.get("min_vote_count", 2)))
        self._seq_min_avg = float(seq.get("min_average_score", 0.72))
        self._seq_sim_thresh = float(seq.get("stable_similarity_threshold", 0.88))
        self._seq_resize_w = max(16, int(seq.get("frame_resize_width", 96)))
        self._seq_use_sharpness = bool(seq.get("use_sharpness_filter", True))
        self._seq_use_similarity = bool(seq.get("use_similarity_filter", False))
        self._seq_allow_lost = max(0, int(seq.get("allow_anchor_lost_frames", 2)))

        self._cooldown_seconds = float(rt.get("sequence_cooldown_seconds", 1.5))

        self._require_roi2 = bool(seq.get("require_roi2_for_result", False))
        sr2 = config.get("sub_roi_2", {})
        self._roi2_enabled = bool(sr2.get("enabled", False))

    def update_config(self, config: dict) -> None:
        self._load_all_config(config)

    # ── debug save switches ──────────────────────────────────────────

    def _debug_save_enabled(self) -> bool:
        """
        GUI 截屏总开关。

        只要 GUI 打开 save_debug_frames：
        - SEQ 会保存最终用于投票的 selected frames。
        """
        return bool(self.config.get("debug", {}).get("save_debug_frames", False))

    def _save_selected_enabled(self) -> bool:
        """SEQ selected frames save: follows the global debug screenshot toggle."""
        return self._debug_save_enabled()

    # ── public API ───────────────────────────────────────────────────

    def set_roi(self, roi: Dict[str, int]) -> None:
        self.roi = roi
        self.cache.rescale_anchor(roi["width"], self._norm_width)
        # Reset all sampling state to avoid stale state from previous ROI
        self._sampling_state = "idle"
        self._seq_triggered = False
        self._cooldown_until = 0
        self._sampling_frames.clear()
        self._sampling_match_results_1.clear()
        self._sampling_match_results_2.clear()
        self._locked_anchor_box = None
        self._locked_sub_roi = None
        self._locked_sub_roi_2 = None
        self._sampling_seq_idx = 0
        self._pause_event.set()

    def refresh_anchor_scale(self) -> None:
        """Re-pre-scale anchor after config reload (called by main.py)."""
        if self.roi is not None:
            self.cache.rescale_anchor(self.roi["width"], self._norm_width)

    def set_sequence_enabled(self, enabled: bool = True) -> None:
        self._sequence_enabled = True  # always SEQ

    def set_preview_enabled(self, enabled: bool) -> None:
        self._show_preview = enabled

    def set_screenshot_only(self, enabled: bool) -> None:
        """In screenshot-only mode, stop after first sub-ROI capture. No pattern matching."""
        self._screenshot_only = enabled

    def stop(self) -> None:
        self._show_preview = False
        self._stop_event.set()
        self._pause_event.set()

    # ── main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        self._sct = mss.mss()
        interval = 1.0 / max(1, self._fps)
        _mss_frame = 0

        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()

                if self._stop_event.is_set():
                    break

                if self.roi is None:
                    time.sleep(0.1)
                    continue

                # Periodic mss recreation to prevent GDI handle leak
                _mss_frame += 1
                if _mss_frame % 500 == 0:
                    self._sct.close()
                    self._sct = mss.mss()

                t0 = time.time()

                result = self._detect_one_frame()
                if result is not None:
                    self.on_result(result)

                elapsed = time.time() - t0
                self._last_cycle_ms = elapsed * 1000
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        finally:
            if self._sct is not None:
                self._sct.close()
                self._sct = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── per-frame logic ──────────────────────────────────────────────

    def _detect_one_frame(self) -> Optional[CascadeDetectionResult]:
        self._frame_idx += 1
        now = time.time()

        # ── locked sampling fast path: skip full ROI capture entirely ──
        if (self._sequence_enabled
                and self._sampling_state == "sampling"
                and self._locked_anchor_box is not None
                and self._locked_sub_roi is not None):
            result = self._handle_locked_sampling(None, now)
            if result is not None:
                if result.debug_frame is not None:
                    self._dbg_roi = result.debug_frame
                self._show_live_preview()
                return result
            return None

        # ── cooldown fast path (SEQ mode): skip anchor matching ──
        if self._sequence_enabled and self._sampling_state == "cooldown":
            if now < self._cooldown_until:
                return None
            self._sampling_state = "idle"
            self._seq_triggered = False

        # ── frame skip: skip anchor matching every N frames ──
        if self._anchor_skip > 0 and self._frame_idx % (self._anchor_skip + 1) != 0:
            return None

        # ── normal path: full ROI capture + anchor matching ──
        capture = self._capture_roi()
        if capture is None:
            return None

        original_roi = capture.copy()
        self._dbg_roi = original_roi
        self._dbg_anchor_box = None
        self._dbg_sub_roi_box = None
        self._dbg_sub_roi_box_2 = None
        self._dbg_pattern_box = None
        self._dbg_pattern_box_2 = None
        self._dbg_pattern_best_box = None
        self._dbg_pattern_best_box_2 = None
        self._dbg_anchor_score = 0.0
        self._dbg_pattern_score = 0.0
        self._dbg_pattern_score_2 = 0.0
        self._dbg_pattern_label = None
        self._dbg_pattern_label_2 = None
        self._dbg_status_text = ""

        original_h, original_w = original_roi.shape[:2]

        scale_factor = 1.0
        if self._norm_width > 0 and original_w > 0:
            scale_factor = self._norm_width / original_w
            normalized_roi = resize_by_width(original_roi, self._norm_width)
        else:
            normalized_roi = original_roi

        anchor_group = self.cache.get_anchor_templates()
        if anchor_group is None or not anchor_group.items:
            self._logger.log("[Skip] No anchor templates loaded.", force=True)
            self._dbg_status_text = "no templates"
            self._show_live_preview()
            return None

        anchor_image = preprocess_image(
            normalized_roi,
            use_grayscale=anchor_group.use_grayscale,
            preprocess_mode=anchor_group.preprocess_mode,
            gamma=anchor_group.gamma,
        )

        t_anchor = time.time()
        coarse_th = self.config["anchor"].get("coarse_threshold", 0)
        early_exit = self.config["anchor"].get("early_exit_score", 0.9)
        anchor_result = match_templates_multiscale(
            anchor_image,
            anchor_group.items,
            anchor_group.threshold,
            anchor_group.scale_min * scale_factor,
            anchor_group.scale_max * scale_factor,
            anchor_group.scale_steps,
            anchor_group.use_grayscale,
            label="anchor",
            coarse_threshold=coarse_th if coarse_th > 0 else None,
            early_exit_score=early_exit,
        )
        t_anchor = time.time() - t_anchor

        anchor_box_original: Optional[Tuple[int, int, int, int]] = None
        self._dbg_anchor_score = anchor_result.score
        if anchor_result.matched and anchor_result.box is not None:
            anchor_box_original = scale_box(anchor_result.box, 1.0 / scale_factor)
            self._dbg_anchor_box = anchor_box_original
        else:
            self._dbg_status_text = (f"anchor not found score={anchor_result.score:.2f} "
                                       f"took={t_anchor:.2f}s")

        result = self._handle_sequence_frame(
            original_roi, anchor_result, anchor_box_original, now)

        if result is not None and result.debug_frame is not None:
            self._dbg_roi = result.debug_frame
        self._show_live_preview()
        return result

    # ── capture ──────────────────────────────────────────────────────

    def _capture_roi(self) -> Optional[np.ndarray]:
        if self.roi is None or self._sct is None:
            return None

        try:
            img = self._sct.grab(self.roi)
            return np.array(img)[:, :, :3]
        except Exception as e:
            self._logger.log(f"[Capture] failed: {e}")
            return None

    def _capture_sub_roi_screen(self, roi_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture a sub-ROI region directly from screen using its ROI-relative box."""
        if self.roi is None or self._sct is None:
            return None
        x, y, w, h = roi_box
        monitor = {
            "left": self.roi["left"] + int(x),
            "top": self.roi["top"] + int(y),
            "width": max(1, int(w)),
            "height": max(1, int(h)),
        }
        try:
            img = self._sct.grab(monitor)
            return np.array(img)[:, :, :3]
        except Exception as e:
            self._logger.log(f"[CaptureSub] failed: {e}")
            return None

    # ── debug preview (unified, called every frame) ───────────────────

    def _show_live_preview(self) -> None:
        """Draw and display the debug preview. Called every frame, FPS-throttled."""
        # Destroy window when preview is turned off (same thread, safe)
        if not self._show_preview:
            if self._preview_window_open:
                self._preview_window_open = False
                try:
                    cv2.destroyWindow("Roco Box Detector Debug")
                except Exception:
                    pass
            return
        # During locked sampling the boxes are static — skip imshow/waitKey
        if self._sampling_state == "sampling" and self._locked_anchor_box is not None:
            return
        if self._dbg_roi is None:
            return
        now = time.time()
        if now - self._last_preview_time < 1.0 / self._preview_fps_limit:
            return
        self._last_preview_time = now

        debug = self.debug_drawer.draw_boxes(
            self._dbg_roi,
            anchor_box=self._dbg_anchor_box,
            sub_roi_box=self._dbg_sub_roi_box,
            sub_roi_box_2=self._dbg_sub_roi_box_2,
            pattern_box=self._dbg_pattern_box,
            pattern_box_2=self._dbg_pattern_box_2,
            pattern_best_box=self._dbg_pattern_best_box,
            pattern_best_box_2=self._dbg_pattern_best_box_2,
            anchor_score=self._dbg_anchor_score,
            pattern_score=self._dbg_pattern_score,
            pattern_score_2=self._dbg_pattern_score_2,
            pattern_label=self._dbg_pattern_label,
            pattern_label_2=self._dbg_pattern_label_2,
        )
        bar_h = 26
        cv2.rectangle(debug, (0, 0), (debug.shape[1], bar_h), (0, 0, 0), -1)
        text = f"cycle={self._last_cycle_ms:.0f}ms"
        if self._dbg_status_text:
            text += f"  {self._dbg_status_text}"
        cv2.putText(debug, text, (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        try:
            cv2.imshow("Roco Box Detector Debug", debug)
            self._preview_window_open = True
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._show_preview = False
        except Exception:
            pass

    # ── locked sampling (SEQ fast path, no anchor re-detection) ──────

    def _handle_locked_sampling(
        self, roi: Optional[np.ndarray], now: float
    ) -> Optional[CascadeDetectionResult]:
        locked_anchor = self._locked_anchor_box
        locked_sub_roi = self._locked_sub_roi
        locked_sub_roi_2 = self._locked_sub_roi_2

        elapsed = now - self._sampling_start_time

        # Show locked boxes on preview
        self._dbg_anchor_box = locked_anchor
        self._dbg_sub_roi_box = locked_sub_roi
        self._dbg_sub_roi_box_2 = locked_sub_roi_2
        self._dbg_status_text = (f"locked sampling {len(self._sampling_frames)}"
                                 f"/{self._seq_max_frames}")

        # Delay before first capture
        if elapsed < self._seq_sample_delay:
            return None

        # Interval throttle
        if now < self._sampling_next_time:
            return None

        self._sampling_next_time = now + self._seq_sample_interval

        t_cap = time.time()

        if self._debug_save_enabled():
            # Debug mode: need full ROI for rich debug images
            if roi is None:
                roi = self._capture_roi()
                if roi is None:
                    return None
            sx, sy, sw, sh = locked_sub_roi
            sub_img = roi[sy:sy + sh, sx:sx + sw].copy()
            sub_img_2 = None
            if locked_sub_roi_2 is not None:
                sx2, sy2, sw2, sh2 = locked_sub_roi_2
                sub_img_2 = roi[sy2:sy2 + sh2, sx2:sx2 + sw2].copy()
            roi_img = roi.copy()
        else:
            # Fast path: capture sub-ROIs directly from screen
            sub_img = self._capture_sub_roi_screen(locked_sub_roi)
            sub_img_2 = None
            if locked_sub_roi_2 is not None:
                sub_img_2 = self._capture_sub_roi_screen(locked_sub_roi_2)
            roi_img = None

        # Store first-frame captures for screenshot preview (not overwritten)
        if self._sampling_seq_idx == 0:
            self._latest_sub_roi1 = sub_img
            self._latest_sub_roi2 = sub_img_2

        t_cap = time.time() - t_cap

        if sub_img is None or sub_img.size == 0:
            return None  # capture failed, skip this frame

        sf = SequenceFrame(
            index=self._sampling_seq_idx,
            timestamp=now,
            roi_image=roi_img,
            sub_roi_image=sub_img,
            anchor_box=locked_anchor,
            sub_roi_box=locked_sub_roi,
            sharpness=calculate_sharpness(sub_img)
            if self._seq_use_sharpness else 0.0,
            sub_roi_image_2=sub_img_2,
            sub_roi_box_2=locked_sub_roi_2,
        )
        self._sampling_seq_idx += 1
        self._sampling_frames.append(sf)

        collected = len(self._sampling_frames)
        print(f"[Sequence] captured frame {collected}/{self._seq_max_frames} "
              f"at elapsed={elapsed:.3f}s cap={t_cap*1000:.1f}ms")

        # Screenshot-only mode: return immediately, skip all pattern matching
        if self._screenshot_only:
            print("[Sequence] screenshot-only: frame captured, skipping pattern match")
            self._sampling_frames.clear()
            self._locked_anchor_box = None
            self._locked_sub_roi = None
            self._locked_sub_roi_2 = None
            self._sampling_state = "cooldown"
            self._seq_triggered = True
            self._cooldown_until = now + self._cooldown_seconds
            self._dbg_sub_roi_box = locked_sub_roi
            self._dbg_sub_roi_box_2 = locked_sub_roi_2
            return CascadeDetectionResult(
                matched=False, label=None,
                anchor_result=None, pattern_result=None,
                sub_roi_box=locked_sub_roi, anchor_box=locked_anchor,
                sub_roi_box_2=locked_sub_roi_2, debug_frame=None,
                status="no_match", mode="SEQ",
                sub_roi1_image=self._latest_sub_roi1,
                sub_roi2_image=self._latest_sub_roi2,
            )

        # Defer all pattern matching to finalize — don't block the capture loop.
        if collected >= self._seq_max_frames:
            print("[Sequence] all frames captured, finalizing")
            return self._finalize_sampling(now, roi, locked_anchor, locked_sub_roi)

        return None

    # ── sequence mode ────────────────────────────────────────────────

    def _handle_sequence_frame(
        self,
        roi: np.ndarray,
        anchor_result: MatchResult,
        anchor_box: Optional[Tuple[int, int, int, int]],
        now: float,
    ) -> Optional[CascadeDetectionResult]:
        # cooldown is handled by fast-path in _detect_one_frame,
        # so here only idle state is possible.
        if not anchor_result.matched or anchor_box is None:
            self._seq_triggered = False
            self._logger.log(
                f"[Anchor] matched=False best_score={anchor_result.score:.2f}")
            return None

        if not self._seq_triggered:
            self._start_sampling(roi, anchor_box, anchor_result, now)
            # Transition to sampling — next frame will use locked fast path
            self._dbg_anchor_box = anchor_box
            self._dbg_sub_roi_box = self._locked_sub_roi
            self._dbg_sub_roi_box_2 = self._locked_sub_roi_2
            return CascadeDetectionResult(matched=False, label=None,
                anchor_result=anchor_result, pattern_result=None,
                sub_roi_box=self._locked_sub_roi,
                anchor_box=anchor_box,
                sub_roi_box_2=self._locked_sub_roi_2, debug_frame=None,
                status="sampling", mode="SEQ",
                sub_roi1_image=self._latest_sub_roi1,
                sub_roi2_image=self._latest_sub_roi2)

        return None

    def _start_sampling(self, roi: np.ndarray,
                        anchor_box: Tuple[int, int, int, int],
                        anchor_result: MatchResult, now: float) -> None:
        self._sampling_state = "sampling"
        self._sampling_frames.clear()
        self._sampling_match_results_1.clear()
        self._sampling_match_results_2.clear()
        self._sampling_seq_idx = 0
        self._sampling_start_time = now
        self._sampling_next_time = now + self._seq_sample_delay
        self._anchor_lost_count = 0

        # Lock anchor position — don't re-detect during this cycle
        h, w = roi.shape[:2]
        self._locked_anchor_box = anchor_box

        # Region 1
        sub_roi_cfg = self.config["sub_roi"]
        self._cur_sub_roi = compute_sub_roi(anchor_box, (h, w), SubRoiConfig(
            x_ratio=sub_roi_cfg["x_ratio"], y_ratio=sub_roi_cfg["y_ratio"],
            w_ratio=sub_roi_cfg["w_ratio"], h_ratio=sub_roi_cfg["h_ratio"]))
        self._locked_sub_roi = self._cur_sub_roi

        # Region 2
        if self._roi2_enabled:
            sub_roi2_cfg = self.config["sub_roi_2"]
            self._locked_sub_roi_2 = compute_sub_roi(anchor_box, (h, w), SubRoiConfig(
                x_ratio=sub_roi2_cfg["x_ratio"], y_ratio=sub_roi2_cfg["y_ratio"],
                w_ratio=sub_roi2_cfg["w_ratio"], h_ratio=sub_roi2_cfg["h_ratio"]))
        else:
            self._locked_sub_roi_2 = None

        print(f"[Sequence] Start sampling: delay={self._seq_sample_delay}s "
              f"interval={self._seq_sample_interval}s max_frames={self._seq_max_frames} "
              f"roi2={'on' if self._locked_sub_roi_2 else 'off'}")

    def _finalize_sampling(
        self,
        now: float,
        roi: np.ndarray,
        anchor_box: Tuple[int, int, int, int],
        sub_roi_box: Tuple[int, int, int, int],
        seq_result_1: Optional[SequenceDetectionResult] = None,
        seq_result_2: Optional[SequenceDetectionResult] = None,
    ) -> Optional[CascadeDetectionResult]:
        # Copy frames/results for analysis, then clear mutable state.
        frames = list(self._sampling_frames)
        fr1_list: List[FrameMatchResult] = list(self._sampling_match_results_1)
        fr2_list: List[FrameMatchResult] = list(self._sampling_match_results_2)
        sub_roi_box_2 = self._locked_sub_roi_2
        self._sampling_frames.clear()
        self._sampling_match_results_1.clear()
        self._sampling_match_results_2.clear()
        self._locked_anchor_box = None
        self._locked_sub_roi = None
        self._locked_sub_roi_2 = None

        self._sampling_state = "cooldown"
        self._seq_triggered = True
        self._cooldown_until = now + self._cooldown_seconds

        n_frames = len(frames)
        print(f"[Sequence] collected {n_frames} frames, finalizing...")

        if n_frames == 0:
            self._dbg_sub_roi_box = sub_roi_box
            return CascadeDetectionResult(matched=False, label=None,
                anchor_result=None, pattern_result=None,
                sub_roi_box=sub_roi_box, debug_frame=None,
                status="no_match", mode="SEQ",
                sub_roi1_image=self._latest_sub_roi1,
                sub_roi2_image=self._latest_sub_roi2)

        # --- ROI1 voting (with early termination) ---
        if seq_result_1 is None:
            matched_idx1 = {fr.frame_index for fr in fr1_list}
            for sf in frames:
                if sf.index not in matched_idx1:
                    t = time.time()
                    fr = match_single_frame_to_patterns(
                        sf.sub_roi_image, sf.sub_roi_box, sf.index,
                        self.cache.get_pattern_groups(), roi_name="roi1",
                        norm_width=self._pat_norm_width)
                    fr1_list.append(fr)
                    print(f"[Finalize] roi1 frame {sf.index} label={fr.label} "
                          f"score={fr.score:.2f} took={time.time()-t:.3f}s")
                    # Early vote: stop if enough votes already
                    er = vote_frame_results(
                        fr1_list, self._seq_min_votes, self._seq_min_avg)
                    if er.matched:
                        print(f"[Finalize] roi1 early stop: {er.label} "
                              f"votes={er.vote_count} avg={er.final_score:.2f}")
                        break
            seq_result_1 = vote_frame_results(
                fr1_list, self._seq_min_votes, self._seq_min_avg)

        # --- ROI2 voting (with early termination) ---
        if self._roi2_enabled and self.cache.get_pattern_groups_2():
            if seq_result_2 is None:
                matched_idx2 = {fr.frame_index for fr in fr2_list}
                for sf in frames:
                    if sf.index not in matched_idx2 and sf.sub_roi_image_2 is not None:
                        t = time.time()
                        fr = match_single_frame_to_patterns(
                            sf.sub_roi_image_2, sf.sub_roi_box_2, sf.index,
                            self.cache.get_pattern_groups_2(), roi_name="roi2",
                            norm_width=self._pat_norm_width)
                        fr2_list.append(fr)
                        print(f"[Finalize] roi2 frame {sf.index} label={fr.label} "
                              f"score={fr.score:.2f} took={time.time()-t:.3f}s")
                        er = vote_frame_results(
                            fr2_list, self._seq_min_votes, self._seq_min_avg)
                        if er.matched:
                            print(f"[Finalize] roi2 early stop: {er.label} "
                                  f"votes={er.vote_count} avg={er.final_score:.2f}")
                            break
                seq_result_2 = vote_frame_results(
                    fr2_list, self._seq_min_votes, self._seq_min_avg)
        else:
            seq_result_2 = None

        # --- Combined result ---
        label1 = seq_result_1.label if seq_result_1.matched else None
        label2 = seq_result_2.label if (seq_result_2 and seq_result_2.matched) else None
        fallback = (self.config.get("result_text_overlay", {})
                    .get("roi_fallback", ""))
        print(f"[Finalize] labels: l1={label1} l2={label2} "
              f"roi2_enabled={self._roi2_enabled} fallback='{fallback}'")

        if label1 and label2:
            combined_label = f"{label1} + {label2}"
        elif label1:
            if self._roi2_enabled and fallback:
                combined_label = f"{label1} + {fallback}"
            else:
                combined_label = label1
        elif label2:
            if fallback:
                combined_label = f"{fallback} + {label2}"
            else:
                combined_label = label2
        else:
            combined_label = None

        if self._require_roi2 and self._roi2_enabled:
            matched = bool(label1 and label2)
        else:
            matched = bool(label1 or label2)

        votes_str = (f"r1={seq_result_1.vote_count}/{n_frames}"
                     + (f" r2={seq_result_2.vote_count}/{n_frames}" if seq_result_2 else ""))

        if matched:
            print(f"[Sequence] final matched=True label={combined_label} votes={votes_str}")
            match_dur = self.config.get("overlay", {}).get("match_show_seconds", 3.0)
            self._cooldown_until = max(self._cooldown_until, now + float(match_dur))
        else:
            print(f"[Sequence] final matched=False label1={label1} label2={label2} "
                  f"votes={votes_str}")

        # --- Populate seq_result with dual labels ---
        seq_result_1.label_1 = label1
        seq_result_1.label_2 = label2
        seq_result_1.final_score_1 = seq_result_1.final_score
        seq_result_1.vote_count_1 = seq_result_1.vote_count
        if seq_result_2:
            seq_result_1.label_2 = label2
            seq_result_1.final_score_2 = seq_result_2.final_score
            seq_result_1.vote_count_2 = seq_result_2.vote_count
        seq_result_1.label = combined_label
        seq_result_1.matched = matched

        # --- Save selected frames ---
        if self._save_selected_enabled():
            all_fr = fr1_list + fr2_list
            self._save_selected_voting_frames(frames, all_fr)

        # --- Debug frame ---
        debug_frame = self.debug_drawer.draw_boxes(
            roi, anchor_box=anchor_box,
            sub_roi_box=sub_roi_box, sub_roi_box_2=sub_roi_box_2,
            pattern_box=None, pattern_best_box=None,
            pattern_box_2=None, pattern_best_box_2=None,
            anchor_score=0.0, pattern_score=0.0, pattern_label=None)
        if debug_frame is not None:
            debug_frame = np.ascontiguousarray(debug_frame)

        # Draw roi1 best box
        best_fr1 = None
        for fr in seq_result_1.frame_results:
            if fr.frame_index == seq_result_1.best_frame_index and fr.box_in_roi:
                best_fr1 = fr
                break
        if debug_frame is not None and best_fr1 and best_fr1.box_in_roi:
            px, py, pw, ph = best_fr1.box_in_roi
            color = (0, 255, 0) if best_fr1.matched else (0, 0, 255)
            cv2.rectangle(debug_frame, (px, py), (px + pw, py + ph), color, 2)
            cv2.putText(debug_frame, f"{best_fr1.label or ''} {best_fr1.score:.2f}",
                        (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw roi2 best box
        if seq_result_2:
            best_fr2 = None
            for fr in seq_result_2.frame_results:
                if fr.frame_index == seq_result_2.best_frame_index and fr.box_in_roi:
                    best_fr2 = fr
                    break
            if debug_frame is not None and best_fr2 and best_fr2.box_in_roi:
                px, py, pw, ph = best_fr2.box_in_roi
                color = (255, 255, 0) if best_fr2.matched else (0, 140, 255)
                cv2.rectangle(debug_frame, (px, py), (px + pw, py + ph), color, 2)
                cv2.putText(debug_frame, f"{best_fr2.label or ''} {best_fr2.score:.2f}",
                            (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return CascadeDetectionResult(
            matched=matched,
            label=combined_label,
            anchor_result=None,
            pattern_result=MatchResult(
                stage="pattern", label=combined_label or "unknown",
                matched=matched,
                score=seq_result_1.final_score,
                box=best_fr1.box_in_roi if best_fr1 else None,
                template_path=best_fr1.template_path if best_fr1 else None,
                scale=best_fr1.scale if best_fr1 else None,
            ),
            sub_roi_box=sub_roi_box,
            sub_roi_box_2=sub_roi_box_2,
            debug_frame=debug_frame,
            status="matched" if matched else "no_match",
            mode="SEQ",
            sequence_result=seq_result_1,
            match_votes=votes_str,
            sub_roi1_image=self._latest_sub_roi1,
            sub_roi2_image=self._latest_sub_roi2,
        )

    # ── debug logging ────────────────────────────────────────────────

    def _log_debug_info(
        self,
        anchor_result: MatchResult,
        sub_roi_box: Optional[Tuple[int, int, int, int]],
        pattern_result: Optional[MatchResult],
        best_pattern_score: float,
    ) -> None:
        if not self.config.get("debug", {}).get("print_scores", True):
            return

        if anchor_result.matched:
            scale_text = (
                f"{anchor_result.scale:.2f}"
                if anchor_result.scale is not None
                else "N/A"
            )

            self._logger.log(
                f"[Anchor] matched=True score={anchor_result.score:.2f} "
                f"template={anchor_result.template_path or 'N/A'} "
                f"scale={scale_text} box={anchor_result.box}"
            )

            if sub_roi_box:
                self._logger.log(f"[SubROI] box={sub_roi_box}")

            if pattern_result:
                st = "True" if pattern_result.matched else "False"
                self._logger.log(
                    f"[Pattern:{pattern_result.label}] matched={st} "
                    f"best_score={pattern_result.score:.2f} "
                    f"template={pattern_result.template_path or 'N/A'} "
                    f"scale={pattern_result.scale}"
                )
        else:
            self._logger.log(
                f"[Anchor] matched=False best_score={anchor_result.score:.2f}"
            )
            self._logger.log("[Skip] Anchor not found, skip pattern matching.")

    # ── sequence debug saving ────────────────────────────────────────

    def _save_selected_voting_frames(
        self,
        selected_frames: List[SequenceFrame],
        frame_results: List[FrameMatchResult],
    ) -> None:
        """
        Save only the frames actually used for voting.

        This is the behavior you want:
        - no raw sequence spam;
        - only selected_frame_count frames;
        - frames are collected after sample_delay_seconds;
        - controlled by GUI screenshot switch.
        """
        result_map = {fr.frame_index: fr for fr in frame_results}

        for sf in selected_frames:
            fr = result_map.get(sf.index)
            if fr is None:
                continue
            self._save_selected_frame(sf, fr)

    def _save_selected_frame(
        self,
        sf: SequenceFrame,
        fr: FrameMatchResult,
    ) -> None:
        try:
            os.makedirs(self._seq_selected_dir, exist_ok=True)

            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(sf.timestamp))
            sim_str = f"_sim{sf.avg_similarity:.2f}" if sf.avg_similarity > 0 else ""

            name = (
                f"selected_{ts}_f{sf.index:03d}"
                f"{sim_str}"
                f"_shp{sf.sharpness:.0f}"
                f"_label{fr.label or 'none'}"
                f"_score{fr.score:.2f}.png"
            )

            path = os.path.join(self._seq_selected_dir, name)

            if sf.roi_image is None:
                return  # no full ROI to paint on — debug save skips
            roi_debug = sf.roi_image.copy()

            ax, ay, aw, ah = sf.anchor_box
            cv2.rectangle(roi_debug, (ax, ay), (ax + aw, ay + ah), (255, 0, 0), 2)

            sx, sy, sw, sh = sf.sub_roi_box
            cv2.rectangle(roi_debug, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
            cv2.putText(roi_debug, "ROI1", (sx, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if sf.sub_roi_box_2 is not None:
                sx2, sy2, sw2, sh2 = sf.sub_roi_box_2
                cv2.rectangle(roi_debug, (sx2, sy2), (sx2 + sw2, sy2 + sh2),
                              (0, 200, 255), 2)
                cv2.putText(roi_debug, "ROI2", (sx2, sy2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            if fr.box_in_roi is not None:
                px, py, pw, ph = fr.box_in_roi
                if fr.roi_name == "roi2":
                    color = (255, 255, 0) if fr.matched else (0, 140, 255)
                else:
                    color = (0, 255, 0) if fr.matched else (0, 0, 255)
                cv2.rectangle(roi_debug, (px, py), (px + pw, py + ph), color, 2)
                cv2.putText(roi_debug,
                            f"{fr.roi_name}:{fr.label or ''} {fr.score:.2f}",
                            (px, max(0, py - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # cv2.imwrite fails on Chinese paths — use imencode + tofile
            _, buf = cv2.imencode('.png', roi_debug)
            buf.tofile(path)
            print(f"[Debug] saved selected voting frame: {path}")

        except Exception as e:
            print(f"[Debug] failed to save selected frame: {e}")