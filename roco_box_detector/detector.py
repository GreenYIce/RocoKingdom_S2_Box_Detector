"""Core detection: anchor matching -> Sub-ROI screenshot capture."""

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


@dataclass
class MatchResult:
    stage: str
    label: str
    matched: bool
    score: float
    box: Optional[Tuple[int, int, int, int]]
    template_path: Optional[str]
    scale: Optional[float]
    template_label: Optional[str] = None  # output label of best-matching template


@dataclass
class CascadeDetectionResult:
    matched: bool
    label: Optional[str]
    anchor_result: Optional[MatchResult]
    sub_roi_box: Optional[Tuple[int, int, int, int]]
    anchor_box: Optional[Tuple[int, int, int, int]] = None
    debug_frame: Optional[np.ndarray] = None
    sub_roi_box_2: Optional[Tuple[int, int, int, int]] = None
    status: str = ""
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
    """Multi-template, multi-scale template matching for anchor detection."""
    best_score = -1.0
    best_box = None
    best_tmpl_path = None
    best_tmpl_label = None
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
                    best_tmpl_label = tmpl.label
                    best_scale = sv.scale
                    if early_exit_score > 0 and best_score >= early_exit_score:
                        return MatchResult(
                            stage=label, label=label, matched=True,
                            score=float(best_score), box=best_box,
                            template_path=best_tmpl_path,
                            scale=float(best_scale),
                            template_label=best_tmpl_label)
    else:
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
                    best_tmpl_label = tmpl.label
                    best_scale = scale
                    if early_exit_score > 0 and best_score >= early_exit_score:
                        return MatchResult(
                            stage=label, label=label, matched=True,
                            score=float(best_score), box=best_box,
                            template_path=best_tmpl_path,
                            scale=float(best_scale),
                            template_label=best_tmpl_label)

    matched = best_score >= threshold and best_box is not None

    return MatchResult(
        stage=label, label=label, matched=matched,
        score=float(best_score if best_score >= 0 else 0.0),
        box=best_box, template_path=best_tmpl_path,
        template_label=best_tmpl_label,
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
    """Background detector thread — screenshot-only mode.

    Flow: anchor detect → lock position → delay → capture sub-ROI → cooldown.
    """

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

        self._norm_width = 800
        self._anchor_skip = 0
        self._log_every = 10
        self._show_preview = False
        self._debug_overlay_enabled = False
        self._latest_sub_roi1: Optional[np.ndarray] = None
        self._latest_sub_roi2: Optional[np.ndarray] = None
        self._preview_window_open = False
        self._preview_fps_limit = 15
        self._last_preview_time = 0.0

        # Per-frame debug state
        self._dbg_roi: Optional[np.ndarray] = None
        self._dbg_anchor_box: Optional[Tuple[int, int, int, int]] = None
        self._dbg_sub_roi_box: Optional[Tuple[int, int, int, int]] = None
        self._dbg_sub_roi_box_2: Optional[Tuple[int, int, int, int]] = None
        self._dbg_anchor_score: float = 0.0
        self._dbg_status_text: str = ""
        self._last_cycle_ms: float = 0.0
        # Icon debug state
        self._dbg_icon_thumbnail: Optional[np.ndarray] = None
        self._dbg_icon_present: bool = False
        self._dbg_icon_score: float = 0.0

        self._logger = ThrottledLogger(every_n=self._log_every)
        self._frame_idx = 0

        self._sct: Optional[mss.mss] = None

        # Sequence / cooldown state
        self._sampling_state = "idle"  # "idle" | "cooldown"
        self._cooldown_until = 0.0
        self._seq_triggered = False

        # Config
        self._resolution_scale = 1.0
        self._cooldown_seconds = 1.5
        self._roi2_enabled = False

        # Icon detection state
        self._icon_detection_enabled = False
        self._icon_roi: Optional[Dict[str, int]] = None
        self._icon_threshold = 0.75
        self._icon_disappear_delay = 0.5
        self._icon_delay_until = 0.0
        self._icon_use_grayscale = True
        self._icon_preprocess_mode = "none"
        self._icon_gamma = 0.75
        self._icon_consecutive_present = 0
        self._icon_consecutive_absent = 0
        self._icon_debounce_frames = 3
        self._icon_reappear_count = 0
        self._gate_open_time = 0.0
        self._capture_offsets = [2.0, 2.5, 3.0]
        self._capture_offset_idx = 0
        self._scheduled_frames: list = []  # (offset, roi_image) pairs
        self._icon_status_text = ""

        self._load_all_config(config)

    # ── config ───────────────────────────────────────────────────────

    def _load_all_config(self, config: dict) -> None:
        self.config = config

        rt = config.get("runtime", {})
        self._norm_width = int(rt.get("normalize_roi_width", 800))
        self._anchor_skip = max(0, int(rt.get("anchor_skip_frames", 0)))
        self._log_every = max(1, int(rt.get("log_every_n_frames", 10)))
        self._logger.every_n = self._log_every

        dbg = config.get("debug", {})
        self._show_preview = bool(dbg.get("show_preview_window", False))

        self._cooldown_seconds = float(rt.get("sequence_cooldown_seconds", 1.5))

        sr2 = config.get("sub_roi_2", {})
        self._roi2_enabled = bool(sr2.get("enabled", False))

        # Icon detection config
        ic = config.get("icon_detection", {})
        self._icon_detection_enabled = bool(ic.get("enabled", False))
        self._icon_threshold = float(ic.get("threshold", 0.75))
        self._icon_disappear_delay = float(ic.get("disappear_delay_seconds", 0.5))
        self._icon_debounce_frames = max(1, int(ic.get("debounce_frames", 3)))
        raw_offsets = ic.get("capture_offsets", [2.0, 2.5, 3.0])
        self._capture_offsets = sorted(float(x) for x in raw_offsets) if raw_offsets else [2.0]
        self._icon_use_grayscale = bool(ic.get("use_grayscale", True))
        self._icon_preprocess_mode = ic.get("preprocess_mode", "none")
        self._icon_gamma = float(ic.get("gamma", 0.75))
        if "icon_roi" in ic:
            ir = ic["icon_roi"]
            self._icon_roi = {
                "left": int(ir["left"]), "top": int(ir["top"]),
                "width": max(1, int(ir["width"])),
                "height": max(1, int(ir["height"])),
            }

        # Resolution scale: base = 1440 (2K), so 720p=0.5, 1080p=0.75, 4K=1.5
        heights = {"720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}
        res = config.get("game_resolution", "2K")
        self._resolution_scale = heights.get(res, 1440) / 1440.0
        self.cache.rescale_icon(self._resolution_scale)
        self.cache.rescale_keywords(self._resolution_scale)
        print(f"[Config] Resolution scale: {self._resolution_scale:.2f} ({res})")

    def update_config(self, config: dict) -> None:
        self._load_all_config(config)

    # ── debug save ──────────────────────────────────────────────────

    def _debug_save_enabled(self) -> bool:
        return bool(self.config.get("debug", {}).get("save_debug_frames", False))

    # ── public API ───────────────────────────────────────────────────

    def set_roi(self, roi: Dict[str, int]) -> None:
        self.roi = roi
        self.cache.rescale_anchor(roi["width"], self._norm_width)
        self._sampling_state = "waiting_icon" if self._icon_detection_enabled else "idle"
        self._icon_status_text = "waiting for icon..." if self._icon_detection_enabled else ""
        self._seq_triggered = False
        self._cooldown_until = 0
        self._pause_event.set()

    def set_icon_roi(self, roi: Dict[str, int]) -> None:
        """Set the icon detection ROI (absolute screen coordinates)."""
        self._icon_roi = {
            "left": int(roi["left"]), "top": int(roi["top"]),
            "width": max(1, int(roi["width"])),
            "height": max(1, int(roi["height"])),
        }

    def refresh_anchor_scale(self) -> None:
        if self.roi is not None:
            self.cache.rescale_anchor(self.roi["width"], self._norm_width)

    def set_preview_enabled(self, enabled: bool) -> None:
        self._show_preview = enabled

    def stop(self) -> None:
        self._show_preview = False
        self._stop_event.set()
        self._pause_event.set()

    # ── main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        self._sct = mss.mss()
        _mss_frame = 0
        _MIN_SLEEP = 0.001

        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()

                if self._stop_event.is_set():
                    break

                if self.roi is None:
                    time.sleep(0.1)
                    continue

                _mss_frame += 1
                if _mss_frame % 500 == 0:
                    self._sct.close()
                    self._sct = mss.mss()

                t0 = time.time()

                result = self._detect_one_frame()
                if result is not None:
                    self.on_result(result)

                elapsed = time.time() - t0
                if elapsed > 0.0005:
                    self._last_cycle_ms = elapsed * 1000
                if elapsed < _MIN_SLEEP:
                    time.sleep(_MIN_SLEEP - elapsed)

        finally:
            if self._sct is not None:
                self._sct.close()
                self._sct = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── icon detection ───────────────────────────────────────────────

    def _check_icon_present(self) -> bool:
        """Capture the icon ROI and run single-scale template matching.
        Stores thumbnail + score for debug preview."""
        self._dbg_icon_thumbnail = None
        self._dbg_icon_score = 0.0

        # ── fail-fast checks (log once on first failure) ──
        if self._icon_roi is None:
            if self._frame_idx == 1:
                print("[Icon] ERROR: icon_roi is None — 图标ROI未设置")
            self._dbg_icon_present = False
            return False
        if self._sct is None:
            self._dbg_icon_present = False
            return False
        icon_tmpl = self.cache.get_icon_template()
        if icon_tmpl is None:
            if self._frame_idx <= 3:
                print("[Icon] ERROR: no icon template loaded — "
                      "请检查 templates/icon/ 目录和 config.json 中 icon_detection.templates")
            self._dbg_icon_present = False
            return False

        # ── capture icon ROI ──
        try:
            icon_img = self._sct.grab(self._icon_roi)
            icon_img = np.array(icon_img)[:, :, :3]
        except Exception as e:
            if self._frame_idx <= 3:
                print(f"[Icon] capture failed: {e}")
            self._dbg_icon_present = False
            return False
        if icon_img is None or icon_img.size == 0:
            self._dbg_icon_present = False
            return False

        # Store thumbnail for debug preview
        thumb = cv2.resize(icon_img, (80, 80), interpolation=cv2.INTER_NEAREST)
        self._dbg_icon_thumbnail = np.ascontiguousarray(thumb)

        icon_gray = preprocess_image(
            icon_img,
            use_grayscale=self._icon_use_grayscale,
            preprocess_mode=self._icon_preprocess_mode,
            gamma=self._icon_gamma,
        )

        # ── multi-scale matching (use pre-scaled variants from cache) ──
        img_h, img_w = icon_gray.shape[:2]
        best_score = -1.0
        if icon_tmpl.scaled_variants:
            for sv in icon_tmpl.scaled_variants:
                if sv.width > img_w or sv.height > img_h:
                    continue
                try:
                    result = cv2.matchTemplate(
                        icon_gray, sv.scaled_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > best_score:
                        best_score = max_val
                except cv2.error:
                    continue
        else:
            # Fallback: single-scale
            th, tw = icon_tmpl.image_gray.shape[:2]
            if th <= img_h and tw <= img_w:
                try:
                    result = cv2.matchTemplate(
                        icon_gray, icon_tmpl.image_gray, cv2.TM_CCOEFF_NORMED)
                    _, best_score, _, _ = cv2.minMaxLoc(result)
                except cv2.error as e:
                    if self._frame_idx <= 3:
                        print(f"[Icon] matchTemplate failed: {e}")
                    self._dbg_icon_present = False
                    return False

        self._dbg_icon_score = float(best_score)
        self._dbg_icon_present = best_score >= self._icon_threshold

        # Log only when match state changes
        prev_icon = self._dbg_icon_present
        self._dbg_icon_present = best_score >= self._icon_threshold
        if prev_icon != self._dbg_icon_present:
            state = "MATCH" if self._dbg_icon_present else "below threshold"
            print(f"[Icon] score={best_score:.3f} (threshold={self._icon_threshold}) → {state}")

        return self._dbg_icon_present

    def _handle_waiting_icon(self, now: float) -> Optional[CascadeDetectionResult]:
        """State: icon NOT present. Poll until it appears."""
        self._refresh_preview_roi()
        icon_present = self._check_icon_present()
        if icon_present:
            self._icon_consecutive_present += 1
            if self._icon_consecutive_present >= self._icon_debounce_frames:
                print(f"[IconState] WAIT_ICON → WAIT_GONE (icon detected, "
                      f"score={self._dbg_icon_score:.3f})")
                self._sampling_state = "waiting_icon_gone"
                self._icon_consecutive_present = 0
                self._icon_consecutive_absent = 0
                self._icon_status_text = "icon appeared, waiting gone..."
        else:
            self._icon_consecutive_present = 0
        self._dbg_status_text = self._icon_status_text
        self._show_live_preview()
        if self._frame_idx % 300 == 0:
            return CascadeDetectionResult(
                matched=False, label=None, anchor_result=None,
                sub_roi_box=None, status="icon_waiting",
                sub_roi1_image=None, sub_roi2_image=None)
        return None

    def _handle_waiting_icon_gone(self, now: float) -> Optional[CascadeDetectionResult]:
        """State: icon IS present. Poll until it disappears."""
        self._refresh_preview_roi()
        icon_present = self._check_icon_present()
        if not icon_present:
            self._icon_consecutive_absent += 1
            if self._icon_consecutive_absent >= self._icon_debounce_frames:
                print(f"[IconState] WAIT_GONE → DELAY (icon disappeared, "
                      f"delay={self._icon_disappear_delay:.1f}s)")
                self._sampling_state = "icon_delay"
                self._icon_delay_until = now + self._icon_disappear_delay
                self._icon_consecutive_absent = 0
                self._icon_consecutive_present = 0
                self._icon_status_text = f"icon gone, capture in {self._icon_disappear_delay:.1f}s..."
        else:
            self._icon_consecutive_absent = 0
        self._dbg_status_text = self._icon_status_text
        self._show_live_preview()
        if self._frame_idx % 300 == 0:
            return CascadeDetectionResult(
                matched=False, label=None, anchor_result=None,
                sub_roi_box=None, status="icon_waiting_gone",
                sub_roi1_image=None, sub_roi2_image=None)
        return None

    def _handle_icon_delay(self, now: float) -> Optional[CascadeDetectionResult]:
        """State: icon just disappeared. Wait delay, then open gate for anchor matching.
        Redundant: if icon reappears (debounced), cancel countdown and go back."""
        self._refresh_preview_roi()
        # Redundant check: cancel countdown if icon reappears
        if self._frame_idx % 5 == 0:
            icon_back = self._check_icon_present()
            if icon_back:
                self._icon_reappear_count += 1
                if self._icon_reappear_count >= self._icon_debounce_frames:
                    print(f"[IconState] DELAY → WAIT_GONE (icon reappeared, "
                          f"countdown cancelled)")
                    self._sampling_state = "waiting_icon_gone"
                    self._icon_reappear_count = 0
                    self._icon_consecutive_present = 0
                    self._icon_consecutive_absent = 0
                    self._icon_status_text = "icon reappeared, waiting gone..."
                    self._dbg_status_text = self._icon_status_text
                    self._show_live_preview()
                    return None
            else:
                self._icon_reappear_count = 0
        remaining = self._icon_delay_until - now
        if remaining <= 0:
            print(f"[IconState] DELAY → IDLE_SCHEDULED "
                  f"(gate open, capture at {self._capture_offsets})")
            self._sampling_state = "idle_scheduled"
            self._gate_open_time = now
            self._capture_offset_idx = 0
            self._icon_status_text = ""
            self._dbg_status_text = ""
            return None
        self._dbg_status_text = f"capture in {remaining:.1f}s"
        self._show_live_preview()
        if self._frame_idx % 300 == 0:
            return CascadeDetectionResult(
                matched=False, label=None, anchor_result=None,
                sub_roi_box=None, status="icon_delay",
                sub_roi1_image=None, sub_roi2_image=None)
        return None

    def _save_scheduled_frame(self, roi_img, anchor_result, scale_factor,
                               offset: float) -> None:
        """Save scheduled capture frame to debug_output for timing review."""
        try:
            out_dir = resolve_path(
                self.config.get("debug", {}).get("debug_output_dir", "debug_output"))
            os.makedirs(out_dir, exist_ok=True)
            debug = roi_img.copy()
            # Draw anchor box if found
            if anchor_result.matched and anchor_result.box is not None:
                box = scale_box(anchor_result.box, 1.0 / scale_factor)
                x, y, w, h = box
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug, f"anchor {anchor_result.score:.2f}",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
            # Status bar
            status = "MATCH" if anchor_result.matched else "NO MATCH"
            cv2.rectangle(debug, (0, 0), (debug.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(debug, f"+{offset:.1f}s best={anchor_result.score:.3f} {status}",
                        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if anchor_result.matched else (0, 0, 255), 1)
            ts = time.strftime("%H%M%S")
            name = f"scheduled_{ts}_+{offset:.2f}s_score{anchor_result.score:.3f}.png"
            path = os.path.join(out_dir, name)
            _, buf = cv2.imencode('.png', debug)
            buf.tofile(path)
            print(f"[Debug] saved scheduled frame: {name}")
        except Exception as e:
            print(f"[Debug] failed to save scheduled frame: {e}")

    def _refresh_preview_roi(self) -> None:
        """Periodically capture game ROI for debug preview.
        Always captures on first call (when _dbg_roi is None),
        then throttles to every 15 frames."""
        if self._dbg_roi is not None and self._frame_idx % 15 != 0:
            return
        capture = self._capture_roi()
        if capture is not None:
            self._dbg_roi = capture.copy()

    # ── scheduled capture (icon mode gate open) ──────────────────────

    def _handle_idle_scheduled(self, now: float) -> Optional[CascadeDetectionResult]:
        """Phase 1: capture frames at each offset (no matching yet).
        Phase 2: run anchor matching on all captured frames, pick best."""
        # ── Phase 2: all captures done, run matching ──
        if self._capture_offset_idx >= len(self._capture_offsets):
            return self._finalize_scheduled(now)

        # ── Phase 1: capture at next offset ──
        next_offset = self._capture_offsets[self._capture_offset_idx]
        elapsed = now - self._gate_open_time
        if elapsed < next_offset:
            self._dbg_status_text = (f"capture in {next_offset - elapsed:.1f}s "
                                     f"({self._capture_offset_idx + 1}/"
                                     f"{len(self._capture_offsets)})")
            self._show_live_preview()
            return None

        self._capture_offset_idx += 1
        capture = self._capture_roi()
        if capture is not None:
            self._scheduled_frames.append((next_offset, capture.copy()))
            self._dbg_roi = capture
            print(f"[IconState] captured +{next_offset:.2f}s "
                  f"({self._capture_offset_idx}/{len(self._capture_offsets)})")

        # Check if all captured
        if self._capture_offset_idx >= len(self._capture_offsets):
            print(f"[IconState] all {len(self._scheduled_frames)} frames captured, "
                  f"starting matching phase...")
            self._dbg_status_text = "matching..."
            self._show_live_preview()
            return None

        return None

    def _finalize_scheduled(self, now: float) -> Optional[CascadeDetectionResult]:
        """Run anchor matching on all captured frames, pick the best match."""
        frames = self._scheduled_frames
        self._scheduled_frames = []

        if not frames:
            print(f"[IconState] IDLE_SCHEDULED → COOLDOWN (no frames captured)")
            self._sampling_state = "cooldown"
            self._cooldown_until = now + self._cooldown_seconds
            self._seq_triggered = True
            self._dbg_status_text = ""
            return None

        anchor_group = self.cache.get_anchor_templates()
        if anchor_group is None or not anchor_group.items:
            self._sampling_state = "cooldown"
            self._cooldown_until = now + self._cooldown_seconds
            self._seq_triggered = True
            return None

        best_offset = 0.0
        best_score = -1.0
        best_box = None
        best_frame = None

        for offset, roi_img in frames:
            h, w = roi_img.shape[:2]
            scale_factor = 1.0
            if self._norm_width > 0 and w > 0:
                scale_factor = self._norm_width / w
                normalized = resize_by_width(roi_img, self._norm_width)
            else:
                normalized = roi_img

            anchor_image = preprocess_image(
                normalized,
                use_grayscale=anchor_group.use_grayscale,
                preprocess_mode=anchor_group.preprocess_mode,
                gamma=anchor_group.gamma,
            )

            coarse_th = self.config["anchor"].get("coarse_threshold", 0)
            early_exit = self.config["anchor"].get("early_exit_score", 0.9)
            result = match_templates_multiscale(
                anchor_image, anchor_group.items,
                anchor_group.threshold,
                anchor_group.scale_min * scale_factor,
                anchor_group.scale_max * scale_factor,
                anchor_group.scale_steps,
                anchor_group.use_grayscale,
                label="anchor",
                coarse_threshold=coarse_th if coarse_th > 0 else None,
                early_exit_score=early_exit,
            )

            # Debug save
            if self._debug_save_enabled():
                self._save_scheduled_frame(roi_img, result, scale_factor, offset)

            matched = result.matched and result.box is not None
            print(f"[IconState] +{offset:.2f}s: score={result.score:.3f} "
                  f"{'MATCH' if matched else 'no match'}")

            if matched and result.score > best_score:
                best_score = result.score
                best_box = scale_box(result.box, 1.0 / scale_factor)
                best_frame = roi_img
                best_offset = offset

        if best_box is not None and best_frame is not None:
            print(f"[IconState] BEST anchor at +{best_offset:.2f}s "
                  f"score={best_score:.3f} → capture")
            self._dbg_anchor_box = best_box

            # Compute pattern1 region and capture screenshot
            h, w = best_frame.shape[:2]
            sub_roi_cfg = self.config["sub_roi"]
            p1_box = compute_sub_roi(best_box, (h, w), SubRoiConfig(
                x_ratio=sub_roi_cfg["x_ratio"], y_ratio=sub_roi_cfg["y_ratio"],
                w_ratio=sub_roi_cfg["w_ratio"], h_ratio=sub_roi_cfg["h_ratio"]))
            px, py, pw, ph = p1_box
            pattern1_img = best_frame[py:py + ph, px:px + pw].copy()
            self._latest_sub_roi1 = pattern1_img

            self._sampling_state = "cooldown"
            self._cooldown_until = now + self._cooldown_seconds
            self._seq_triggered = True
            self._dbg_status_text = "anchor matched"

            return CascadeDetectionResult(
                matched=False, label=None,
                anchor_result=MatchResult(
                    stage="anchor", label="anchor", matched=True,
                    score=best_score, box=best_box,
                    template_path=None, scale=None,
                ),
                sub_roi_box=p1_box,
                anchor_box=best_box,
                status="matched",
                sub_roi1_image=pattern1_img,
            )

        # No anchor found in any frame → "普通"
        print(f"[IconState] IDLE_SCHEDULED → COOLDOWN "
              f"(no anchor in {len(frames)} frames → '普通')")
        self._sampling_state = "cooldown"
        self._cooldown_until = now + self._cooldown_seconds
        self._seq_triggered = True
        self._dbg_status_text = "普通"
        return CascadeDetectionResult(
            matched=False, label="普通",
            anchor_result=None,
            sub_roi_box=None,
            status="normal_fallback",
            sub_roi1_image=None,
        )

    # ── per-frame logic ──────────────────────────────────────────────

    def _detect_one_frame(self) -> Optional[CascadeDetectionResult]:
        self._frame_idx += 1
        now = time.time()

        # ── icon detection fast paths (before anchor matching) ──
        if self._icon_detection_enabled:
            if self._sampling_state == "waiting_icon":
                return self._handle_waiting_icon(now)
            if self._sampling_state == "waiting_icon_gone":
                return self._handle_waiting_icon_gone(now)
            if self._sampling_state == "icon_delay":
                return self._handle_icon_delay(now)
            if self._sampling_state == "idle_scheduled":
                return self._handle_idle_scheduled(now)

        # ── cooldown fast path ──
        if self._sampling_state == "cooldown":
            if now < self._cooldown_until:
                # Icon mode: keep thumbnail live during cooldown
                if self._icon_detection_enabled and self._frame_idx % 15 == 0:
                    self._check_icon_present()
                    self._show_live_preview()
                return None
            self._seq_triggered = False
            if self._icon_detection_enabled:
                # After cooldown, re-enter icon detection cycle
                icon_present = self._check_icon_present()
                if icon_present:
                    print(f"[IconState] COOLDOWN → WAIT_GONE (icon still present)")
                    self._sampling_state = "waiting_icon_gone"
                    self._icon_status_text = "waiting for icon to disappear..."
                else:
                    print(f"[IconState] COOLDOWN → WAIT_ICON (icon absent)")
                    self._sampling_state = "waiting_icon"
                    self._icon_status_text = "waiting for icon..."
                self._icon_consecutive_present = 0
                self._icon_consecutive_absent = 0
            else:
                self._sampling_state = "idle"

        # ── frame skip ──
        if self._anchor_skip > 0 and self._frame_idx % (self._anchor_skip + 1) != 0:
            return None

        # ── normal path: ROI capture + anchor matching ──
        if self._icon_detection_enabled and self._frame_idx % 300 == 0:
            print(f"[IconState] anchor search active (gate open)")
        capture = self._capture_roi()
        if capture is None:
            return None

        need_copy = self._show_preview or self._debug_save_enabled()
        original_roi = capture.copy() if need_copy else capture
        self._dbg_roi = original_roi
        self._dbg_anchor_box = None
        self._dbg_sub_roi_box = None
        self._dbg_sub_roi_box_2 = None
        self._dbg_anchor_score = 0.0
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
            self._logger.log(f"[Anchor] matched score={anchor_result.score:.2f}")
        else:
            self._dbg_status_text = (f"anchor not found score={anchor_result.score:.2f} "
                                       f"took={t_anchor:.2f}s")
            self._logger.log(f"[Anchor] no match best={anchor_result.score:.2f}")

        # Legacy non-icon path — logging only; keyword detection uses icon mode
        self._show_live_preview()
        return None

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

    # ── debug preview ────────────────────────────────────────────────

    def _show_live_preview(self) -> None:
        if not self._show_preview:
            if self._preview_window_open:
                self._preview_window_open = False
                try:
                    cv2.destroyWindow("Roco Box Detector Debug")
                except Exception:
                    pass
            return
        if self._dbg_roi is None and not self._icon_detection_enabled:
            return
        if self._dbg_roi is None and self._dbg_icon_thumbnail is None:
            return
        now = time.time()
        if now - self._last_preview_time < 1.0 / self._preview_fps_limit:
            return
        self._last_preview_time = now

        if self._dbg_roi is not None:
            debug = self.debug_drawer.draw_boxes(
                self._dbg_roi,
                anchor_box=self._dbg_anchor_box,
                sub_roi_box=self._dbg_sub_roi_box,
                sub_roi_box_2=self._dbg_sub_roi_box_2,
                anchor_score=self._dbg_anchor_score,
            )
            debug = np.ascontiguousarray(debug)
        else:
            # Icon-only mode: game ROI not yet captured, use placeholder
            debug = np.zeros((360, 480, 3), dtype=np.uint8)

        # ── pad debug image to fit icon thumbnail if needed ──
        if self._icon_detection_enabled and self._dbg_icon_thumbnail is not None:
            thumb = self._dbg_icon_thumbnail
            th, tw = thumb.shape[:2]
            margin = 4
            min_w = tw + margin
            min_h = 28 + th  # iy=28 + thumb height
            if debug.shape[1] < min_w or debug.shape[0] < min_h:
                new_w = max(debug.shape[1], min_w)
                new_h = max(debug.shape[0], min_h)
                padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                padded[:debug.shape[0], :debug.shape[1]] = debug
                debug = padded

        # ── icon thumbnail overlay (top-right corner) ──
        if self._icon_detection_enabled and self._dbg_icon_thumbnail is not None:
            thumb = self._dbg_icon_thumbnail
            th, tw = thumb.shape[:2]
            ix = debug.shape[1] - tw - margin
            iy = 28
            border_color = (0, 220, 0) if self._dbg_icon_present else (0, 0, 220)
            cv2.rectangle(debug, (ix - 2, iy - 2), (ix + tw + 2, iy + th + 2),
                          border_color, 2)
            debug[iy:iy + th, ix:ix + tw] = thumb
            cv2.putText(debug, f"icon {self._dbg_icon_score:.2f}",
                        (ix, iy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, border_color, 1)

        bar_h = 26
        cv2.rectangle(debug, (0, 0), (debug.shape[1], bar_h), (0, 0, 0), -1)
        eff_fps = 1000.0 / self._last_cycle_ms if self._last_cycle_ms > 0 else 0

        # ── status bar text ──
        text = f"cycle={self._last_cycle_ms:.0f}ms ({eff_fps:.0f}fps)"
        if self._icon_detection_enabled:
            # Colored dot + state text
            dot = "(●)" if self._dbg_icon_present else "(○)"
            state_map = {
                "waiting_icon": f"{dot} waiting for icon",
                "waiting_icon_gone": f"{dot} icon present",
                "icon_delay": f"{dot} {self._dbg_status_text}",
            }
            icon_text = state_map.get(self._sampling_state, f"{dot} {self._dbg_status_text}")
            text += f"  |  {icon_text}"
        elif self._dbg_status_text:
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

