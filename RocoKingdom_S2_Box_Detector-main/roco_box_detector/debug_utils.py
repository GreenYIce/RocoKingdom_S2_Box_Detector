"""Debug visualization: draw boxes, save debug images, throttled logging."""

import cv2
import os
import time
from datetime import datetime
from typing import Optional, Tuple

from image_utils import resolve_path


class DebugDrawer:
    """Handles drawing detection boxes and saving debug frames."""

    def __init__(self, config: dict):
        self.config = config
        self.debug = config.get("debug", {})
        self.enabled = self.debug.get("enabled", True)
        self.output_dir = resolve_path(
            self.debug.get("debug_output_dir", "debug_output"))
        self.save_interval = self.debug.get("save_every_n_seconds", 3)
        self._last_save_time = 0.0
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_boxes(
        self,
        frame: np.ndarray,
        anchor_box: Optional[Tuple[int, int, int, int]] = None,
        sub_roi_box: Optional[Tuple[int, int, int, int]] = None,
        sub_roi_box_2: Optional[Tuple[int, int, int, int]] = None,
        pattern_box: Optional[Tuple[int, int, int, int]] = None,
        pattern_box_2: Optional[Tuple[int, int, int, int]] = None,
        pattern_best_box: Optional[Tuple[int, int, int, int]] = None,
        pattern_best_box_2: Optional[Tuple[int, int, int, int]] = None,
        anchor_score: float = 0.0,
        pattern_score: float = 0.0,
        pattern_score_2: float = 0.0,
        pattern_label: Optional[str] = None,
        pattern_label_2: Optional[str] = None,
    ) -> np.ndarray:
        """Draw detection boxes on a debug copy of the frame."""
        if not self.enabled or frame is None:
            return frame

        debug = frame.copy()

        if self.debug.get("draw_anchor_box") and anchor_box is not None:
            x, y, w, h = anchor_box
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(debug, f"anchor {anchor_score:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if self.debug.get("draw_sub_roi_box") and sub_roi_box is not None:
            x, y, w, h = sub_roi_box
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(debug, "ROI1", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if self.debug.get("draw_sub_roi_box") and sub_roi_box_2 is not None:
            x, y, w, h = sub_roi_box_2
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.putText(debug, "ROI2", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        if self.debug.get("draw_pattern_box"):
            if pattern_box is not None:
                x, y, w, h = pattern_box
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{pattern_label or 'pat'} {pattern_score:.2f}"
                cv2.putText(debug, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif pattern_best_box is not None:
                x, y, w, h = pattern_best_box
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(debug, f"best {pattern_score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            if pattern_box_2 is not None:
                x, y, w, h = pattern_box_2
                cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 255, 0), 2)
                label = f"{pattern_label_2 or 'pat2'} {pattern_score_2:.2f}"
                cv2.putText(debug, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif pattern_best_box_2 is not None:
                x, y, w, h = pattern_best_box_2
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 140, 255), 2)
                cv2.putText(debug, f"best2 {pattern_score_2:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

        return debug

    def maybe_save(self, frame: np.ndarray, tag: str = "") -> None:
        """Save debug frame, throttled by save_interval."""
        if not self.enabled or not self.debug.get("save_debug_frames"):
            return
        now = time.time()
        if now - self._last_save_time < self.save_interval:
            return
        self._last_save_time = now
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"debug_{ts}{'_' + tag if tag else ''}.png"
        path = os.path.join(self.output_dir, name)
        _, buf = cv2.imencode('.png', frame)
        buf.tofile(path)


class ThrottledLogger:
    """Logs messages at most once every N calls."""

    def __init__(self, every_n: int = 10):
        self.every_n = every_n
        self._counter = 0

    def log(self, msg: str, force: bool = False) -> None:
        if force:
            print(msg)
            return
        self._counter += 1
        if self._counter % self.every_n == 0:
            print(msg)

    def reset(self):
        self._counter = 0
