"""Roco Box Detector — main entry point."""

import json
import os
import sys
from typing import Optional, Tuple

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from selector import ROISelector
from overlay import ResultTextOverlay, DebugBoxOverlay
from detector import CascadeDetector, CascadeDetectionResult
from template_cache import TemplateCache
from debug_utils import DebugDrawer
from settings_panel import SettingsWindow
from image_utils import resolve_path

import keyboard


CONFIG_PATH = resolve_path("config.json")


# ── Thread-safe bridge ──────────────────────────────────────────────

class AppBridge(QObject):
    """Signals for cross-thread communication: detector → main thread."""
    result_ready = pyqtSignal(object)   # CascadeDetectionResult
    request_re_select = pyqtSignal()
    request_toggle_panel = pyqtSignal()
    request_quit = pyqtSignal()


# ── App ──────────────────────────────────────────────────────────────

class App:
    """Main application controller."""

    def __init__(self):
        self.config = self._load_config()
        self._ensure_dirs()

        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        # Thread-safe bridge
        self._bridge = AppBridge()
        self._bridge.result_ready.connect(self._on_result_main_thread)
        self._bridge.request_re_select.connect(self._start_roi_selection)
        self._bridge.request_toggle_panel.connect(
            lambda: self.result_text.toggle_visibility())
        self._bridge.request_quit.connect(self._quit)

        # Debug box overlay (transparent boxes painted over game screen)
        self.debug_box_overlay = DebugBoxOverlay()

        # Debug drawer & template cache
        self.debug_drawer = DebugDrawer(self.config)
        print("[Init] Loading templates...")
        self.cache = TemplateCache(self.config)
        print(f"[Init] Anchor templates: {self.cache.anchor_count}")
        print(f"[Init] Pattern templates: {self.cache.pattern_count} "
              f"(across {len(self.cache.pattern_groups)} groups)")

        # Result text overlay (floating recognition history panel)
        self.result_text = ResultTextOverlay(self.config)
        self.result_text._signals.position_changed.connect(self._on_result_text_moved)
        self.result_text._signals.size_changed.connect(self._on_result_text_resized)
        self.result_text._signals.open_settings.connect(self._toggle_settings)
        self.result_text._signals.request_re_select.connect(
            lambda: self._bridge.request_re_select.emit())
        self.result_text._signals.toggle_debug_save.connect(self._on_toggle_debug_save)
        self.result_text._signals.toggle_preview.connect(self._on_toggle_preview)
        self.result_text._signals.toggle_debug_overlay.connect(self._on_toggle_debug_overlay)
        self.result_text._signals.request_quit.connect(self._quit)

        # Detector — callback emits signal via bridge (thread-safe)
        self.detector = CascadeDetector(
            config=self.config,
            cache=self.cache,
            debug_drawer=self.debug_drawer,
            on_result=lambda r: self._bridge.result_ready.emit(r),
        )

        # Settings panel
        self.settings_window = SettingsWindow(self.config)
        self.settings_window.config_saved.connect(self._apply_settings)

        self.roi = None
        self._shutting_down = False

    def _load_config(self) -> dict:
        if not os.path.exists(CONFIG_PATH):
            print(f"[ERROR] Config file not found: {CONFIG_PATH}")
            sys.exit(1)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_dirs(self) -> None:
        dirs = [
            "templates/box_anchor",
            "templates/patterns",
            "templates/patterns_2",
            self.config.get("debug", {}).get("debug_output_dir", "debug_output"),
        ]
        for d in dirs:
            dpath = resolve_path(d)
            os.makedirs(dpath, exist_ok=True)

    def run(self) -> None:
        self._register_hotkeys()
        self._start_roi_selection()
        self.detector.start()
        sys.exit(self.app.exec_())

    # ── ROI selection ─────────────────────────────────────────────────

    def _start_roi_selection(self) -> None:
        selector = ROISelector()
        roi = selector.select()
        if roi is not None:
            self.roi = roi
            self.detector.set_roi(roi)
            print(f"[ROI] Selected: left={roi['left']}, top={roi['top']}, "
                  f"width={roi['width']}, height={roi['height']}")
        else:
            print("[ROI] Selection cancelled.")

    # ── result processing (runs on main thread via bridge) ────────────

    def _on_result_main_thread(self, result: CascadeDetectionResult) -> None:
        """Called on main thread via AppBridge signal. Safe to touch overlay."""
        status = result.status

        # Always update screenshot preview if captures exist
        self.result_text.update_screenshot_preview(
            result.sub_roi1_image, result.sub_roi2_image)

        print(f"[Callback] status={status} matched={result.matched} "
              f"label={result.label}"
              f"{' votes=' + result.match_votes if result.match_votes else ''}")

        self._update_debug_boxes(result)

        if status == "sampling":
            self.result_text.show_sampling()
        elif status == "matched" and result.label:
            score = 0
            votes_text = result.match_votes
            if result.sequence_result is not None:
                score = result.sequence_result.final_score
            elif result.pattern_result is not None:
                score = result.pattern_result.score
            self.result_text.show_match(f"{result.label} {score:.2f}"
                                        + (f" 票数 {votes_text}" if votes_text else ""))
            self.result_text.add_result(result.label)
        elif status == "no_match":
            self.result_text.show_no_match()

    # ── hotkeys (emit signals; bridge queues to main thread) ──────────

    def _register_hotkeys(self) -> None:
        try:
            keyboard.add_hotkey(
                "ctrl+shift+r",
                lambda: self._bridge.request_re_select.emit(),
                suppress=False)
            keyboard.add_hotkey(
                "ctrl+shift+h",
                lambda: self._bridge.request_toggle_panel.emit(),
                suppress=False)
            keyboard.add_hotkey(
                "ctrl+shift+q",
                lambda: self._bridge.request_quit.emit(),
                suppress=False)
            print("[Hotkeys] Ctrl+Shift+R: re-select ROI | "
                  "Ctrl+Shift+H: toggle panel | Ctrl+Shift+Q: quit")
        except Exception as e:
            print(f"[WARN] Could not register hotkeys (need admin?): {e}")

    # ── settings / toggles ────────────────────────────────────────────
    def _on_toggle_debug_save(self, enabled: bool) -> None:
        """Toggle all debug screenshot saving on/off."""
        self.config.setdefault("debug", {})
        self.config["debug"]["save_debug_frames"] = enabled
        self.config["debug"]["save_sequence_frames"] = enabled
        self.config["debug"]["save_selected_frames"] = enabled

        self.debug_drawer.debug["save_debug_frames"] = enabled
        # Persist to file so toggle survives restart
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print(f"[Debug] Screenshot saving: {'ON' if enabled else 'OFF'}")

    def _on_result_text_moved(self, x: int, y: int) -> None:
        self.config.setdefault("result_text_overlay", {})
        self.config["result_text_overlay"]["x"] = x
        self.config["result_text_overlay"]["y"] = y

    def _on_result_text_resized(self, w: int, h: int) -> None:
        self.config.setdefault("result_text_overlay", {})
        self.config["result_text_overlay"]["width"] = w
        self.config["result_text_overlay"]["height"] = h

    def _on_toggle_preview(self, enabled: bool) -> None:
        self.config["debug"]["show_preview_window"] = enabled
        self.detector.set_preview_enabled(enabled)
        print(f"[Preview] Debug window: {'ON' if enabled else 'OFF'}")

    def _on_toggle_debug_overlay(self, enabled: bool) -> None:
        print(f"[DebugOverlay] Toggle: {'ON' if enabled else 'OFF'}")
        if enabled:
            self.debug_box_overlay.show_overlay()
        else:
            self.debug_box_overlay.hide_overlay()
        self.detector._debug_overlay_enabled = enabled

    @staticmethod
    def _to_screen(rx: int, ry: int, box) -> Optional[Tuple]:
        """Convert ROI-relative box to screen coordinates."""
        if box is None:
            return None
        x, y, w, h = box
        return (rx + x, ry + y, w, h)

    def _update_debug_boxes(self, result: CascadeDetectionResult) -> None:
        """Compute screen-space boxes and update the debug overlay."""
        if not self.debug_box_overlay.isVisible():
            return
        if self.roi is None:
            return
        rx, ry = self.roi["left"], self.roi["top"]
        boxes = []

        # Anchor box (blue) — use original-ROI coords if available
        anchor_box = result.anchor_box
        if not anchor_box and result.anchor_result:
            anchor_box = result.anchor_result.box
        if anchor_box:
            b = self._to_screen(rx, ry, anchor_box)
            if b:
                score = result.anchor_result.score if result.anchor_result else 0
                boxes.append((*b, 255, 0, 0, f"anchor {score:.2f}"))

        # Sub-ROI1 box (yellow)
        if result.sub_roi_box:
            b = self._to_screen(rx, ry, result.sub_roi_box)
            if b:
                boxes.append((*b, 0, 255, 255, "ROI1"))

        # Sub-ROI2 box (orange) — if available in result
        if hasattr(result, 'sub_roi_box_2') and result.sub_roi_box_2:
            b = self._to_screen(rx, ry, result.sub_roi_box_2)
            if b:
                boxes.append((*b, 0, 200, 255, "ROI2"))

        # Pattern box — green if matched, red if best-unmatched
        if result.pattern_result and result.pattern_result.box:
            b = self._to_screen(rx, ry, result.pattern_result.box)
            if b:
                if result.pattern_result.matched:
                    label = f"{result.pattern_result.label or ''} {result.pattern_result.score:.2f}"
                    boxes.append((*b, 0, 255, 0, label))
                else:
                    boxes.append((*b, 0, 0, 255,
                                  f"best {result.pattern_result.score:.2f}"))

        self.debug_box_overlay.update_boxes(boxes)

        # Auto-hide boxes after 3s of no updates
        if hasattr(self, '_box_clear_timer'):
            self._box_clear_timer.stop()
        else:
            self._box_clear_timer = QTimer()
            self._box_clear_timer.setSingleShot(True)
            self._box_clear_timer.timeout.connect(self.debug_box_overlay.clear_boxes)
        self._box_clear_timer.start(3000)

    def _toggle_settings(self) -> None:
        if self.settings_window.isVisible():
            self.settings_window.hide()
        else:
            self.settings_window._load_all()  # refresh from live config
            self.settings_window.show()

    def _apply_settings(self, new_config: dict) -> None:
        print("[Settings] Applying changes to live components...")
        self.config = new_config
        self.cache.reload(new_config)
        self.detector.update_config(new_config)
        self.result_text.reload_config(new_config)
        self.debug_drawer = DebugDrawer(new_config)
        self.detector.debug_drawer = self.debug_drawer
        print("[Settings] Changes applied.")

    def _quit(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        print("[Shutdown] Stopping detector...")
        self.detector.stop()
        self.detector.join(timeout=2.0)
        print("[Shutdown] Closing windows...")
        self.settings_window.close()
        self.result_text.close()
        self.debug_box_overlay.close()
        print("[Shutdown] Done.")
        self.app.quit()


if __name__ == "__main__":
    app = App()
    app.run()
