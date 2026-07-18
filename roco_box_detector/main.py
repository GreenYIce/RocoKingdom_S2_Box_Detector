"""Roco Box Detector — main entry point."""

import json
import os
import sys
from typing import Optional, Tuple

from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from selector import ROISelector
from overlay import ResultTextOverlay, DebugBoxOverlay
from detector import CascadeDetector, CascadeDetectionResult
from template_cache import TemplateCache
from debug_utils import DebugDrawer 
from settings_panel import SettingsWindow
from startup_dialog import StartupDialog
from image_utils import resolve_path

import keyboard


# Resolve config path: always prefer exe directory in frozen mode so saves persist
def _get_config_path() -> str:
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        cfg = os.path.join(exe_dir, "config.json")
        if os.path.exists(cfg):
            return cfg
        # First run: copy bundled config to exe dir so it's writable
        bundled = resolve_path("config.json")
        if os.path.exists(bundled) and bundled != cfg:
            import shutil
            shutil.copy(bundled, cfg)
        return cfg
    return resolve_path("config.json")

CONFIG_PATH = _get_config_path()


# ── Thread-safe bridge ──────────────────────────────────────────────

class AppBridge(QObject):
    """Signals for cross-thread communication: detector → main thread."""
    result_ready = pyqtSignal(object)   # CascadeDetectionResult
    request_re_select = pyqtSignal()
    request_toggle_panel = pyqtSignal()
    request_toggle_lock = pyqtSignal()
    request_quit = pyqtSignal()


# ── App ──────────────────────────────────────────────────────────────

class App:
    """Main application controller — screenshot-only mode."""

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
        self._bridge.request_toggle_lock.connect(
            lambda: self.result_text.toggle_mouse_lock())
        self._bridge.request_quit.connect(self._quit)

        # Debug box overlay
        self.debug_box_overlay = DebugBoxOverlay()

        # Debug drawer & template cache
        self.debug_drawer = DebugDrawer(self.config)
        print("[Init] Loading templates...")
        self.cache = TemplateCache(self.config)
        print(f"[Init] Anchor templates: {self.cache.anchor_count}")
        if self.config.get("icon_detection", {}).get("enabled"):
            icon_tmpl = self.cache.get_icon_template()
            if icon_tmpl:
                print(f"[Init] Icon template: {icon_tmpl.path} "
                      f"({icon_tmpl.image_gray.shape[1]}x{icon_tmpl.image_gray.shape[0]})")
                roi = self.config["icon_detection"].get("icon_roi", {})
                print(f"[Init] Icon ROI: left={roi.get('left')} top={roi.get('top')} "
                      f"{roi.get('width')}x{roi.get('height')}")
            else:
                print("[Init] Icon template: NOT LOADED — check templates/icon/ directory")

        # Keyword templates
        if self.config.get("keyword_detection", {}).get("enabled"):
            kw_group = self.cache.get_keyword_templates()
            if kw_group and kw_group.items:
                labels = set(it.label for it in kw_group.items)
                print(f"[Init] Keyword templates: {', '.join(sorted(labels))}")
            else:
                print("[Init] Keyword templates: NOT LOADED — check templates/keywords/")

        # Result overlay (screenshot preview + counter)
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

        # Detector
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
        self._selecting_roi = False
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
            "templates/icon",
            "templates/keywords",
            self.config.get("debug", {}).get("debug_output_dir", "debug_output"),
        ]
        for d in dirs:
            dpath = resolve_path(d)
            os.makedirs(dpath, exist_ok=True)

    def run(self) -> None:
        dlg = StartupDialog(self.config)
        if dlg.exec_() == QDialog.Accepted:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self._apply_settings(self.config)

        self._register_hotkeys()
        self._start_roi_selection()
        self.detector.start()
        sys.exit(self.app.exec_())

    # ── ROI selection ─────────────────────────────────────────────────

    def _start_roi_selection(self) -> None:
        if self._selecting_roi:
            return
        self._selecting_roi = True
        try:
            selector = ROISelector()
            roi = selector.select(message="框选小加尔提示出现位置")
            if roi is not None:
                self.roi = roi
                self.detector.set_roi(roi)
                print(f"[ROI] Selected: left={roi['left']}, top={roi['top']}, "
                      f"width={roi['width']}, height={roi['height']}")

                # Icon detection mode: prompt for second ROI
                if self.config.get("icon_detection", {}).get("enabled", False):
                    print("[ROI] Icon detection enabled — select icon area...")
                    icon_selector = ROISelector()
                    icon_roi = icon_selector.select(message="框选右上角四叶草铅绘图标")
                    if icon_roi is not None:
                        self.detector.set_icon_roi(icon_roi)
                        self.config["icon_detection"]["icon_roi"] = icon_roi
                        self._save_config()
                        print(f"[ROI] Icon ROI: left={icon_roi['left']}, "
                              f"top={icon_roi['top']}, "
                              f"width={icon_roi['width']}, height={icon_roi['height']}")
                    else:
                        print("[ROI] Icon ROI selection cancelled.")
            else:
                print("[ROI] Selection cancelled.")
        finally:
            self._selecting_roi = False

    def _save_config(self) -> None:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ── result processing (runs on main thread via bridge) ────────────

    def _on_result_main_thread(self, result: CascadeDetectionResult) -> None:
        """Called on main thread via AppBridge signal. Safe to touch overlay."""
        status = result.status

        # Update screenshot preview
        self.result_text.update_screenshot_preview(
            result.sub_roi1_image, result.sub_roi2_image)

        print(f"[Callback] status={status}")

        self._update_debug_boxes(result)

        if status == "matched":
            self.result_text.record_detection()
            self.result_text.show_sampling()
        elif status == "normal_fallback":
            self.result_text.show_normal_fallback()
        elif status == "sampling":
            self.result_text.show_sampling()
        elif status == "no_match":
            self.result_text.show_no_match()

    # ── hotkeys ───────────────────────────────────────────────────────

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
                "ctrl+shift+l",
                lambda: self._bridge.request_toggle_lock.emit(),
                suppress=False)
            keyboard.add_hotkey(
                "ctrl+shift+q",
                lambda: self._bridge.request_quit.emit(),
                suppress=False)
            print("[Hotkeys] Ctrl+Shift+R: re-select ROI | "
                  "Ctrl+Shift+H: toggle panel | Ctrl+Shift+L: lock mouse | "
                  "Ctrl+Shift+Q: quit")
        except Exception as e:
            print(f"[WARN] Could not register hotkeys (need admin?): {e}")

    # ── settings / toggles ────────────────────────────────────────────

    def _on_toggle_debug_save(self, enabled: bool) -> None:
        self.config.setdefault("debug", {})
        self.config["debug"]["save_debug_frames"] = enabled
        self.debug_drawer.debug["save_debug_frames"] = enabled
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

        # Anchor box (blue)
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

        # Sub-ROI2 box (orange)
        if result.sub_roi_box_2:
            b = self._to_screen(rx, ry, result.sub_roi_box_2)
            if b:
                boxes.append((*b, 0, 200, 255, "ROI2"))

        self.debug_box_overlay.update_boxes(boxes)

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
            self.settings_window._load_all()
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


def _set_process_priority():
    """提升当前进程的CPU调度优先级，避免系统高负载下截屏延迟过大。"""
    import ctypes
    from ctypes import wintypes

    HIGH_PRIORITY_CLASS = 0x00000080
    ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000

    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.SetPriorityClass.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()

        if kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS):
            print("[Priority] 进程优先级: HIGH_PRIORITY_CLASS")
        elif kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS):
            print("[Priority] 进程优先级: ABOVE_NORMAL_PRIORITY_CLASS (无需管理员)")
        else:
            err = kernel32.GetLastError()
            priority = kernel32.GetPriorityClass(handle)
            names = {0x40: "IDLE", 0x4000: "BELOW_NORMAL", 0x20: "NORMAL",
                     0x8000: "ABOVE_NORMAL", 0x80: "HIGH", 0x100: "REALTIME"}
            name = names.get(priority, f"未知({priority})")
            print(f"[Priority] 设置失败 (错误码: {err})，当前优先级: {name}")
    except Exception as e:
        print(f"[Priority] 设置进程优先级异常: {e}")


if __name__ == "__main__":
    _set_process_priority()
    app = App()
    app.run()
