"""PyQt5 frameless overlay windows: status bar and result history panel."""

import cv2
import time
import numpy as np
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QPushButton, QScrollArea, QSizeGrip,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap


class OverlaySignals(QObject):
    update_text = pyqtSignal(str)
    update_style = pyqtSignal(str)
    start_match_timer = pyqtSignal()
    stop_match_timer = pyqtSignal()
    start_debounce = pyqtSignal()      # thread-safe debounce trigger
    close_app = pyqtSignal()
    open_settings = pyqtSignal()
    toggle_debug_save = pyqtSignal(bool)
    toggle_preview = pyqtSignal(bool)
    toggle_debug_overlay = pyqtSignal(bool)
    request_re_select = pyqtSignal()


class Overlay(QWidget):
    """Frameless, topmost overlay with customizable duration, position, size, and colors."""

    def __init__(self, config: dict):
        super().__init__()
        self.signals = OverlaySignals()
        ov = config.get("overlay", {})

        self._width = ov.get("width", 680)
        self._height = ov.get("height", 64)
        self._position = ov.get("position", "top")
        self._normal_text = ov.get("normal_text", "● 运行中")
        self._matched_prefix = ov.get("matched_prefix", "识别到：")
        self._match_show_seconds = ov.get("match_show_seconds", 3.0)
        self._sampling_text = ov.get("sampling_text", "正在采样识别...")

        self._bg_color = ov.get("bg_color", "rgba(0, 0, 0, 180)")
        self._text_color = ov.get("text_color", "#ffffff")
        self._matched_text_color = ov.get("matched_text_color", "#00ff88")

        self._debug_saving = config.get("debug", {}).get("save_debug_frames", False)
        self._preview_showing = config.get("debug", {}).get("show_preview_window", False)

        # Timer created in main thread; all start/stop via signal slots
        self._reset_timer = QTimer()
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(self._on_timer_fire)

        # Debounce timer for rapid-fire show_match calls — only last one wins
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._apply_debounced)
        self._pending_text = ""
        self._pending_style = ""

        self._init_ui()

        # Force-topmost timer for fullscreen games
        self._topmost_timer = QTimer()
        self._topmost_timer.setInterval(2000)
        self._topmost_timer.timeout.connect(self._force_topmost)
        self._topmost_timer.start()

        # Wire signals → slots (all slot execution on main thread)
        self.signals.update_text.connect(self._label.setText)
        self.signals.update_style.connect(self._label.setStyleSheet)
        self.signals.start_match_timer.connect(self._start_timer_slot)
        self.signals.stop_match_timer.connect(self._reset_timer.stop)
        self.signals.start_debounce.connect(self._start_debounce_slot)
        self.signals.close_app.connect(self._on_close)

    def _make_stylesheet(self) -> str:
        return (
            f"RocoDetectorOverlay {{"
            f"  background-color: {self._bg_color};"
            f"  border-radius: 8px;"
            f"}}"
            f"QLabel {{"
            f"  color: {self._text_color};"
            f"  font-size: 16px;"
            f"}}"
            f"QPushButton {{"
            f"  color: #888;"
            f"  background: transparent;"
            f"  border: none;"
            f"  font-size: 14px;"
            f"  padding: 0 6px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  color: #fff;"
            f"}}"
        )

    def _init_ui(self):
        self.setWindowTitle("Roco Detector")
        self.setObjectName("RocoDetectorOverlay")
        self.setFixedSize(self._width, self._height)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setStyleSheet(self._make_stylesheet())

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 0, 8, 0)

        self._label = QLabel(self._normal_text)
        self._label.setFont(QFont("Microsoft YaHei", 11))
        self._label.setObjectName("statusLabel")
        layout.addWidget(self._label)
        layout.addStretch()

        self._debug_btn = QPushButton("⬤" if self._debug_saving else "○")
        self._debug_btn.setFixedSize(28, 24)
        self._debug_btn.setToolTip("调试截图保存开关")
        self._debug_btn.clicked.connect(self._toggle_debug_save)
        self._update_debug_btn_style()
        layout.addWidget(self._debug_btn)

        # Preview window toggle
        self._preview_btn = QPushButton("🔍" if self._preview_showing else "◌")
        self._preview_btn.setFixedSize(26, 24)
        self._preview_btn.setToolTip("预览窗口开关")
        self._preview_btn.clicked.connect(self._toggle_preview)
        self._update_preview_btn_style()
        layout.addWidget(self._preview_btn)

        # Debug box overlay toggle
        self._debug_overlay_enabled = False
        self._debug_overlay_btn = QPushButton("▣")
        self._debug_overlay_btn.setFixedSize(26, 24)
        self._debug_overlay_btn.setToolTip("画面覆盖框开关")
        self._debug_overlay_btn.clicked.connect(self._toggle_debug_overlay)
        self._update_debug_overlay_btn_style()
        layout.addWidget(self._debug_overlay_btn)

        # Manual ROI re-select
        reselect_btn = QPushButton("↻")
        reselect_btn.setFixedSize(24, 24)
        reselect_btn.setToolTip("重新框选游戏区域")
        reselect_btn.clicked.connect(lambda: self.signals.request_re_select.emit())
        reselect_btn.setStyleSheet(
            "QPushButton { color: #888; background: transparent; border: none; "
            "font-size: 13px; padding: 0 2px; }"
            "QPushButton:hover { color: #fff; }"
        )
        layout.addWidget(reselect_btn)

        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(24, 24)
        settings_btn.setToolTip("设置面板")
        settings_btn.clicked.connect(lambda: self.signals.open_settings.emit())
        layout.addWidget(settings_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self._on_close)
        layout.addWidget(close_btn)

        self.setLayout(layout)
        self._center_on_screen()

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.geometry()
        if self._position == "top":
            x = (geom.width() - self._width) // 2
            y = 10
        elif self._position == "bottom":
            x = (geom.width() - self._width) // 2
            y = geom.height() - self._height - 40
        else:
            x = (geom.width() - self._width) // 2
            y = (geom.height() - self._height) // 2
        self.move(x, y)

    # ── public API (thread-safe — only emits signals) ──────────────────

    def show_match(self, label: str, score: float, votes=None) -> None:
        """Called from any thread. If votes is None/empty, no vote info is displayed."""
        vote_part = f"  票数 {votes}" if votes else ""
        text = f"{self._matched_prefix}{label} {score:.2f}{vote_part}"
        style = f"QLabel {{ color: {self._matched_text_color}; font-size: 16px; }}"
        self._pending_text = text
        self._pending_style = style
        self.signals.stop_match_timer.emit()
        self.signals.start_debounce.emit()

    def clear_match_state(self) -> None:
        """Reset all display state."""
        self._debounce_timer.stop()
        self._pending_text = ""
        self._pending_style = ""
        self.signals.stop_match_timer.emit()
        self._apply_normal_style()

    def show_sampling(self) -> None:
        """Show sampling-in-progress text immediately (cancels pending debounce)."""
        self._pending_text = ""
        self.signals.stop_match_timer.emit()
        self._debounce_timer.stop()      # safe — only called from main thread via callback
        self.signals.update_text.emit(self._sampling_text)
        self.signals.update_style.emit(
            f"QLabel {{ color: {self._text_color}; font-size: 16px; }}"
        )

    def show_no_match(self) -> None:
        """Show 'not matched' feedback immediately (cancels pending debounce)."""
        self._pending_text = ""
        self.signals.stop_match_timer.emit()
        self._debounce_timer.stop()      # safe — only called from main thread via callback
        self.signals.update_text.emit("未识别到目标标志")
        self.signals.update_style.emit(
            f"QLabel {{ color: {self._text_color}; font-size: 16px; }}"
        )
        self.signals.start_match_timer.emit()

    def _start_debounce_slot(self):
        """Starts the actual debounce QTimer on the main thread."""
        self._debounce_timer.start(300)

    def _apply_debounced(self):
        """Called 300ms after the last show_match — actually commit the display."""
        if self._pending_text:
            self.signals.update_text.emit(self._pending_text)
            self.signals.update_style.emit(self._pending_style)
            self.signals.start_match_timer.emit()
            self._pending_text = ""

    def reset_to_normal(self) -> None:
        """Explicitly reset (used when settings change, etc.)."""
        self.signals.stop_match_timer.emit()
        self._apply_normal_style()

    def _apply_normal_style(self):
        self.signals.update_text.emit(self._normal_text)
        self.signals.update_style.emit(
            f"QLabel {{ color: {self._text_color}; font-size: 16px; }}"
        )

    # ── timer slot (runs on main thread) ───────────────────────────────

    def _start_timer_slot(self):
        """Start (or restart) the single-shot reset timer. Runs on main thread."""
        self._reset_timer.start(int(self._match_show_seconds * 1000))

    def _on_timer_fire(self):
        """Timer expired — no match seen for match_show_seconds."""
        self._apply_normal_style()

    # ── internal ───────────────────────────────────────────────────────

    def _toggle_debug_save(self):
        self._debug_saving = not self._debug_saving
        self._update_debug_btn_style()
        self.signals.toggle_debug_save.emit(self._debug_saving)

    def _toggle_preview(self):
        self._preview_showing = not self._preview_showing
        self._update_preview_btn_style()
        self.signals.toggle_preview.emit(self._preview_showing)

    def _update_preview_btn_style(self):
        if self._preview_showing:
            self._preview_btn.setText("🔍")
            self._preview_btn.setStyleSheet(
                "QPushButton { color: #66aaff; background: transparent; border: none; "
                "font-size: 12px; padding: 0 2px; }"
                "QPushButton:hover { color: #88ccff; }"
            )
        else:
            self._preview_btn.setText("◌")
            self._preview_btn.setStyleSheet(
                "QPushButton { color: #666; background: transparent; border: none; "
                "font-size: 14px; padding: 0 2px; }"
                "QPushButton:hover { color: #aaa; }"
            )

    def _toggle_debug_overlay(self):
        self._debug_overlay_enabled = not self._debug_overlay_enabled
        self._update_debug_overlay_btn_style()
        self.signals.toggle_debug_overlay.emit(self._debug_overlay_enabled)

    def _update_debug_overlay_btn_style(self):
        if self._debug_overlay_enabled:
            self._debug_overlay_btn.setText("▣")
            self._debug_overlay_btn.setStyleSheet(
                "QPushButton { color: #ff8844; background: transparent; border: none; "
                "font-size: 12px; padding: 0 2px; }"
                "QPushButton:hover { color: #ffaa66; }"
            )
        else:
            self._debug_overlay_btn.setText("□")
            self._debug_overlay_btn.setStyleSheet(
                "QPushButton { color: #666; background: transparent; border: none; "
                "font-size: 12px; padding: 0 2px; }"
                "QPushButton:hover { color: #aaa; }"
            )

    def _update_debug_btn_style(self):
        if self._debug_saving:
            self._debug_btn.setText("⬤")
            self._debug_btn.setStyleSheet(
                "QPushButton { color: #e04040; background: transparent; border: none; "
                "font-size: 14px; padding: 0 4px; }"
                "QPushButton:hover { color: #ff6666; }"
            )
        else:
            self._debug_btn.setText("○")
            self._debug_btn.setStyleSheet(
                "QPushButton { color: #666; background: transparent; border: none; "
                "font-size: 14px; padding: 0 4px; }"
                "QPushButton:hover { color: #aaa; }"
            )

    def _on_close(self):
        self.signals.close_app.emit()

    def show_overlay(self):
        self.show()

    def _force_topmost(self):
        """Windows: forcefully keep window on top using SetWindowPos."""
        try:
            import ctypes
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010)
        except Exception:
            pass

    # ── Setters for settings panel ──

    def set_size(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self.setFixedSize(width, height)
        self._center_on_screen()

    def set_position(self, pos: str) -> None:
        self._position = pos
        self._center_on_screen()

    def set_normal_text(self, text: str) -> None:
        self._normal_text = text

    def set_matched_prefix(self, prefix: str) -> None:
        self._matched_prefix = prefix

    def set_match_show_seconds(self, seconds: float) -> None:
        self._match_show_seconds = max(0.5, seconds)

    def set_debug_saving(self, enabled: bool):
        self._debug_saving = enabled
        self._update_debug_btn_style()

    def reload_config(self, cfg: dict) -> None:
        ov = cfg.get("overlay", {})
        self.set_size(ov.get("width", self._width), ov.get("height", self._height))
        self.set_position(ov.get("position", self._position))
        self.set_normal_text(ov.get("normal_text", self._normal_text))
        self.set_matched_prefix(ov.get("matched_prefix", self._matched_prefix))
        self.set_match_show_seconds(ov.get("match_show_seconds", self._match_show_seconds))
        self._sampling_text = ov.get("sampling_text", self._sampling_text)
        self._bg_color = ov.get("bg_color", self._bg_color)
        self._text_color = ov.get("text_color", self._text_color)
        self._matched_text_color = ov.get("matched_text_color", self._matched_text_color)
        self.setStyleSheet(self._make_stylesheet())
        # Reset to normal display
        self.signals.stop_match_timer.emit()
        self._apply_normal_style()


# ── Result History Overlay ───────────────────────────────────────────


def parse_combined_label(label: str):
    """Parse '血脉 + 属性' or '血脉+属性' into (bloodline, attribute_or_None)."""
    label = label.replace("＋", "+")
    if "+" in label:
        parts = label.split("+", 1)
        return parts[0].strip(), parts[1].strip()
    return label.strip(), None


class _ChipLabel(QLabel):
    """Rounded chip for bloodline/attribute display."""
    def __init__(self, text: str, bg_color: QColor, font_size: int = 14, parent=None):
        super().__init__(text, parent)
        r, g, b, a = bg_color.red(), bg_color.green(), bg_color.blue(), bg_color.alpha()
        self.setStyleSheet(
            f"QLabel {{"
            f"  background: rgba({r},{g},{b},{a});"
            f"  color: #111;"
            f"  border-radius: 4px;"
            f"  padding: 2px 8px;"
            f"  font-size: {font_size}px;"
            f"  font-weight: bold;"
            f"}}"
        )


class _HistoryItem(QWidget):
    """A single row: [bloodline chip] + [attribute chip]."""

    def __init__(self, bloodline: str, attribute: str,
                 bl_color: QColor, attr_color: QColor, plus_color: str,
                 chip_font_size: int = 14, plus_font_size: int = 14,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("RTO_history_item")
        self.setStyleSheet(
            "#RTO_history_item { background: rgba(255,255,255,18); border-radius: 8px; }"
            "#RTO_history_item:hover { background: rgba(255,255,255,32); }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        bl_chip = _ChipLabel(bloodline, bl_color, font_size=chip_font_size)
        layout.addWidget(bl_chip)

        plus = QLabel("+")
        plus.setStyleSheet(
            f"color: {plus_color}; font-size: {plus_font_size}px; background: transparent;")
        layout.addWidget(plus)

        if attribute:
            attr_chip = _ChipLabel(attribute, attr_color, font_size=chip_font_size)
            layout.addWidget(attr_chip)
        else:
            unknown = QLabel("?")
            unknown.setStyleSheet(
                f"color: #666; font-size: {chip_font_size}px; background: transparent;")
            layout.addWidget(unknown)

        layout.addStretch()


class _RTOSignals(QObject):
    add_result = pyqtSignal(str)
    clear_results = pyqtSignal()
    position_changed = pyqtSignal(int, int)
    size_changed = pyqtSignal(int, int)
    open_settings = pyqtSignal()
    request_re_select = pyqtSignal()
    request_quit = pyqtSignal()
    toggle_debug_save = pyqtSignal(bool)
    toggle_preview = pyqtSignal(bool)
    toggle_debug_overlay = pyqtSignal(bool)
    set_status_text = pyqtSignal(str)
    screenshot_mode_changed = pyqtSignal(bool)


def _parse_rgba(s: str) -> QColor:
    s = s.strip()
    if s.startswith("rgba("):
        parts = s[5:-1].split(",")
        if len(parts) == 4:
            return QColor(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    return QColor(s)


class ResultTextOverlay(QWidget):
    """Recognition history panel with 3-zone layout.

    Zones:
      HeaderBar   — title, status, action buttons
      HistoryArea — scrollable chip-style [血脉] + [属性] items
      CounterArea — bloodline counts | attribute counts
    """

    _HEADER_H = 42

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._signals = _RTOSignals()
        cfg = config.get("result_text_overlay", {})

        # Timers
        self._topmost_timer = QTimer()
        self._topmost_timer.setInterval(2000)
        self._topmost_timer.timeout.connect(self._force_topmost)
        self._topmost_timer.start()

        self._alt_timer = QTimer()
        self._alt_timer.setInterval(100)
        self._alt_timer.timeout.connect(self._check_alt_key)
        self._alt_timer.start()

        self._cooldown_duration = self.config.get("runtime", {}).get(
            "sequence_cooldown_seconds", 1.5)
        self._cooldown_until = 0.0
        self._cooldown_timer = QTimer()
        self._cooldown_timer.setInterval(100)
        self._cooldown_timer.timeout.connect(self._update_cooldown)

        # Config
        self._enabled = cfg.get("enabled", True)
        self._font_family = cfg.get("font_family", "Microsoft YaHei")
        self._header_font_size = cfg.get("header_font_size", 12)
        self._status_font_size = cfg.get("status_font_size", 11)
        self._chip_font_size = cfg.get("chip_font_size", 14)
        self._counter_font_size = cfg.get("counter_font_size", 11)
        self._counter_title_font_size = cfg.get("counter_title_font_size", 10)
        self._default_color = QColor(cfg.get("default_color", "#FFD700"))
        self._label_colors: dict = cfg.get("label_colors", {})
        self._bloodline_colors: dict = cfg.get("bloodline_colors", {})
        self._attribute_colors: dict = cfg.get("attribute_colors", {})
        self._plus_color = cfg.get("plus_color", "#DDDDDD")
        self._count_color = cfg.get("count_color", "#AAAAAA")
        self._max_items = cfg.get("max_items", 104)
        self._show_counter = cfg.get("show_counter_area", True)
        self._counter_top_n = cfg.get("counter_top_n", 5)

        self._bg_color = _parse_rgba(cfg.get("background_color", "rgba(8,10,16,190)"))
        self._border_color = _parse_rgba(cfg.get("border_color", "rgba(255,255,255,45)"))
        self._border_radius = cfg.get("border_radius", 12)
        self._padding = cfg.get("padding", 10)
        self._title_text = cfg.get("title", "识别记录")
        self._scrollbar_width = cfg.get("scrollbar_width", 7)
        self._min_w = cfg.get("min_width", 220)
        self._min_h = cfg.get("min_height", 160)

        # Position / size — defaults on primary screen, validates across all screens
        screen = QApplication.primaryScreen()
        geom = screen.geometry() if screen else None
        if geom:
            def_w = max(220, geom.width() // 4)
            def_h = 240
            def_x = max(0, (geom.width() - def_w) // 2)
            def_y = max(0, geom.height() - def_h - 30)
        else:
            def_w, def_h, def_x, def_y = 360, 240, 800, 500
        self._width = cfg.get("width", def_w)
        self._height = cfg.get("height", def_h)
        config_x = cfg.get("x", def_x)
        config_y = cfg.get("y", def_y)
        if not self._pos_on_any_screen(config_x, config_y):
            self._pos_x = def_x
            self._pos_y = def_y
        else:
            self._pos_x = config_x
            self._pos_y = config_y

        # State — default: screenshot mode
        self._records: list = []
        self._history_items: list = []
        self._bloodline_counts: dict = {}
        self._attribute_counts: dict = {}
        self._dragging = False
        self._drag_start = QPoint()
        self._status_screenshot_on = True  # start in screenshot mode
        self._mouse_locked = False

        self.setCursor(Qt.ArrowCursor)

        self._build_ui()
        self.resize(self._width, self._height)
        self.move(self._pos_x, self._pos_y)

        # Signal wiring
        self._signals.add_result.connect(self._do_add)
        self._signals.clear_results.connect(self._do_clear)
        self._signals.set_status_text.connect(self._do_set_status)

        if self._status_screenshot_on:
            self._status_lbl.setText("📷 截图模式")
            self._status_lbl.setStyleSheet(
                f"color: #44dd88; font-size: {self._status_font_size}px; background: transparent;")

        if self._enabled:
            QWidget.show(self)
            QApplication.processEvents()  # force layout so widgets have valid sizes

        # Belt-and-suspenders: re-apply default screenshot mode after show
        if self._status_screenshot_on:
            self._scroll.hide()
            self._screenshot_preview.show()
            self._mode_btn.setText("📷 截图模式")
            self._mode_btn.setStyleSheet(
                "QPushButton { color: #44dd88; background: rgba(68,221,136,12); "
                "border: 1px solid rgba(68,221,136,40); border-radius: 6px; "
                "font-size: 15px; padding: 4px 12px; font-weight: bold; }"
                "QPushButton:hover { background: rgba(68,221,136,25); "
                "border-color: rgba(68,221,136,80); }")
            self._refresh_screenshot_counter()

    # ── build ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("ResultHistory")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setMinimumSize(self._min_w, self._min_h)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── panel ──
        self._panel = QWidget()
        self._panel.setObjectName("RTO_panel")
        self._panel.setAttribute(Qt.WA_StyledBackground, True)
        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # ── header bar ──
        self._header_bar = QWidget()
        self._header_bar.setObjectName("RTO_header")
        self._header_bar.setFixedHeight(self._HEADER_H)
        hl = QHBoxLayout(self._header_bar)
        hl.setContentsMargins(self._padding, 4, self._padding, 2)
        hl.setSpacing(4)

        self._title_lbl = QLabel(self._title_text)
        self._title_lbl.setStyleSheet(
            f"color: #ccc; font-size: {self._header_font_size}px; "
            "background: transparent; font-weight: bold;")
        hl.addWidget(self._title_lbl)

        self._status_lbl = QLabel("● 运行中")
        self._status_lbl.setStyleSheet(
            f"color: #aaa; font-size: {self._status_font_size}px; "
            "background: transparent;")
        hl.addWidget(self._status_lbl)

        self._lock_lbl = QLabel("")
        self._lock_lbl.setStyleSheet(
            "color: #e04040; font-size: 11px; background: transparent; font-weight: bold;")
        hl.addWidget(self._lock_lbl)

        hl.addStretch()

        # Preview toggle
        self._status_preview_on = False
        self._preview_btn = self._make_header_btn("◌ 预览", "预览窗口开关")
        self._preview_btn.clicked.connect(self._toggle_status_preview)
        hl.addWidget(self._preview_btn)

        # Debug save toggle
        self._status_debug_on = False
        self._debug_btn = self._make_header_btn("○ 存图", "调试截图开关")
        self._debug_btn.clicked.connect(self._toggle_status_debug)
        hl.addWidget(self._debug_btn)

        # Debug overlay toggle
        self._status_overlay_on = False
        self._overlay_btn = self._make_header_btn("□ 画框", "画面覆盖框开关")
        self._overlay_btn.clicked.connect(self._toggle_status_overlay)
        hl.addWidget(self._overlay_btn)

        # Re-select ROI
        reselect = self._make_header_btn("↻ 重选", "重新框选区域")
        reselect.clicked.connect(lambda: self._signals.request_re_select.emit())
        hl.addWidget(reselect)

        # Settings
        gear = self._make_header_btn("⚙ 设置", "设置面板")
        gear.clicked.connect(lambda: self._signals.open_settings.emit())
        hl.addWidget(gear)

        # Minimize panel to taskbar
        close_btn = self._make_header_btn("✕ 隐藏", "最小化到任务栏")
        close_btn.clicked.connect(lambda: self.showMinimized())
        hl.addWidget(close_btn)

        # Quit app
        quit_btn = self._make_header_btn("⏻ 退出", "退出程序")
        quit_btn.setStyleSheet(
            "QPushButton { color: #e04040; background: transparent; border: none; "
            "font-size: 14px; padding: 2px 6px; }"
            "QPushButton:hover { color: #ff6666; background: rgba(255,64,64,30); "
            "border-radius: 4px; }")
        quit_btn.clicked.connect(lambda: self._signals.request_quit.emit())
        hl.addWidget(quit_btn)

        # Mode toggle (larger, prominent)
        self._mode_btn = QPushButton("🔍 识图模式")
        self._mode_btn.setMinimumHeight(36)
        self._mode_btn.setToolTip("切换识图/截图模式")
        self._mode_btn.clicked.connect(self._toggle_screenshot_mode)
        self._mode_btn.setStyleSheet(
            "QPushButton { color: #66aaff; background: rgba(102,170,255,12); "
            "border: 1px solid rgba(102,170,255,40); border-radius: 6px; "
            "font-size: 15px; padding: 4px 12px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(102,170,255,25); "
            "border-color: rgba(102,170,255,80); }")
        hl.addWidget(self._mode_btn)

        panel_layout.addWidget(self._header_bar)

        # ── scroll area (history) ──
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setFrameShape(QScrollArea.NoFrame)

        self._content = QWidget()
        self._content.setObjectName("RTO_content")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(self._padding, 6, self._padding, 4)
        self._content_layout.setSpacing(4)
        self._content_layout.addStretch()
        self._scroll.setWidget(self._content)

        panel_layout.addWidget(self._scroll, 1)

        # ── screenshot preview (hidden, replaces scroll when on) ──
        self._screenshot_preview = QWidget()
        self._screenshot_preview.setObjectName("RTO_screenshot")
        ss_layout = QVBoxLayout(self._screenshot_preview)
        ss_layout.setContentsMargins(4, 4, 4, 4)
        ss_layout.setSpacing(6)

        ss_row = QHBoxLayout()
        ss_row.setSpacing(6)

        self._ss_img1 = QLabel("等待截图...")
        self._ss_img1.setAlignment(Qt.AlignCenter)
        self._ss_img1.setStyleSheet(
            "color: #666; font-size: 10px; background: rgba(0,0,0,60); "
            "border-radius: 4px; padding: 4px;")
        self._ss_img1.setMinimumSize(160, 100)
        ss_row.addWidget(self._ss_img1, 1)

        self._ss_img2 = QLabel("")
        self._ss_img2.setAlignment(Qt.AlignCenter)
        self._ss_img2.setStyleSheet(
            "color: #666; font-size: 10px; background: rgba(0,0,0,60); "
            "border-radius: 4px; padding: 4px;")
        self._ss_img2.setMinimumSize(160, 100)
        ss_row.addWidget(self._ss_img2, 1)

        ss_layout.addLayout(ss_row)
        self._screenshot_preview.hide()
        panel_layout.addWidget(self._screenshot_preview, 1)

        # ── counter area ──
        self._counter_area = QWidget()
        self._counter_area.setObjectName("RTO_counter")
        counter_layout = QHBoxLayout(self._counter_area)
        counter_layout.setContentsMargins(self._padding, 6, self._padding, 6)
        counter_layout.setSpacing(12)

        self._bl_counter = QLabel("")
        self._bl_counter.setObjectName("RTO_bl_counter")
        self._bl_counter.setStyleSheet(
            f"color: #aaa; font-size: {self._counter_font_size}px; background: transparent;")
        self._bl_counter.setWordWrap(True)
        self._bl_counter.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        counter_layout.addWidget(self._bl_counter, 1)

        self._attr_counter = QLabel("")
        self._attr_counter.setObjectName("RTO_attr_counter")
        self._attr_counter.setStyleSheet(
            f"color: #aaa; font-size: {self._counter_font_size}px; background: transparent;")
        self._attr_counter.setWordWrap(True)
        self._attr_counter.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        counter_layout.addWidget(self._attr_counter, 1)

        # Cooldown countdown
        self._cooldown_label = QLabel("")
        self._cooldown_label.setObjectName("RTO_cooldown")
        self._cooldown_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._cooldown_label.setStyleSheet(
            f"color: #888; font-size: {self._counter_font_size}px; "
            "background: transparent;")
        counter_layout.addWidget(self._cooldown_label)

        # Clear button in counter bar
        self._counter_clear_btn = QPushButton("清空")
        self._counter_clear_btn.setMinimumHeight(22)
        self._counter_clear_btn.setToolTip("清空识别记录")
        self._counter_clear_btn.clicked.connect(self._do_clear)
        self._counter_clear_btn.setStyleSheet(
            "QPushButton { color: #888; background: transparent; border: none; "
            "font-size: 12px; padding: 2px 6px; }"
            "QPushButton:hover { color: #fff; background: rgba(255,255,255,18); "
            "border-radius: 4px; }")
        counter_layout.addWidget(self._counter_clear_btn)

        panel_layout.addWidget(self._counter_area)

        if not self._show_counter:
            self._counter_area.hide()

        outer.addWidget(self._panel)

        # ── size grip ──
        self._size_grip = QSizeGrip(self)
        self._size_grip.setFixedSize(16, 16)
        self._size_grip.setStyleSheet("QSizeGrip { background: transparent; }")

        # ── stylesheets ──
        self._panel.setStyleSheet(self._panel_style())
        self._scroll.setStyleSheet(self._scroll_style())
        self._header_bar.setStyleSheet(self._header_style())
        self._counter_area.setStyleSheet(self._counter_style())

        # If starting in screenshot mode, apply initial state now
        if self._status_screenshot_on:
            self._scroll.hide()
            self._screenshot_preview.show()
            self._mode_btn.setText("📷 截图模式")
            self._mode_btn.setStyleSheet(
                "QPushButton { color: #44dd88; background: rgba(68,221,136,12); "
                "border: 1px solid rgba(68,221,136,40); border-radius: 6px; "
                "font-size: 15px; padding: 4px 12px; font-weight: bold; }"
                "QPushButton:hover { background: rgba(68,221,136,25); "
                "border-color: rgba(68,221,136,80); }")
            self._refresh_screenshot_counter()

        self._panel.repaint()

    def _make_header_btn(self, text: str, tooltip: str):
        btn = QPushButton(text)
        btn.setMinimumHeight(28)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(
            "QPushButton { color: #888; background: transparent; border: none; "
            "font-size: 14px; padding: 2px 6px; }"
            "QPushButton:hover { color: #fff; background: rgba(255,255,255,18); "
            "border-radius: 4px; }"
        )
        return btn

    # ── styles ────────────────────────────────────────────────────────

    def _panel_style(self) -> str:
        bg = self._bg_color
        br = self._border_radius
        bc = self._border_color
        return (
            f"#RTO_panel {{"
            f"  background-color: rgba({bg.red()},{bg.green()},{bg.blue()},{bg.alpha()});"
            f"  border: 1px solid "
            f"rgba({bc.red()},{bc.green()},{bc.blue()},{bc.alpha()});"
            f"  border-radius: {br}px;"
            f"}}"
        )

    def _header_style(self) -> str:
        return (
            "#RTO_header {"
            "  background: transparent;"
            "  border-bottom: 1px solid rgba(255,255,255,35);"
            "}"
        )

    def _scroll_style(self) -> str:
        w = self._scrollbar_width
        return (
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollArea > QWidget > QWidget {{ background: transparent; }}"
            f"QScrollBar:vertical {{"
            f"  background: transparent; width: {w}px; margin: 4px 2px 4px 2px;"
            f"}}"
            f"QScrollBar::handle:vertical {{"
            f"  background: rgba(255,255,255,50); border-radius: {w // 2}px;"
            f"  min-height: 30px;"
            f"}}"
            f"QScrollBar::handle:vertical:hover {{"
            f"  background: rgba(255,255,255,150);"
            f"}}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{"
            f"  height: 0px;"
            f"}}"
            f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{"
            f"  background: transparent;"
            f"}}"
        )

    def _counter_style(self) -> str:
        return (
            "#RTO_counter {"
            "  background: rgba(0,0,0,60);"
            "  border-top: 1px solid rgba(255,255,255,30);"
            "}"
        )

    # ── window management ─────────────────────────────────────────────

    def _force_topmost(self):
        try:
            import ctypes
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010)
        except Exception:
            pass

    def toggle_mouse_lock(self):
        self._mouse_locked = not self._mouse_locked
        if self._mouse_locked:
            self._lock_lbl.setText("锁定: 开")
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.setCursor(Qt.BlankCursor)
            # Disable all header buttons
            for btn in self._header_bar.findChildren(QPushButton):
                btn.setEnabled(False)
            self._counter_clear_btn.setEnabled(False)
        else:
            self._lock_lbl.setText("")
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.setCursor(Qt.ArrowCursor)
            for btn in self._header_bar.findChildren(QPushButton):
                btn.setEnabled(True)
            self._counter_clear_btn.setEnabled(True)

    def _check_alt_key(self):
        """Alt held → interactive; released → click-through.
        Mouse lock overrides: always click-through + hidden cursor when locked."""
        if self._mouse_locked:
            if not self.testAttribute(Qt.WA_TransparentForMouseEvents):
                self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                self.setCursor(Qt.BlankCursor)
                self._panel.setStyleSheet(self._panel_style())
            return
        try:
            import ctypes
            alt_down = ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000
            if alt_down and self.isVisible():
                if self.testAttribute(Qt.WA_TransparentForMouseEvents):
                    self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
                    self.setCursor(Qt.ArrowCursor)
                    self._panel.setStyleSheet(self._panel_style().replace(
                        "border: 1px solid",
                        "border: 2px solid rgba(255,255,255,180);\n  border: 1px solid"))
                    self._panel.repaint()
            elif not alt_down:
                if not self.testAttribute(Qt.WA_TransparentForMouseEvents):
                    self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                    self.setCursor(Qt.ArrowCursor)
                    self._panel.setStyleSheet(self._panel_style())
        except Exception:
            pass

    # ── header toggles ────────────────────────────────────────────────

    def _toggle_status_debug(self):
        self._status_debug_on = not self._status_debug_on
        self._debug_btn.setText("⬤ 存图" if self._status_debug_on else "○ 存图")
        self._debug_btn.setStyleSheet(
            f"QPushButton {{ color: {'#e04040' if self._status_debug_on else '#888'}; "
            "background: transparent; border: none; font-size: 14px; padding: 2px 6px; }"
            "QPushButton:hover { color: #fff; background: rgba(255,255,255,18); "
            "border-radius: 4px; }")
        self._signals.toggle_debug_save.emit(self._status_debug_on)

    def _toggle_status_preview(self):
        self._status_preview_on = not self._status_preview_on
        self._preview_btn.setText("🔍 预览" if self._status_preview_on else "◌ 预览")
        self._preview_btn.setStyleSheet(
            f"QPushButton {{ color: {'#66aaff' if self._status_preview_on else '#888'}; "
            "background: transparent; border: none; font-size: 14px; padding: 2px 6px; }"
            "QPushButton:hover { color: #fff; background: rgba(255,255,255,18); "
            "border-radius: 4px; }")
        self._signals.toggle_preview.emit(self._status_preview_on)

    def _toggle_status_overlay(self):
        self._status_overlay_on = not self._status_overlay_on
        self._overlay_btn.setText("▣ 画框" if self._status_overlay_on else "□ 画框")
        self._overlay_btn.setStyleSheet(
            f"QPushButton {{ color: {'#ff8844' if self._status_overlay_on else '#888'}; "
            "background: transparent; border: none; font-size: 14px; padding: 2px 6px; }"
            "QPushButton:hover { color: #fff; background: rgba(255,255,255,18); "
            "border-radius: 4px; }")
        self._signals.toggle_debug_overlay.emit(self._status_overlay_on)

    def _toggle_screenshot_mode(self):
        self._status_screenshot_on = not self._status_screenshot_on
        self._signals.screenshot_mode_changed.emit(self._status_screenshot_on)
        if self._status_screenshot_on:
            self._mode_btn.setText("📷 截图模式")
            self._mode_btn.setStyleSheet(
                "QPushButton { color: #44dd88; background: rgba(68,221,136,12); "
                "border: 1px solid rgba(68,221,136,40); border-radius: 6px; "
                "font-size: 15px; padding: 4px 12px; font-weight: bold; }"
                "QPushButton:hover { background: rgba(68,221,136,25); "
                "border-color: rgba(68,221,136,80); }")
            # Swap: hide history, show screenshot preview
            self._scroll.hide()
            self._screenshot_preview.show()
            self._refresh_screenshot_counter()
        else:
            self._mode_btn.setText("🔍 识图模式")
            self._mode_btn.setStyleSheet(
                "QPushButton { color: #66aaff; background: rgba(102,170,255,12); "
                "border: 1px solid rgba(102,170,255,40); border-radius: 6px; "
                "font-size: 15px; padding: 4px 12px; font-weight: bold; }"
                "QPushButton:hover { background: rgba(102,170,255,25); "
                "border-color: rgba(102,170,255,80); }")
            # Swap: show history, hide screenshot preview
            self._scroll.show()
            self._screenshot_preview.hide()
            self._refresh_counter_area()

    def _refresh_screenshot_counter(self):
        """Show simplified counter: total box count on one line."""
        total = len(self._records)
        if total > 0:
            self._bl_counter.setText(
                f"<html><body style='margin:0;padding:0'>"
                f"<span style='color:#aaa;font-size:{self._counter_title_font_size}px;'>捕捉次数</span> "
                f"<span style='color:#ffaa00;font-size:{self._counter_font_size}px;'>x {total}</span>"
                f"</body></html>")
        else:
            self._bl_counter.setText("")
        self._attr_counter.setText("")

    def update_screenshot_preview(self, img1: Optional[np.ndarray],
                                   img2: Optional[np.ndarray]):
        """Update the screenshot preview with latest sub-ROI captures.
        Always processes images regardless of mode, so they are ready on toggle.
        In screenshot mode, counts each valid capture as one box detection."""
        if img1 is not None and img1.size > 0:
            self._ss_ref1 = img1.copy()
        else:
            self._ss_ref1 = None
        if img2 is not None and img2.size > 0:
            self._ss_ref2 = img2.copy()
        else:
            self._ss_ref2 = None

        # Use actual widget size, with fallback for pre-layout startup
        max_w = max(200, self._ss_img1.width() or self.width() * 2 // 3)
        max_h = max(130, self._ss_img1.height() or self.height() * 2 // 3)
        if img1 is not None and img1.size > 0:
            h, w = img1.shape[:2]
            if img1.ndim == 3:
                rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                fmt = QImage.Format_RGB888
                bpl = rgb.strides[0]
            else:
                rgb = img1
                fmt = QImage.Format_Grayscale8
                bpl = w
            qimg = QImage(rgb.data, w, h, bpl, fmt)
            self._ss_img1.setPixmap(QPixmap.fromImage(qimg).scaled(
                max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if img2 is not None and img2.size > 0:
            h, w = img2.shape[:2]
            if img2.ndim == 3:
                rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                fmt = QImage.Format_RGB888
                bpl = rgb.strides[0]
            else:
                rgb = img2
                fmt = QImage.Format_Grayscale8
                bpl = w
            qimg = QImage(rgb.data, w, h, bpl, fmt)
            self._ss_img2.setPixmap(QPixmap.fromImage(qimg).scaled(
                max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ── public API ────────────────────────────────────────────────────

    def add_result(self, label: str) -> None:
        if not self._enabled:
            return
        self._signals.add_result.emit(label)

    def clear_results(self) -> None:
        self._signals.clear_results.emit()

    def toggle_visibility(self):
        if self.isMinimized():
            self.showNormal()
        elif self.isVisible():
            self.showMinimized()
        else:
            self.show()

    def set_status_text(self, text: str) -> None:
        self._signals.set_status_text.emit(text)

    def show_sampling(self):
        self._start_cooldown()
        if self._status_screenshot_on:
            self._status_lbl.setText("📷 截图模式")
            self._status_lbl.setStyleSheet(
                f"color: #44dd88; font-size: {self._status_font_size}px; background: transparent;")
            return
        self.set_status_text("正在采样识别...")
        self._status_lbl.setStyleSheet(
            f"color: #66aaff; font-size: {self._status_font_size}px; background: transparent;")

    def show_match(self, text: str):
        self._start_cooldown()
        if self._status_screenshot_on:
            self._status_lbl.setText("📷 截图模式")
            self._status_lbl.setStyleSheet(
                f"color: #44dd88; font-size: {self._status_font_size}px; background: transparent;")
            return
        self.set_status_text(f"识别到：{text}")
        self._status_lbl.setStyleSheet(
            f"color: #00954f; font-size: {self._status_font_size}px; background: transparent;")

    def show_no_match(self):
        self._start_cooldown()
        if self._status_screenshot_on:
            dummy_label = f"screenshot_{len(self._records)}"
            self._records.append(dummy_label)
            if len(self._records) > self._max_items:
                self._records.pop(0)
            self._refresh_screenshot_counter()
            self._status_lbl.setText("📷 截图模式")
            self._status_lbl.setStyleSheet(
                f"color: #44dd88; font-size: {self._status_font_size}px; "
                "background: transparent;")
            return
        self.set_status_text("未识别到目标")
        self._status_lbl.setStyleSheet(
            f"color: #aaa; font-size: {self._status_font_size}px; background: transparent;")

    def _start_cooldown(self):
        self._cooldown_until = time.time() + self._cooldown_duration
        if not self._cooldown_timer.isActive():
            self._cooldown_timer.start()

    def _update_cooldown(self):
        remaining = self._cooldown_until - time.time()
        if remaining <= 0:
            self._cooldown_label.setText("")
            self._cooldown_timer.stop()
        else:
            self._cooldown_label.setText(f"冷却 {remaining:.1f}s")

    # ── slots ─────────────────────────────────────────────────────────

    def _do_set_status(self, text: str):
        self._status_lbl.setText(text)

    def _do_add(self, label: str):
        # Screenshot mode: counting handled by update_screenshot_preview
        if self._status_screenshot_on:
            return

        bloodline, attribute = parse_combined_label(label)

        self._records.append(label)
        if len(self._records) > self._max_items:
            self._records.pop(0)
            if self._history_items:
                w = self._history_items.pop(0)
                self._content_layout.removeWidget(w)
                w.deleteLater()

        # Update counts
        self._bloodline_counts[bloodline] = self._bloodline_counts.get(bloodline, 0) + 1
        if attribute:
            self._attribute_counts[attribute] = self._attribute_counts.get(attribute, 0) + 1

        bl_color = self._resolve_color(bloodline, is_bloodline=True)
        attr_color = self._resolve_color(attribute, is_bloodline=False) if attribute else self._default_color

        item = _HistoryItem(bloodline, attribute, bl_color, attr_color,
                           self._plus_color, self._chip_font_size, self._chip_font_size)
        self._history_items.append(item)
        self._content_layout.insertWidget(
            self._content_layout.count() - 1, item)

        self._refresh_counter_area()
        QWidget.show(self)
        QTimer.singleShot(20, self._scroll_to_bottom)

    def _do_clear(self):
        self._records.clear()
        for w in self._history_items:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._history_items.clear()
        self._bloodline_counts.clear()
        self._attribute_counts.clear()
        if self._status_screenshot_on:
            self._refresh_screenshot_counter()
        else:
            self._refresh_counter_area()

    # ── counter ───────────────────────────────────────────────────────

    def _refresh_counter_area(self):
        if self._status_screenshot_on:
            self._refresh_screenshot_counter()
            return
        if not self._show_counter:
            return

        top_n = self._counter_top_n

        # Bloodline column
        bl_sorted = sorted(
            self._bloodline_counts.items(), key=lambda x: -x[1])[:top_n]
        if bl_sorted:
            bl_rows = [f"<span style='color:#aaa;font-size:{self._counter_title_font_size}px;'>血脉统计</span>"]
            for name, cnt in bl_sorted:
                c = self._resolve_color(name, is_bloodline=True)
                bl_rows.append(
                    f"<span style='color:{c.name()}'>{name}</span>"
                    f"<span style='color:{self._count_color}'> ×{cnt}</span>")
            self._bl_counter.setText(
                "<html><body style='margin:0;padding:0'>"
                + "<br>".join(bl_rows) + "</body></html>")
        else:
            self._bl_counter.setText("")

        # Attribute column
        attr_sorted = sorted(
            self._attribute_counts.items(), key=lambda x: -x[1])[:top_n]
        if attr_sorted:
            attr_rows = [f"<span style='color:#aaa;font-size:{self._counter_title_font_size}px;'>属性统计</span>"]
            for name, cnt in attr_sorted:
                c = self._resolve_color(name, is_bloodline=False)
                attr_rows.append(
                    f"<span style='color:{c.name()}'>{name}</span>"
                    f"<span style='color:{self._count_color}'> ×{cnt}</span>")
            self._attr_counter.setText(
                "<html><body style='margin:0;padding:0'>"
                + "<br>".join(attr_rows) + "</body></html>")
        else:
            self._attr_counter.setText("")

    # ── color resolution ──────────────────────────────────────────────

    def _resolve_color(self, label: str, is_bloodline: bool = True) -> QColor:
        if not label:
            return self._default_color
        key = "bloodline_colors" if is_bloodline else "attribute_colors"
        colors = self.config.get("result_text_overlay", {}).get(key, {})
        if label in colors:
            return QColor(colors[label])
        if label in self._label_colors:
            return QColor(self._label_colors[label])
        return self._default_color

    # ── drag (on header bar only) ─────────────────────────────────────

    def _in_header(self, y: int) -> bool:
        return y <= self._HEADER_H

    def _alt_held(self) -> bool:
        if self._mouse_locked:
            return False
        try:
            import ctypes
            return bool(ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000)
        except Exception:
            return False

    def _current_screen(self):
        """Return the QScreen containing this window's center, or primary."""
        cx = self.x() + self.width() // 2
        cy = self.y() + self.height() // 2
        for s in QApplication.screens():
            if s.geometry().contains(cx, cy):
                return s
        return QApplication.primaryScreen()

    @staticmethod
    def _pos_on_any_screen(x: int, y: int) -> bool:
        """Check if (x, y) falls within any connected screen's geometry."""
        for s in QApplication.screens():
            g = s.geometry()
            if g.x() - 200 <= x <= g.x() + g.width() + 200 and \
               g.y() - 200 <= y <= g.y() + g.height() + 200:
                return True
        return False

    def mousePressEvent(self, event):
        if not self._alt_held():
            event.ignore()
            return
        if event.button() == Qt.LeftButton and self._in_header(event.y()):
            self._dragging = True
            self._drag_start = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._alt_held():
            event.ignore()
            return
        if self._dragging:
            new_pos = event.globalPos() - self._drag_start
            self.move(new_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._alt_held():
            event.ignore()
            return
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._pos_x = self.x()
            self._pos_y = self.y()
            self._signals.position_changed.emit(self._pos_x, self._pos_y)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._width = self.width()
        self._height = self.height()
        if hasattr(self, '_size_grip'):
            self._size_grip.move(
                self.width() - self._size_grip.width() - 2,
                self.height() - self._size_grip.height() - 2)
        self._signals.size_changed.emit(self._width, self._height)

    def _scroll_to_bottom(self):
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── config reload ─────────────────────────────────────────────────

    def reload_config(self, config: dict):
        self.config = config
        cfg = config.get("result_text_overlay", {})
        self._enabled = cfg.get("enabled", self._enabled)
        self._font_family = cfg.get("font_family", self._font_family)
        self._header_font_size = cfg.get("header_font_size", self._header_font_size)
        self._status_font_size = cfg.get("status_font_size", self._status_font_size)
        self._chip_font_size = cfg.get("chip_font_size", self._chip_font_size)
        self._counter_font_size = cfg.get("counter_font_size", self._counter_font_size)
        self._counter_title_font_size = cfg.get("counter_title_font_size", self._counter_title_font_size)
        self._default_color = QColor(cfg.get("default_color", self._default_color.name()))
        self._label_colors = cfg.get("label_colors", self._label_colors)
        self._bloodline_colors = cfg.get("bloodline_colors", self._bloodline_colors)
        self._attribute_colors = cfg.get("attribute_colors", self._attribute_colors)
        self._plus_color = cfg.get("plus_color", self._plus_color)
        self._count_color = cfg.get("count_color", self._count_color)
        self._max_items = cfg.get("max_items", self._max_items)
        self._show_counter = cfg.get("show_counter_area", self._show_counter)
        self._counter_top_n = cfg.get("counter_top_n", self._counter_top_n)
        self._cooldown_duration = self.config.get("runtime", {}).get(
            "sequence_cooldown_seconds", self._cooldown_duration)
        self._bg_color = _parse_rgba(cfg.get("background_color",
            f"rgba({self._bg_color.red()},{self._bg_color.green()},"
            f"{self._bg_color.blue()},{self._bg_color.alpha()})"))
        self._border_color = _parse_rgba(cfg.get("border_color",
            f"rgba({self._border_color.red()},{self._border_color.green()},"
            f"{self._border_color.blue()},{self._border_color.alpha()})"))
        self._border_radius = cfg.get("border_radius", self._border_radius)
        self._padding = cfg.get("padding", self._padding)
        self._title_text = cfg.get("title", self._title_text)
        self._scrollbar_width = cfg.get("scrollbar_width", self._scrollbar_width)
        self._min_w = cfg.get("min_width", self._min_w)
        self._min_h = cfg.get("min_height", self._min_h)
        w = cfg.get("width", self._width)
        h = cfg.get("height", self._height)
        self._width, self._height = w, h
        self.resize(w, h)
        self.setMinimumSize(self._min_w, self._min_h)
        self._title_lbl.setText(self._title_text)
        self._title_lbl.setStyleSheet(
            f"color: #ccc; font-size: {self._header_font_size}px; "
            "background: transparent; font-weight: bold;")
        self._status_lbl.setStyleSheet(
            f"color: #aaa; font-size: {self._status_font_size}px; "
            "background: transparent;")
        self._bl_counter.setStyleSheet(
            f"color: #aaa; font-size: {self._counter_font_size}px; background: transparent;")
        self._attr_counter.setStyleSheet(
            f"color: #aaa; font-size: {self._counter_font_size}px; background: transparent;")
        self._panel.setStyleSheet(self._panel_style())
        self._scroll.setStyleSheet(self._scroll_style())
        self._header_bar.setStyleSheet(self._header_style())
        self._counter_area.setStyleSheet(self._counter_style())
        if self._show_counter:
            self._counter_area.show()
        else:
            self._counter_area.hide()
        # Rebuild all history items with new color config
        old_items = self._history_items[:]
        self._history_items.clear()
        for rec_label in self._records:
            bloodline, attribute = parse_combined_label(rec_label)
            bl_color = self._resolve_color(bloodline, is_bloodline=True)
            attr_color = self._resolve_color(attribute, is_bloodline=False) if attribute else self._default_color
            new_item = _HistoryItem(bloodline, attribute, bl_color, attr_color,
                                   self._plus_color, self._chip_font_size,
                                   self._chip_font_size)
            self._history_items.append(new_item)
            self._content_layout.insertWidget(
                self._content_layout.count() - 1, new_item)
        for w in old_items:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._refresh_counter_area()


# ── Debug Box Overlay ──────────────────────────────────────────────────

from PyQt5.QtGui import QPainter, QPen


class DebugBoxOverlay(QWidget):
    """Transparent fullscreen overlay that paints detection boxes directly over the game.

    - Always-on-top, frameless, click-through.
    - Draggable: hold the overlay to reposition it.
    - Updated from main thread via update_boxes().
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BoxOverlay")
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # click-through

        screen = QApplication.primaryScreen()
        geom = screen.geometry() if screen else None
        if geom:
            self.setGeometry(geom)
        else:
            self.resize(1920, 1080)

        self.boxes = []  # list of (x, y, w, h, r, g, b, label)

        # Force-topmost timer for fullscreen games
        self._topmost_timer = QTimer()
        self._topmost_timer.setInterval(1500)
        self._topmost_timer.timeout.connect(self._force_topmost)
        self._topmost_timer.start()

        # Drag state
        self._dragging = False
        self._drag_start = QPoint()

    def _force_topmost(self):
        """Windows: forcefully keep window on top using SetWindowPos."""
        try:
            import ctypes
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010)
        except Exception:
            pass

    def update_boxes(self, boxes):
        """boxes: list of (x, y, w, h, r, g, b, label) in screen coordinates."""
        self.boxes = boxes
        self.update()

    def clear_boxes(self):
        self.boxes.clear()
        self.update()

    def paintEvent(self, event):
        if not self.boxes:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Microsoft YaHei", 10)
        painter.setFont(font)
        for bx in self.boxes:
            x, y, w, h, r, g, b = bx[:7]
            label = bx[7] if len(bx) > 7 else ""
            color = QColor(r, g, b)
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, w, h)
            if label:
                # White text with dark outline for readability
                text_x, text_y = x, max(2, y - 6)
                painter.setPen(QColor(0, 0, 0, 180))
                painter.drawText(text_x + 1, text_y + 1, label)
                painter.drawText(text_x - 1, text_y + 1, label)
                painter.drawText(text_x + 1, text_y - 1, label)
                painter.drawText(text_x - 1, text_y - 1, label)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_x, text_y, label)

        painter.end()

    def show_overlay(self):
        self.show()

    def hide_overlay(self):
        self.hide()
        self.clear_boxes()
