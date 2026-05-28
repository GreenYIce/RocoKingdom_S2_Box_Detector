"""Visual settings panel for Roco Box Detector. Non-modal, tabbed UI."""

import os
import json
import copy
from typing import Optional
from image_utils import resolve_path

from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QDoubleSpinBox, QSpinBox, QCheckBox,
    QComboBox, QListWidget, QListWidgetItem, QLineEdit,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QInputDialog,
    QFrame, QColorDialog, QFontComboBox, QScrollArea, QAbstractSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QEvent, QObject
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPixmap


# ── helpers ──────────────────────────────────────────────────────────

def _make_slider_spin_double(
    parent, layout, label_text, minimum, maximum, default, step=0.01, decimals=2,
):
    """Add a labeled row: QLabel | QSlider | QDoubleSpinBox to a form layout."""
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setFixedWidth(120)
    row.addWidget(lbl)

    slider = QSlider(Qt.Horizontal)
    slider.setRange(int(minimum / step), int(maximum / step))
    slider.setValue(int(default / step))
    row.addWidget(slider, 1)

    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(default)
    spin.setFixedWidth(80)
    row.addWidget(spin)

    slider.valueChanged.connect(lambda v: spin.setValue(v * step))
    spin.valueChanged.connect(lambda v: slider.blockSignals(True) or slider.setValue(int(v / step)) or slider.blockSignals(False))

    layout.addRow(row)
    return slider, spin


def _make_checkbox_row(layout, label_text, default=False):
    """Add a checkbox to a form layout, returns the checkbox."""
    cb = QCheckBox(label_text)
    cb.setChecked(default)
    layout.addRow(cb)
    return cb


def _add_separator(layout):
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    layout.addRow(line)


# ── Sub-ROI Preview Widget ───────────────────────────────────────────

class SubRoiPreview(QLabel):
    """Draws a rectangle diagram showing anchor and up to two sub-ROI regions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.x_ratio = 0.42
        self.y_ratio = 0.33
        self.w_ratio = 0.42
        self.h_ratio = 0.34
        self.x2_ratio = 0.02
        self.y2_ratio = 0.33
        self.w2_ratio = 0.38
        self.h2_ratio = 0.34
        self.show_roi2 = False
        self.setFixedSize(280, 220)
        self.setStyleSheet("background: #2b2b2b; border: 1px solid #555;")
        self._redraw()

    def update_ratios(self, x, y, w, h):
        self.x_ratio = x
        self.y_ratio = y
        self.w_ratio = w
        self.h_ratio = h
        self._redraw()

    def update_ratios_2(self, x, y, w, h, visible=True):
        self.x2_ratio = x
        self.y2_ratio = y
        self.w2_ratio = w
        self.h2_ratio = h
        self.show_roi2 = visible
        self._redraw()

    def _redraw(self):
        pix = QPixmap(self.width(), self.height())
        pix.fill(QColor("#2b2b2b"))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        margin = 20
        draw_w = self.width() - 2 * margin
        draw_h = self.height() - 2 * margin

        # Anchor box (blue)
        painter.setPen(QPen(QColor("#4488ff"), 2))
        painter.setBrush(QColor(30, 60, 120, 80))
        painter.drawRect(margin, margin, draw_w, draw_h)

        # Sub-ROI1 box (yellow)
        sx = margin + int(draw_w * self.x_ratio)
        sy = margin + int(draw_h * self.y_ratio)
        sw = max(4, int(draw_w * self.w_ratio))
        sh = max(4, int(draw_h * self.h_ratio))
        painter.setPen(QPen(QColor("#ffdd44"), 2))
        painter.setBrush(QColor(120, 100, 20, 100))
        painter.drawRect(sx, sy, sw, sh)

        # Sub-ROI2 box (orange)
        if self.show_roi2:
            sx2 = margin + int(draw_w * self.x2_ratio)
            sy2 = margin + int(draw_h * self.y2_ratio)
            sw2 = max(4, int(draw_w * self.w2_ratio))
            sh2 = max(4, int(draw_h * self.h2_ratio))
            painter.setPen(QPen(QColor("#ff8833"), 2))
            painter.setBrush(QColor(120, 60, 20, 100))
            painter.drawRect(sx2, sy2, sw2, sh2)

        # Labels
        font = QFont("Microsoft YaHei", 8)
        painter.setFont(font)
        painter.setPen(QColor("#4488ff"))
        painter.drawText(margin + 4, margin + 14, "Anchor")
        painter.setPen(QColor("#ffdd44"))
        painter.drawText(sx + 2, sy + 12, "ROI1")
        if self.show_roi2:
            painter.setPen(QColor("#ff8833"))
            painter.drawText(sx2 + 2, sy2 + 12, "ROI2")

        painter.end()
        self.setPixmap(pix)


# ── Tab contents ─────────────────────────────────────────────────────

class AnchorTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        ac = self.config["anchor"]
        self.thresh_slider, self.thresh_spin = _make_slider_spin_double(
            self, form, "匹配阈值", 0.50, 0.99, ac["threshold"], 0.01, 2)
        self.scale_min_slider, self.scale_min_spin = _make_slider_spin_double(
            self, form, "最小缩放", 0.30, 1.50, ac["scale_min"], 0.05, 2)
        self.scale_max_slider, self.scale_max_spin = _make_slider_spin_double(
            self, form, "最大缩放", 0.50, 2.00, ac["scale_max"], 0.05, 2)
        self.scale_steps_spin = QSpinBox()
        self.scale_steps_spin.setRange(3, 50)
        self.scale_steps_spin.setValue(ac["scale_steps"])
        form.addRow("缩放步数", self.scale_steps_spin)

        self.gray_cb = QCheckBox("灰度匹配")
        self.gray_cb.setChecked(ac["use_grayscale"])
        form.addRow(self.gray_cb)
        self.canny_cb = QCheckBox("Canny边缘匹配")
        self.canny_cb.setChecked(ac["use_canny"])
        form.addRow(self.canny_cb)

        self.coarse_thresh_slider, self.coarse_thresh_spin = _make_slider_spin_double(
            self, form, "粗筛阈值", 0.00, 0.80, ac.get("coarse_threshold", 0.36), 0.01, 2)
        hint = QLabel("0=关闭粗筛。粗筛用单档scale快速过滤空帧，设太高会漏掉真盒子。")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        form.addRow(hint)

        _add_separator(form)

        # Template list
        tmpl_group = QGroupBox("Anchor 模板列表")
        tmpl_layout = QVBoxLayout(tmpl_group)
        self.tmpl_list = QListWidget()
        for p in ac["templates"]:
            self.tmpl_list.addItem(p)
        tmpl_layout.addWidget(self.tmpl_list)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ 添加模板")
        add_btn.clicked.connect(self._add_template)
        btn_row.addWidget(add_btn)
        del_btn = QPushButton("- 删除选中")
        del_btn.clicked.connect(self._del_template)
        btn_row.addWidget(del_btn)
        tmpl_layout.addLayout(btn_row)
        form.addRow(tmpl_group)

        layout.addLayout(form)
        layout.addStretch()

    def _add_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择Anchor模板图片", resolve_path("templates/box_anchor"),
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            # Store relative path
            # Normalize to templates/xxx.png (strip dev prefix, fix slashes)
            idx = path.replace('\\', '/').find('templates/')
            rel = path.replace('\\', '/')[idx:] if idx >= 0 else path
            self.tmpl_list.addItem(rel)

    def _del_template(self):
        for item in self.tmpl_list.selectedItems():
            self.tmpl_list.takeItem(self.tmpl_list.row(item))

    def collect(self, cfg: dict):
        ac = cfg["anchor"]
        ac["threshold"] = self.thresh_spin.value()
        ac["scale_min"] = self.scale_min_spin.value()
        ac["scale_max"] = self.scale_max_spin.value()
        ac["scale_steps"] = self.scale_steps_spin.value()
        ac["use_grayscale"] = self.gray_cb.isChecked()
        ac["use_canny"] = self.canny_cb.isChecked()
        ac["coarse_threshold"] = self.coarse_thresh_spin.value()
        ac["templates"] = [
            self.tmpl_list.item(i).text()
            for i in range(self.tmpl_list.count())
        ]


class SubRoiTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)

        # Left: controls
        ctrl_layout = QVBoxLayout()
        form = QFormLayout()

        sr = self.config["sub_roi"]
        self.x_slider, self.x_spin = _make_slider_spin_double(
            self, form, "X 比例", 0.00, 1.00, sr["x_ratio"], 0.01, 2)
        self.y_slider, self.y_spin = _make_slider_spin_double(
            self, form, "Y 比例", 0.00, 1.00, sr["y_ratio"], 0.01, 2)
        self.w_slider, self.w_spin = _make_slider_spin_double(
            self, form, "宽度比例", 0.01, 1.00, sr["w_ratio"], 0.01, 2)
        self.h_slider, self.h_spin = _make_slider_spin_double(
            self, form, "高度比例", 0.01, 1.00, sr["h_ratio"], 0.01, 2)

        ctrl_layout.addLayout(form)

        hint = QLabel(
            "提示：这些比例相对于 Anchor 框的宽高。\n"
            "X/Y 决定 Sub-ROI 左上角位置；W/H 决定大小。\n"
            "右侧预览图：蓝色=Anchor框，黄色=Sub-ROI。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        ctrl_layout.addWidget(hint)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Right: preview
        self.preview = SubRoiPreview()
        layout.addWidget(self.preview)

        # Connect signals
        for s, sp in [(self.x_slider, self.x_spin), (self.y_slider, self.y_spin),
                       (self.w_slider, self.w_spin), (self.h_slider, self.h_spin)]:
            pass  # connections handled by slider-spin sync

        # Use a timer to update preview (avoid per-tick repaint spam)
        self._preview_timer = QTimer()
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start()

    def _update_preview(self):
        self.preview.update_ratios(
            self.x_spin.value(), self.y_spin.value(),
            self.w_spin.value(), self.h_spin.value(),
        )

    def collect(self, cfg: dict):
        sr = cfg["sub_roi"]
        sr["x_ratio"] = self.x_spin.value()
        sr["y_ratio"] = self.y_spin.value()
        sr["w_ratio"] = self.w_spin.value()
        sr["h_ratio"] = self.h_spin.value()


class SubRoi2Tab(QWidget):
    """Settings for second sub-ROI region (yellow/orange box 2)."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)

        # Left: controls
        ctrl_layout = QVBoxLayout()
        form = QFormLayout()

        sr = self.config.get("sub_roi_2", {})
        self.enabled_cb = QCheckBox("启用区域二 (ROI2)")
        self.enabled_cb.setChecked(sr.get("enabled", False))
        self.enabled_cb.toggled.connect(self._on_enabled_toggled)
        form.addRow(self.enabled_cb)

        self.x_slider, self.x_spin = _make_slider_spin_double(
            self, form, "X 比例", 0.00, 1.00, sr.get("x_ratio", 0.02), 0.01, 2)
        self.y_slider, self.y_spin = _make_slider_spin_double(
            self, form, "Y 比例", 0.00, 1.00, sr.get("y_ratio", 0.33), 0.01, 2)
        self.w_slider, self.w_spin = _make_slider_spin_double(
            self, form, "宽度比例", 0.01, 1.00, sr.get("w_ratio", 0.38), 0.01, 2)
        self.h_slider, self.h_spin = _make_slider_spin_double(
            self, form, "高度比例", 0.01, 1.00, sr.get("h_ratio", 0.34), 0.01, 2)

        ctrl_layout.addLayout(form)

        hint = QLabel(
            "提示：区域二是第二个独立识别区域。\n"
            "启用后在 Anchor 命中时同时截取两个黄框。\n"
            "请在 Patterns 2 标签页中配置区域二的模板。\n"
            "右侧预览图：蓝色=Anchor，黄=ROI1，橙=ROI2。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        ctrl_layout.addWidget(hint)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Right: preview
        self.preview = SubRoiPreview()
        layout.addWidget(self.preview)

        # Timer for preview update
        self._preview_timer = QTimer()
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start()

        self._on_enabled_toggled(self.enabled_cb.isChecked())

    def _on_enabled_toggled(self, enabled):
        for w in [self.x_spin, self.y_spin, self.w_spin, self.h_spin,
                   self.x_slider, self.y_slider, self.w_slider, self.h_slider]:
            w.setEnabled(enabled)

    def _update_preview(self):
        # Also show ROI1 position from current config
        sr1 = self.config.get("sub_roi", {})
        self.preview.update_ratios(
            sr1.get("x_ratio", 0.0), sr1.get("y_ratio", 0.48),
            sr1.get("w_ratio", 0.6), sr1.get("h_ratio", 0.25),
        )
        self.preview.update_ratios_2(
            self.x_spin.value(), self.y_spin.value(),
            self.w_spin.value(), self.h_spin.value(),
            visible=self.enabled_cb.isChecked(),
        )

    def collect(self, cfg: dict):
        sr = cfg.setdefault("sub_roi_2", {})
        sr["enabled"] = self.enabled_cb.isChecked()
        sr["x_ratio"] = self.x_spin.value()
        sr["y_ratio"] = self.y_spin.value()
        sr["w_ratio"] = self.w_spin.value()
        sr["h_ratio"] = self.h_spin.value()


class PatternsTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_name: Optional[str] = None
        self._building = False
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Pattern selector
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("选择图案:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.currentTextChanged.connect(self._on_pattern_changed)
        sel_row.addWidget(self.pattern_combo, 1)

        self.rename_btn = QPushButton("重命名")
        self.rename_btn.clicked.connect(self._rename_pattern)
        sel_row.addWidget(self.rename_btn)
        layout.addLayout(sel_row)

        # Pattern settings
        form = QFormLayout()
        self.thresh_slider, self.thresh_spin = _make_slider_spin_double(
            self, form, "匹配阈值", 0.50, 0.99, 0.76, 0.01, 2)
        self.scale_min_slider, self.scale_min_spin = _make_slider_spin_double(
            self, form, "最小缩放", 0.30, 1.50, 0.75, 0.05, 2)
        self.scale_max_slider, self.scale_max_spin = _make_slider_spin_double(
            self, form, "最大缩放", 0.50, 2.00, 1.25, 0.05, 2)
        self.scale_steps_spin = QSpinBox()
        self.scale_steps_spin.setRange(3, 50)
        self.scale_steps_spin.setValue(11)
        form.addRow("缩放步数", self.scale_steps_spin)
        self.gray_cb = QCheckBox("灰度匹配")
        self.gray_cb.setChecked(True)
        form.addRow(self.gray_cb)
        self.canny_cb = QCheckBox("Canny边缘匹配")
        self.canny_cb.setChecked(False)
        form.addRow(self.canny_cb)
        layout.addLayout(form)

        # Template list
        tmpl_group = QGroupBox("图案模板列表")
        tmpl_layout = QVBoxLayout(tmpl_group)
        self.tmpl_list = QListWidget()
        tmpl_layout.addWidget(self.tmpl_list)
        btn_row = QHBoxLayout()
        btn_row.addWidget(QPushButton("+ 添加模板", clicked=self._add_template))
        btn_row.addWidget(QPushButton("- 删除选中", clicked=self._del_template))
        tmpl_layout.addLayout(btn_row)
        layout.addWidget(tmpl_group)

        # Pattern add/remove
        pat_btn_row = QHBoxLayout()
        pat_btn_row.addWidget(QPushButton("+ 新增图案", clicked=self._add_pattern))
        self.del_pat_btn = QPushButton("- 删除当前图案", clicked=self._del_pattern)
        pat_btn_row.addWidget(self.del_pat_btn)
        pat_btn_row.addStretch()
        layout.addLayout(pat_btn_row)

        self._refresh_combo()

    def _refresh_combo(self):
        self._building = True
        self.pattern_combo.clear()
        names = list(self.config["patterns"].keys())
        self.pattern_combo.addItems(names)
        if names:
            self.pattern_combo.setCurrentIndex(0)
            self._current_name = names[0]
            self._load_pattern(names[0])
        else:
            self._current_name = None
        self._building = False

    def _on_pattern_changed(self, name: str):
        if self._building or not name:
            return
        self._current_name = name
        self._load_pattern(name)

    def _load_pattern(self, name: str):
        pcfg = self.config["patterns"].get(name)
        if not pcfg:
            return
        self.thresh_spin.setValue(pcfg["threshold"])
        self.scale_min_spin.setValue(pcfg["scale_min"])
        self.scale_max_spin.setValue(pcfg["scale_max"])
        self.scale_steps_spin.setValue(pcfg["scale_steps"])
        self.gray_cb.setChecked(pcfg["use_grayscale"])
        self.canny_cb.setChecked(pcfg["use_canny"])
        self.tmpl_list.clear()
        for p in pcfg["templates"]:
            self.tmpl_list.addItem(p)

    def _add_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图案模板图片", resolve_path("templates/patterns"),
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            # Normalize to templates/xxx.png (strip dev prefix, fix slashes)
            idx = path.replace('\\', '/').find('templates/')
            rel = path.replace('\\', '/')[idx:] if idx >= 0 else path
            self.tmpl_list.addItem(rel)

    def _del_template(self):
        for item in self.tmpl_list.selectedItems():
            self.tmpl_list.takeItem(self.tmpl_list.row(item))

    def _add_pattern(self):
        name, ok = QInputDialog.getText(self, "新增图案", "图案名称:")
        if ok and name.strip():
            name = name.strip()
            if name in self.config["patterns"]:
                QMessageBox.warning(self, "错误", f"图案 '{name}' 已存在。")
                return
            self.config["patterns"][name] = {
                "templates": [],
                "threshold": 0.76,
                "scale_min": 0.75,
                "scale_max": 1.25,
                "scale_steps": 11,
                "use_grayscale": True,
                "use_canny": False,
            }
            self._refresh_combo()
            idx = self.pattern_combo.findText(name)
            if idx >= 0:
                self.pattern_combo.setCurrentIndex(idx)

    def _del_pattern(self):
        if not self._current_name:
            return
        if len(self.config["patterns"]) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一个图案。")
            return
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除图案 '{self._current_name}' 吗？",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.config["patterns"][self._current_name]
            self._refresh_combo()

    def _rename_pattern(self):
        if not self._current_name:
            return
        new_name, ok = QInputDialog.getText(
            self, "重命名图案", "新名称:", text=self._current_name)
        if ok and new_name.strip() and new_name.strip() != self._current_name:
            new_name = new_name.strip()
            if new_name in self.config["patterns"]:
                QMessageBox.warning(self, "错误", f"图案 '{new_name}' 已存在。")
                return
            self.config["patterns"][new_name] = self.config["patterns"].pop(self._current_name)
            self._refresh_combo()
            idx = self.pattern_combo.findText(new_name)
            if idx >= 0:
                self.pattern_combo.setCurrentIndex(idx)

    def collect(self, cfg: dict):
        """Write current tab fields back to the target config dict."""
        if not self._current_name:
            return
        # Save current pattern edits before collecting
        pcfg = cfg["patterns"].get(self._current_name)
        if pcfg is None:
            return
        pcfg["threshold"] = self.thresh_spin.value()
        pcfg["scale_min"] = self.scale_min_spin.value()
        pcfg["scale_max"] = self.scale_max_spin.value()
        pcfg["scale_steps"] = self.scale_steps_spin.value()
        pcfg["use_grayscale"] = self.gray_cb.isChecked()
        pcfg["use_canny"] = self.canny_cb.isChecked()
        pcfg["templates"] = [
            self.tmpl_list.item(i).text()
            for i in range(self.tmpl_list.count())
        ]


class Patterns2Tab(QWidget):
    """Settings for second pattern group (ROI2 templates)."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_name: Optional[str] = None
        self._building = False
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Pattern selector
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("选择图案:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.currentTextChanged.connect(self._on_pattern_changed)
        sel_row.addWidget(self.pattern_combo, 1)

        self.rename_btn = QPushButton("重命名")
        self.rename_btn.clicked.connect(self._rename_pattern)
        sel_row.addWidget(self.rename_btn)
        layout.addLayout(sel_row)

        # Pattern settings
        form = QFormLayout()
        self.thresh_slider, self.thresh_spin = _make_slider_spin_double(
            self, form, "匹配阈值", 0.50, 0.99, 0.76, 0.01, 2)
        self.scale_min_slider, self.scale_min_spin = _make_slider_spin_double(
            self, form, "最小缩放", 0.30, 1.50, 0.75, 0.05, 2)
        self.scale_max_slider, self.scale_max_spin = _make_slider_spin_double(
            self, form, "最大缩放", 0.50, 2.00, 1.25, 0.05, 2)
        self.scale_steps_spin = QSpinBox()
        self.scale_steps_spin.setRange(3, 50)
        self.scale_steps_spin.setValue(11)
        form.addRow("缩放步数", self.scale_steps_spin)
        self.gray_cb = QCheckBox("灰度匹配")
        self.gray_cb.setChecked(True)
        form.addRow(self.gray_cb)
        self.canny_cb = QCheckBox("Canny边缘匹配")
        self.canny_cb.setChecked(False)
        form.addRow(self.canny_cb)
        layout.addLayout(form)

        # Template list
        tmpl_group = QGroupBox("ROI2 图案模板列表")
        tmpl_layout = QVBoxLayout(tmpl_group)
        self.tmpl_list = QListWidget()
        tmpl_layout.addWidget(self.tmpl_list)
        btn_row = QHBoxLayout()
        btn_row.addWidget(QPushButton("+ 添加模板", clicked=self._add_template))
        btn_row.addWidget(QPushButton("- 删除选中", clicked=self._del_template))
        tmpl_layout.addLayout(btn_row)
        layout.addWidget(tmpl_group)

        # Pattern add/remove
        pat_btn_row = QHBoxLayout()
        pat_btn_row.addWidget(QPushButton("+ 新增图案", clicked=self._add_pattern))
        self.del_pat_btn = QPushButton("- 删除当前图案", clicked=self._del_pattern)
        pat_btn_row.addWidget(self.del_pat_btn)
        pat_btn_row.addStretch()
        layout.addLayout(pat_btn_row)

        self._refresh_combo()

    def _pattern_key(self):
        return "patterns_2"

    def _refresh_combo(self):
        self._building = True
        self.pattern_combo.clear()
        names = list(self.config.get(self._pattern_key(), {}).keys())
        self.pattern_combo.addItems(names)
        if names:
            self.pattern_combo.setCurrentIndex(0)
            self._current_name = names[0]
            self._load_pattern(names[0])
        else:
            self._current_name = None
        self._building = False

    def _on_pattern_changed(self, name: str):
        if self._building or not name:
            return
        self._current_name = name
        self._load_pattern(name)

    def _load_pattern(self, name: str):
        pcfg = self.config.get(self._pattern_key(), {}).get(name)
        if not pcfg:
            return
        self.thresh_spin.setValue(pcfg["threshold"])
        self.scale_min_spin.setValue(pcfg["scale_min"])
        self.scale_max_spin.setValue(pcfg["scale_max"])
        self.scale_steps_spin.setValue(pcfg["scale_steps"])
        self.gray_cb.setChecked(pcfg["use_grayscale"])
        self.canny_cb.setChecked(pcfg["use_canny"])
        self.tmpl_list.clear()
        for p in pcfg["templates"]:
            self.tmpl_list.addItem(p)

    def _add_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择ROI2图案模板图片", resolve_path("templates/patterns_2"),
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            idx = path.replace('\\', '/').find('templates/')
            rel = path.replace('\\', '/')[idx:] if idx >= 0 else path
            self.tmpl_list.addItem(rel)

    def _del_template(self):
        for item in self.tmpl_list.selectedItems():
            self.tmpl_list.takeItem(self.tmpl_list.row(item))

    def _add_pattern(self):
        name, ok = QInputDialog.getText(self, "新增ROI2图案", "图案名称:")
        if ok and name.strip():
            name = name.strip()
            pk = self._pattern_key()
            if name in self.config.get(pk, {}):
                QMessageBox.warning(self, "错误", f"图案 '{name}' 已存在。")
                return
            self.config.setdefault(pk, {})[name] = {
                "templates": [],
                "threshold": 0.76,
                "scale_min": 0.75,
                "scale_max": 1.25,
                "scale_steps": 11,
                "use_grayscale": True,
                "use_canny": False,
            }
            self._refresh_combo()
            idx = self.pattern_combo.findText(name)
            if idx >= 0:
                self.pattern_combo.setCurrentIndex(idx)

    def _del_pattern(self):
        if not self._current_name:
            return
        pk = self._pattern_key()
        if len(self.config.get(pk, {})) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一个图案。")
            return
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除ROI2图案 '{self._current_name}' 吗？",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.config[pk][self._current_name]
            self._refresh_combo()

    def _rename_pattern(self):
        if not self._current_name:
            return
        new_name, ok = QInputDialog.getText(
            self, "重命名ROI2图案", "新名称:", text=self._current_name)
        if ok and new_name.strip() and new_name.strip() != self._current_name:
            new_name = new_name.strip()
            pk = self._pattern_key()
            if new_name in self.config.get(pk, {}):
                QMessageBox.warning(self, "错误", f"图案 '{new_name}' 已存在。")
                return
            self.config[pk][new_name] = self.config[pk].pop(self._current_name)
            self._refresh_combo()
            idx = self.pattern_combo.findText(new_name)
            if idx >= 0:
                self.pattern_combo.setCurrentIndex(idx)

    def collect(self, cfg: dict):
        if not self._current_name:
            return
        pk = self._pattern_key()
        pcfg = cfg.get(pk, {}).get(self._current_name)
        if pcfg is None:
            return
        pcfg["threshold"] = self.thresh_spin.value()
        pcfg["scale_min"] = self.scale_min_spin.value()
        pcfg["scale_max"] = self.scale_max_spin.value()
        pcfg["scale_steps"] = self.scale_steps_spin.value()
        pcfg["use_grayscale"] = self.gray_cb.isChecked()
        pcfg["use_canny"] = self.canny_cb.isChecked()
        pcfg["templates"] = [
            self.tmpl_list.item(i).text()
            for i in range(self.tmpl_list.count())
        ]


class RuntimeTab(QWidget):
    resolution_changed = pyqtSignal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.res_combo = QComboBox()
        self.res_combo.addItems(["720p", "1080p", "2K", "4K"])
        cur = self.config.get("game_resolution", "2K")
        self.res_combo.setCurrentText(cur)
        self.res_combo.currentTextChanged.connect(self._apply_resolution_preset)
        form.addRow("游戏分辨率预设", self.res_combo)

        # Solo / Duo
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        mode_row = QHBoxLayout()
        mode_label = QLabel("抓捕模式")
        mode_label.setFixedWidth(120)
        mode_row.addWidget(mode_label)
        self.solo_btn = QRadioButton("单人")
        self.solo_btn.setStyleSheet("color: #aaa;")
        self.duo_btn = QRadioButton("双人")
        self.duo_btn.setStyleSheet("color: #aaa;")
        self._mode_group = QButtonGroup(mode_row)
        self._mode_group.addButton(self.solo_btn, 0)
        self._mode_group.addButton(self.duo_btn, 1)
        cur_mode = self.config.get("capture_mode", "solo")
        self.solo_btn.setChecked(cur_mode != "duo")
        self.duo_btn.setChecked(cur_mode == "duo")
        self._mode_group.buttonClicked.connect(self._apply_mode)
        mode_row.addWidget(self.solo_btn)
        mode_row.addWidget(self.duo_btn)
        mode_row.addStretch()
        form.addRow(mode_row)

        rt = self.config["runtime"]
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(rt["capture_fps"])
        self.fps_spin.setSuffix(" fps")
        form.addRow("检测帧率", self.fps_spin)

        self.norm_spin = QSpinBox()
        self.norm_spin.setRange(0, 3840)
        self.norm_spin.setValue(rt["normalize_roi_width"])
        self.norm_spin.setSpecialValueText("禁用")
        self.norm_spin.setSuffix(" px" if rt["normalize_roi_width"] > 0 else "")
        form.addRow("ROI归一化宽度", self.norm_spin)

        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 10)
        self.skip_spin.setValue(rt.get("anchor_skip_frames", 0))
        self.skip_spin.setSpecialValueText("关闭")
        self.skip_spin.setSuffix(" 帧跳1" if rt.get("anchor_skip_frames", 0) > 0 else "")
        form.addRow("Anchor跳帧间隔", self.skip_spin)

        self.log_spin = QSpinBox()
        self.log_spin.setRange(1, 1000)
        self.log_spin.setValue(rt["log_every_n_frames"])
        form.addRow("每N帧输出日志", self.log_spin)

        self.hide_spin = QSpinBox()
        self.hide_spin.setRange(1, 300)
        self.hide_spin.setValue(rt["hide_after_frames"])
        self.hide_spin.setSuffix(" 帧")
        form.addRow("识别消失延迟", self.hide_spin)

        layout.addLayout(form)

        # ── width-only scale group ──
        ws_cfg = self.config.get("pattern_width_scale", {})
        ws_group = QGroupBox("宽度缩放匹配 (对抗透视畸变)")
        ws_layout = QFormLayout(ws_group)

        self.ws_enabled_cb = _make_checkbox_row(ws_layout, "启用", ws_cfg.get("enabled", False))
        hint = QLabel("对每个图案组中最宽的几个模板，额外生成仅压缩宽度的缩放变体")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        ws_layout.addRow(hint)

        self.ws_top_n_spin = QSpinBox()
        self.ws_top_n_spin.setRange(1, 10)
        self.ws_top_n_spin.setValue(ws_cfg.get("top_n", 3))
        self.ws_top_n_spin.setSuffix(" 个最宽模板")
        ws_layout.addRow("覆盖模板数", self.ws_top_n_spin)

        self.ws_scmin_slider, self.ws_scmin_spin = _make_slider_spin_double(
            self, ws_layout, "最小宽度比", 0.30, 0.95, ws_cfg.get("scale_min", 0.5), 0.05, 2)
        self.ws_scmax_slider, self.ws_scmax_spin = _make_slider_spin_double(
            self, ws_layout, "最大宽度比", 0.50, 1.00, ws_cfg.get("scale_max", 1.0), 0.05, 2)

        self.ws_scsteps_spin = QSpinBox()
        self.ws_scsteps_spin.setRange(2, 20)
        self.ws_scsteps_spin.setValue(ws_cfg.get("scale_steps", 5))
        ws_layout.addRow("宽度缩放档数", self.ws_scsteps_spin)

        layout.addWidget(ws_group)
        layout.addStretch()

    def _apply_mode(self):
        self.config["capture_mode"] = "duo" if self.duo_btn.isChecked() else "solo"
        self._apply_resolution_preset(self.res_combo.currentText())

    def _apply_resolution_preset(self, res: str):
        self.config["game_resolution"] = res
        presets = {
            "720p":  {"anchor": (0.55, 0.65, 4), "pattern": (0.35, 0.75, 6)},
            "1080p": {"anchor": (0.75, 0.90, 5), "pattern": (0.50, 1.05, 7)},
            "2K":    {"anchor": (0.90, 1.25, 5), "pattern": (0.65, 1.35, 8)},
            "4K":    {"anchor": (1.35, 1.80, 5), "pattern": (1.00, 1.70, 8)},
        }
        p = presets.get(res, presets["2K"])
        # Update anchor
        ac = self.config.setdefault("anchor", {})
        ac["threshold"] = 0.85
        smin, smax, ssteps = p["anchor"]
        # Duo mode: reduce anchor scale by 0.1
        if self.duo_btn.isChecked():
            smin = max(0.3, smin - 0.1)
            smax = max(smin + 0.1, smax - 0.1)
        ac["scale_min"], ac["scale_max"], ac["scale_steps"] = smin, smax, ssteps
        # Update all pattern groups
        for pk in ("patterns", "patterns_2"):
            for _, pcfg in self.config.get(pk, {}).items():
                pcfg["scale_min"], pcfg["scale_max"], pcfg["scale_steps"] = p["pattern"]
        self.resolution_changed.emit()

    def collect(self, cfg: dict):
        cfg["game_resolution"] = self.res_combo.currentText()
        cfg["capture_mode"] = "duo" if self.duo_btn.isChecked() else "solo"
        rt = cfg["runtime"]
        rt["capture_fps"] = self.fps_spin.value()
        rt["normalize_roi_width"] = self.norm_spin.value()
        rt["anchor_skip_frames"] = self.skip_spin.value()
        rt["log_every_n_frames"] = self.log_spin.value()
        rt["hide_after_frames"] = self.hide_spin.value()
        ws = cfg.setdefault("pattern_width_scale", {})
        ws["enabled"] = self.ws_enabled_cb.isChecked()
        ws["top_n"] = self.ws_top_n_spin.value()
        ws["scale_min"] = self.ws_scmin_spin.value()
        ws["scale_max"] = self.ws_scmax_spin.value()
        ws["scale_steps"] = self.ws_scsteps_spin.value()


class DebugTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        db = self.config["debug"]
        self.enabled_cb = _make_checkbox_row(form, "启用调试", db["enabled"])
        self.show_preview_cb = _make_checkbox_row(form, "显示预览窗口", db["show_preview_window"])
        self.save_frames_cb = _make_checkbox_row(form, "保存调试图片", db["save_debug_frames"])

        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(1, 60)
        self.save_interval_spin.setValue(db["save_every_n_seconds"])
        self.save_interval_spin.setSuffix(" 秒")
        form.addRow("保存间隔", self.save_interval_spin)

        self.draw_anchor_cb = _make_checkbox_row(form, "画Anchor框 (蓝)", db["draw_anchor_box"])
        self.draw_sub_cb = _make_checkbox_row(form, "画Sub-ROI框 (黄)", db["draw_sub_roi_box"])
        self.draw_pattern_cb = _make_checkbox_row(form, "画Pattern框 (绿/红)", db["draw_pattern_box"])
        self.print_scores_cb = _make_checkbox_row(form, "控制台输出分数", db["print_scores"])

        _add_separator(form)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit(db["debug_output_dir"])
        dir_row.addWidget(self.dir_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        form.addRow("输出目录", dir_row)

        layout.addLayout(form)
        layout.addStretch()

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择调试输出目录", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def collect(self, cfg: dict):
        db = cfg["debug"]
        db["enabled"] = self.enabled_cb.isChecked()
        db["show_preview_window"] = self.show_preview_cb.isChecked()
        db["save_debug_frames"] = self.save_frames_cb.isChecked()
        db["save_every_n_seconds"] = self.save_interval_spin.value()
        db["draw_anchor_box"] = self.draw_anchor_cb.isChecked()
        db["draw_sub_roi_box"] = self.draw_sub_cb.isChecked()
        db["draw_pattern_box"] = self.draw_pattern_cb.isChecked()
        db["print_scores"] = self.print_scores_cb.isChecked()
        db["debug_output_dir"] = self.dir_edit.text()


def _make_color_row(layout, label_text, default_color):
    """Add a row: label | color preview button | hex text."""
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setFixedWidth(120)
    row.addWidget(lbl)
    preview = QPushButton()
    preview.setFixedSize(28, 22)
    preview.setStyleSheet(f"background: {default_color}; border: 1px solid #555; border-radius: 3px;")
    hex_edit = QLineEdit(default_color)
    hex_edit.setFixedWidth(160)
    row.addWidget(preview)
    row.addWidget(hex_edit)

    def pick():
        c = QColor(default_color) if default_color.startswith("#") else QColor(0, 0, 0, 180)
        color = QColorDialog.getColor(c, None, f"选择{label_text}")
        if color.isValid():
            if color.alpha() < 255:
                new = f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()})"
            else:
                new = color.name()
            hex_edit.setText(new)
            preview.setStyleSheet(f"background: {new}; border: 1px solid #555; border-radius: 3px;")

    preview.clicked.connect(pick)
    hex_edit.textChanged.connect(lambda t: preview.setStyleSheet(
        f"background: {t}; border: 1px solid #555; border-radius: 3px;"))
    layout.addRow(row)
    return hex_edit


class OverlayTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        ov = self.config["overlay"]
        self.width_spin = QSpinBox()
        self.width_spin.setRange(200, 1920)
        self.width_spin.setValue(ov["width"])
        self.width_spin.setSuffix(" px")
        form.addRow("悬浮窗宽度", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(30, 200)
        self.height_spin.setValue(ov["height"])
        self.height_spin.setSuffix(" px")
        form.addRow("悬浮窗高度", self.height_spin)

        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["top", "bottom", "center"])
        self.pos_combo.setCurrentText(ov["position"])
        form.addRow("悬浮窗位置", self.pos_combo)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 60.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(ov.get("match_show_seconds", 3.0))
        self.duration_spin.setSuffix(" 秒")
        form.addRow("识别提示持续", self.duration_spin)

        form.addRow(QLabel(""))

        self.normal_edit = QLineEdit(ov["normal_text"])
        form.addRow("默认文本", self.normal_edit)

        self.prefix_edit = QLineEdit(ov.get("matched_prefix", "识别到："))
        form.addRow("识别前缀", self.prefix_edit)

        form.addRow(QLabel(""))

        self.bg_color_edit = _make_color_row(form, "背景颜色",
                                              ov.get("bg_color", "rgba(0, 0, 0, 180)"))
        self.text_color_edit = _make_color_row(form, "默认文字颜色",
                                               ov.get("text_color", "#ffffff"))
        self.matched_color_edit = _make_color_row(form, "识别文字颜色",
                                                  ov.get("matched_text_color", "#00ff88"))

        layout.addLayout(form)
        layout.addStretch()

    def collect(self, cfg: dict):
        ov = cfg["overlay"]
        ov["width"] = self.width_spin.value()
        ov["height"] = self.height_spin.value()
        ov["position"] = self.pos_combo.currentText()
        ov["match_show_seconds"] = self.duration_spin.value()
        ov["normal_text"] = self.normal_edit.text()
        ov["matched_prefix"] = self.prefix_edit.text()
        ov["bg_color"] = self.bg_color_edit.text()
        ov["text_color"] = self.text_color_edit.text()
        ov["matched_text_color"] = self.matched_color_edit.text()


# ── Result Text Tab ──────────────────────────────────────────────────

class ResultTextTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._build()
        self._scroll = None

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll = scroll

        container = QWidget()
        layout = QVBoxLayout(container)
        form = QFormLayout()

        cfg = self.config.get("result_text_overlay", {})
        self.enabled_cb = _make_checkbox_row(form, "启用识别历史面板", cfg.get("enabled", True))

        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(cfg.get("font_family", "Microsoft YaHei")))
        form.addRow("字体", self.font_combo)

        self.header_font_spin = QSpinBox()
        self.header_font_spin.setRange(8, 24)
        self.header_font_spin.setValue(cfg.get("header_font_size", 12))
        self.header_font_spin.setSuffix(" px")
        form.addRow("标题字号", self.header_font_spin)

        self.status_font_spin = QSpinBox()
        self.status_font_spin.setRange(8, 24)
        self.status_font_spin.setValue(cfg.get("status_font_size", 11))
        self.status_font_spin.setSuffix(" px")
        form.addRow("状态字号", self.status_font_spin)

        self.chip_font_spin = QSpinBox()
        self.chip_font_spin.setRange(10, 36)
        self.chip_font_spin.setValue(cfg.get("chip_font_size", 14))
        self.chip_font_spin.setSuffix(" px")
        form.addRow("识别结果字号", self.chip_font_spin)

        self.counter_font_spin = QSpinBox()
        self.counter_font_spin.setRange(8, 24)
        self.counter_font_spin.setValue(cfg.get("counter_font_size", 11))
        self.counter_font_spin.setSuffix(" px")
        form.addRow("计数区字号", self.counter_font_spin)

        self.counter_title_font_spin = QSpinBox()
        self.counter_title_font_spin.setRange(8, 24)
        self.counter_title_font_spin.setValue(cfg.get("counter_title_font_size", 10))
        self.counter_title_font_spin.setSuffix(" px")
        form.addRow("计数区标题字号", self.counter_title_font_spin)

        self.color_edit = _make_color_row(form, "默认颜色", cfg.get("default_color", "#FFD700"))
        self.plus_color_edit = _make_color_row(form, "加号颜色", cfg.get("plus_color", "#DDDDDD"))
        self.count_color_edit = _make_color_row(form, "计数颜色", cfg.get("count_color", "#AAAAAA"))

        self.max_items_spin = QSpinBox()
        self.max_items_spin.setRange(1, 500)
        self.max_items_spin.setValue(cfg.get("max_items", 104))
        form.addRow("最大记录数", self.max_items_spin)

        self.fallback_edit = QLineEdit(cfg.get("roi_fallback", ""))
        self.fallback_edit.setPlaceholderText("未识别时的替代文字，如：太快了没看清喵")
        form.addRow("未识别替代文字", self.fallback_edit)

        self.title_edit = QLineEdit(cfg.get("title", "识别记录"))
        form.addRow("标题文字", self.title_edit)

        self.bg_color_edit = _make_color_row(form, "背景颜色", cfg.get("background_color", "rgba(8,10,16,190)"))
        self.border_color_edit = _make_color_row(form, "边框颜色", cfg.get("border_color", "rgba(255,255,255,45)"))

        self.border_radius_spin = QSpinBox()
        self.border_radius_spin.setRange(0, 30)
        self.border_radius_spin.setValue(cfg.get("border_radius", 12))
        form.addRow("圆角", self.border_radius_spin)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(2, 30)
        self.padding_spin.setValue(cfg.get("padding", 10))
        form.addRow("内边距", self.padding_spin)

        self.scrollbar_width_spin = QSpinBox()
        self.scrollbar_width_spin.setRange(2, 16)
        self.scrollbar_width_spin.setValue(cfg.get("scrollbar_width", 7))
        form.addRow("滚动条宽度", self.scrollbar_width_spin)

        self.click_through_cb = _make_checkbox_row(form, "鼠标穿透", cfg.get("click_through", True))

        layout.addLayout(form)

        # ── counter settings ──
        counter_group = QGroupBox("底部计数区")
        counter_form = QFormLayout(counter_group)
        self.show_counter_cb = _make_checkbox_row(counter_form, "显示计数区",
                                                    cfg.get("show_counter_area", True))
        self.counter_top_n_spin = QSpinBox()
        self.counter_top_n_spin.setRange(1, 20)
        self.counter_top_n_spin.setValue(cfg.get("counter_top_n", 5))
        counter_form.addRow("最多显示条目", self.counter_top_n_spin)
        layout.addWidget(counter_group)

        # ── bloodline colors (patterns / ROI1) ──
        bl_colors = cfg.get("bloodline_colors", {})
        patterns1 = self.config.get("patterns", {})
        if patterns1:
            bl_group = QGroupBox("血脉颜色 (样本1)")
            bl_group_layout = QFormLayout(bl_group)
            self._bl_color_edits = {}
            for pname in patterns1:
                default = bl_colors.get(pname, "#FFD700")
                edit = _make_color_row(bl_group_layout, pname, default)
                self._bl_color_edits[pname] = edit
            layout.addWidget(bl_group)

        # ── attribute colors (patterns_2 / ROI2) ──
        attr_colors = cfg.get("attribute_colors", {})
        patterns2 = self.config.get("patterns_2", {})
        if patterns2:
            attr_group = QGroupBox("属性颜色 (样本2)")
            attr_group_layout = QFormLayout(attr_group)
            self._attr_color_edits = {}
            for pname in patterns2:
                default = attr_colors.get(pname, "#FFD700")
                edit = _make_color_row(attr_group_layout, pname, default)
                self._attr_color_edits[pname] = edit
            layout.addWidget(attr_group)

        # ── legacy label_colors (fallback) ──
        lbl_colors = cfg.get("label_colors", {})
        all_patterns = {**patterns1, **patterns2}
        if all_patterns:
            lc_group = QGroupBox("兼容颜色 (label_colors)")
            lc_group_layout = QFormLayout(lc_group)
            self._label_color_edits = {}
            for pname in all_patterns:
                default = lbl_colors.get(pname, "#FFD700")
                tag = " [ROI2]" if pname in patterns2 and pname not in patterns1 else ""
                edit = _make_color_row(lc_group_layout, pname + tag, default)
                self._label_color_edits[pname] = edit
            layout.addWidget(lc_group)

        hint = QLabel("血脉=样本1匹配结果，属性=样本2匹配结果。\n"
                      "面板可拖动标题栏移动，右下角拖拽调整大小。\n"
                      "按 Alt 键可临时显示鼠标进行交互。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)
        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    def collect(self, cfg: dict):
        rt = cfg.setdefault("result_text_overlay", {})
        rt["enabled"] = self.enabled_cb.isChecked()
        rt["font_family"] = self.font_combo.currentFont().family()
        rt["header_font_size"] = self.header_font_spin.value()
        rt["status_font_size"] = self.status_font_spin.value()
        rt["chip_font_size"] = self.chip_font_spin.value()
        rt["counter_font_size"] = self.counter_font_spin.value()
        rt["counter_title_font_size"] = self.counter_title_font_spin.value()
        rt["default_color"] = self.color_edit.text()
        rt["plus_color"] = self.plus_color_edit.text()
        rt["count_color"] = self.count_color_edit.text()
        rt["max_items"] = self.max_items_spin.value()
        rt["roi_fallback"] = self.fallback_edit.text()
        rt["title"] = self.title_edit.text()
        rt["background_color"] = self.bg_color_edit.text()
        rt["border_color"] = self.border_color_edit.text()
        rt["border_radius"] = self.border_radius_spin.value()
        rt["padding"] = self.padding_spin.value()
        rt["scrollbar_width"] = self.scrollbar_width_spin.value()
        rt["click_through"] = self.click_through_cb.isChecked()
        rt["show_counter_area"] = self.show_counter_cb.isChecked()
        rt["counter_top_n"] = self.counter_top_n_spin.value()
        # Carry over position/size from live state
        old = self.config.get("result_text_overlay", {})
        for k in ("x", "y", "width", "height", "min_width", "min_height"):
            if k in old:
                rt.setdefault(k, old[k])
        if hasattr(self, '_bl_color_edits'):
            bc = {}
            for pname, edit in self._bl_color_edits.items():
                bc[pname] = edit.text()
            rt["bloodline_colors"] = bc
        if hasattr(self, '_attr_color_edits'):
            ac = {}
            for pname, edit in self._attr_color_edits.items():
                ac[pname] = edit.text()
            rt["attribute_colors"] = ac
        if hasattr(self, '_label_color_edits'):
            lc = {}
            for pname, edit in self._label_color_edits.items():
                lc[pname] = edit.text()
            rt["label_colors"] = lc


# ── Wheel blocker event filter ────────────────────────────────────────


class _WheelBlocker(QObject):
    """Block mouse wheel on spinboxes/sliders/combos, allow on scroll areas."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            w = obj
            while w is not None:
                if isinstance(w, (QAbstractSpinBox, QSlider, QComboBox)):
                    return True  # block
                if isinstance(w, (QScrollArea, QListWidget)):
                    return False  # allow
                w = w.parent()
        return False


# ── Main Settings Window ─────────────────────────────────────────────

class SettingsWindow(QWidget):
    """Non-modal settings panel with tabs for all config sections."""

    config_saved = pyqtSignal(dict)   # emitted with the new config after save+apply
    config_save_only = pyqtSignal(dict)

    def _force_topmost(self):
        try:
            import ctypes
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010)
        except Exception:
            pass

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._original_config = config
        self._working_config = copy.deepcopy(config)

        self.setWindowTitle("设置面板 — Roco Box Detector")
        self.setMinimumSize(620, 520)
        self.resize(640, 580)

        # Always-on-top timer
        self._topmost_timer = QTimer()
        self._topmost_timer.setInterval(2000)
        self._topmost_timer.timeout.connect(self._force_topmost)
        self._topmost_timer.start()

        self._build_ui()
        self._load_all()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Block mouse wheel on spinboxes/sliders/combos
        self._wheel_blocker = _WheelBlocker(self)
        self.installEventFilter(self._wheel_blocker)

        self.tabs = QTabWidget()
        self.anchor_tab = AnchorTab(self._working_config)
        self.subroi_tab = SubRoiTab(self._working_config)
        self.subroi2_tab = SubRoi2Tab(self._working_config)
        self.patterns_tab = PatternsTab(self._working_config)
        self.patterns2_tab = Patterns2Tab(self._working_config)
        self.runtime_tab = RuntimeTab(self._working_config)
        self.runtime_tab.resolution_changed.connect(self._on_resolution_changed)
        self.debug_tab = DebugTab(self._working_config)
        self.overlay_tab = OverlayTab(self._working_config)

        self.tabs.addTab(self.anchor_tab, "Anchor")
        self.tabs.addTab(self.subroi_tab, "黄框1")
        self.tabs.addTab(self.subroi2_tab, "黄框2")
        self.tabs.addTab(self.patterns_tab, "样本1")
        self.tabs.addTab(self.patterns2_tab, "样本2")
        self.tabs.addTab(self.runtime_tab, "Runtime")
        self.tabs.addTab(self.debug_tab, "Debug")
        self.result_text_tab = ResultTextTab(self._working_config)
        self.tabs.addTab(self.overlay_tab, "悬浮窗")
        self.tabs.addTab(self.result_text_tab, "提示文字")

        main_layout.addWidget(self.tabs)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QPushButton("保存并应用")
        apply_btn.setStyleSheet(
            "QPushButton { background: #1a8; color: white; padding: 6px 20px; "
            "border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #2b9; }"
        )
        apply_btn.clicked.connect(self._on_save_apply)
        btn_row.addWidget(apply_btn)

        save_btn = QPushButton("仅保存")
        save_btn.clicked.connect(self._on_save_only)
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.hide)
        btn_row.addWidget(close_btn)

        main_layout.addLayout(btn_row)

    def _on_resolution_changed(self):
        """Propagate resolution preset values to the UI widgets."""
        ac = self._working_config.get("anchor", {})
        self.anchor_tab.scale_min_spin.setValue(ac.get("scale_min", 0.75))
        self.anchor_tab.scale_max_spin.setValue(ac.get("scale_max", 1.25))
        self.anchor_tab.scale_steps_spin.setValue(ac.get("scale_steps", 6))
        # Refresh patterns tabs to show new scale values
        self.patterns_tab._refresh_combo()
        self.patterns2_tab._refresh_combo()

    def _load_all(self):
        """Refresh all tabs from the working config (e.g. after external change)."""
        self._working_config = copy.deepcopy(self._original_config)
        self.anchor_tab.config = self._working_config
        self.subroi_tab.config = self._working_config
        self.subroi2_tab.config = self._working_config
        self.patterns_tab.config = self._working_config
        self.patterns2_tab.config = self._working_config
        self.runtime_tab.config = self._working_config
        self.debug_tab.config = self._working_config
        self.overlay_tab.config = self._working_config
        self.result_text_tab.config = self._working_config
        self.patterns_tab._refresh_combo()
        self.patterns2_tab._refresh_combo()

    def _collect_all(self) -> dict:
        """Gather values from all tabs into the working config, return it."""
        self.anchor_tab.collect(self._working_config)
        self.subroi_tab.collect(self._working_config)
        self.subroi2_tab.collect(self._working_config)
        self.patterns_tab.collect(self._working_config)
        self.patterns2_tab.collect(self._working_config)
        self.runtime_tab.collect(self._working_config)
        self.debug_tab.collect(self._working_config)
        self.overlay_tab.collect(self._working_config)
        self.result_text_tab.collect(self._working_config)
        return self._working_config

    def _on_save_apply(self):
        cfg = self._collect_all()
        CONFIG_PATH = resolve_path("config.json")
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        # Sync original config dict in-place so live components see changes
        self._original_config.clear()
        self._original_config.update(copy.deepcopy(cfg))
        self.config_saved.emit(self._original_config)

    def _on_save_only(self):
        cfg = self._collect_all()
        CONFIG_PATH = resolve_path("config.json")
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.config_save_only.emit(cfg)