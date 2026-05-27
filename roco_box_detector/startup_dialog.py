"""Startup settings dialog — resolution + solo/duo selection."""

import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QRadioButton, QPushButton, QButtonGroup, QGroupBox,
)
from PyQt5.QtCore import Qt
from image_utils import resolve_path

CONFIG_PATH = resolve_path("config.json")


class StartupDialog(QDialog):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("初始设置 — Roco Box Detector")
        self.setFixedSize(380, 220)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Dialog)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Resolution
        res_label = QLabel("游戏分辨率：")
        res_label.setStyleSheet("font-size: 13px; color: #ccc;")
        layout.addWidget(res_label)

        self.res_combo = QComboBox()
        self.res_combo.addItems(["720p", "1080p", "2K", "4K"])
        cur_res = self.config.get("game_resolution", "2K")
        self.res_combo.setCurrentText(cur_res)
        self.res_combo.setStyleSheet(
            "QComboBox { font-size: 13px; padding: 4px 8px; "
            "background: #222; color: #fff; border: 1px solid #555; border-radius: 4px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #222; color: #fff; }")
        layout.addWidget(self.res_combo)

        # Solo / Duo
        mode_group = QGroupBox("抓捕模式")
        mode_group.setStyleSheet(
            "QGroupBox { color: #ccc; font-size: 13px; border: 1px solid #555; "
            "border-radius: 6px; margin-top: 8px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }")
        mode_layout = QHBoxLayout(mode_group)

        self.solo_btn = QRadioButton("单人抓捕")
        self.solo_btn.setStyleSheet("color: #aaa; font-size: 13px;")
        self.solo_btn.setChecked(True)
        self.duo_btn = QRadioButton("双人抓捕")
        self.duo_btn.setStyleSheet("color: #aaa; font-size: 13px;")

        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.solo_btn, 0)
        self._mode_group.addButton(self.duo_btn, 1)

        mode_layout.addWidget(self.solo_btn)
        mode_layout.addWidget(self.duo_btn)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        layout.addStretch()

        # OK button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("开始使用")
        ok_btn.setStyleSheet(
            "QPushButton { background: #1a8; color: white; padding: 6px 30px; "
            "border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #2b9; }")
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _on_ok(self):
        res = self.res_combo.currentText()
        is_duo = self.duo_btn.isChecked()

        # Apply resolution preset
        presets = {
            "720p":  {"anchor": (0.55, 0.65, 4), "pattern": (0.35, 0.75, 6)},
            "1080p": {"anchor": (0.75, 0.90, 5), "pattern": (0.50, 1.05, 7)},
            "2K":    {"anchor": (0.90, 1.25, 5), "pattern": (0.65, 1.35, 8)},
            "4K":    {"anchor": (1.35, 1.80, 5), "pattern": (1.00, 1.70, 8)},
        }
        p = presets.get(res, presets["2K"])
        self.config["game_resolution"] = res

        ac = self.config.setdefault("anchor", {})
        ac["threshold"] = 0.75
        ac["scale_min"], ac["scale_max"], ac["scale_steps"] = p["anchor"]

        self.config["capture_mode"] = "duo" if is_duo else "solo"

        # Duo: reduce anchor scale by 0.1
        if is_duo:
            ac["scale_min"] = max(0.3, ac["scale_min"] - 0.1)
            ac["scale_max"] = max(ac["scale_min"] + 0.1, ac["scale_max"] - 0.1)

        # Update pattern groups
        for pk in ("patterns", "patterns_2"):
            for _, pcfg in self.config.get(pk, {}).items():
                pcfg["scale_min"], pcfg["scale_max"], pcfg["scale_steps"] = p["pattern"]

        # Mark startup complete
        self.config["startup_complete"] = True

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        self.accept()
