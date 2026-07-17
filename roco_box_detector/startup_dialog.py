"""Startup settings dialog — resolution selection."""

import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton,
)
from PyQt5.QtCore import Qt
from image_utils import resolve_path

CONFIG_PATH = resolve_path("config.json")


class StartupDialog(QDialog):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("初始设置 — Roco Box Detector")
        self.setFixedSize(380, 150)
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

        # Resolution scale: base = 1440 (2K), range = ±0.1
        heights = {"720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}
        res_scale = heights.get(res, 1440) / 1440.0
        scale_min = round(res_scale - 0.1, 2)
        scale_max = round(res_scale + 0.1, 2)
        scale_steps = 5

        self.config["game_resolution"] = res
        self.config["capture_mode"] = "solo"

        ac = self.config.setdefault("anchor", {})
        ac["threshold"] = 0.6
        ac["scale_min"], ac["scale_max"], ac["scale_steps"] = scale_min, scale_max, scale_steps

        # Mark startup complete
        self.config["startup_complete"] = True

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        self.accept()
