"""Template reading and caching. Loads anchor templates once at startup, pre-scales variants."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from image_utils import (
    imread_chinese, preprocess_image, make_gaussian_mask,
    safe_resize_template, safe_resize_mask,
)


@dataclass
class ScaledVariant:
    """One pre-scaled copy of a template, ready for cv2.matchTemplate."""
    scale: float
    scaled_gray: np.ndarray
    scaled_mask: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0


@dataclass
class TemplateItem:
    path: str
    label: str
    image_color: np.ndarray
    image_gray: np.ndarray
    mask: Optional[np.ndarray] = None
    scaled_variants: List[ScaledVariant] = field(default_factory=list)


@dataclass
class TemplateGroup:
    label: str
    threshold: float
    scale_min: float
    scale_max: float
    scale_steps: int
    use_grayscale: bool
    items: List[TemplateItem] = field(default_factory=list)
    preprocess_mode: str = "none"
    gamma: float = 0.75


class TemplateCache:
    """Loads and caches anchor templates at startup. Variants are pre-scaled once."""

    def __init__(self, config: dict):
        self._config = config
        self.anchor_group: Optional[TemplateGroup] = None
        self.icon_group: Optional[TemplateGroup] = None
        self.keyword_group: Optional[TemplateGroup] = None
        self._load_anchor(config)
        self._load_icon(config)
        self._load_keywords(config)

    # ── public ──────────────────────────────────────────────────────────

    def get_anchor_templates(self) -> Optional[TemplateGroup]:
        return self.anchor_group

    def get_icon_template(self) -> Optional[TemplateItem]:
        """Return the first icon template, or None if not loaded."""
        if self.icon_group and self.icon_group.items:
            return self.icon_group.items[0]
        return None

    def get_keyword_templates(self) -> Optional[TemplateGroup]:
        """Return keyword pattern templates, or None if not loaded."""
        return self.keyword_group

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_group.items) if self.anchor_group else 0

    def reload(self, config: dict) -> None:
        self._config = config
        self.anchor_group = None
        self.icon_group = None
        self.keyword_group = None
        self._load_anchor(config)
        self._load_icon(config)
        self._load_keywords(config)
        icon_info = f", {len(self.icon_group.items)} icon" if self.icon_group and self.icon_group.items else ""
        kw_count = len(self.keyword_group.items) if self.keyword_group and self.keyword_group.items else 0
        kw_info = f", {kw_count} keywords" if kw_count else ""
        print(f"[Cache] Reloaded: {self.anchor_count} anchor templates{icon_info}{kw_info}")

    def rescale_anchor(self, roi_width: int, norm_width: int) -> None:
        """Re-pre-scale anchor templates for the current ROI (adaptive scale)."""
        if self.anchor_group is None or not self.anchor_group.items:
            return
        cfg = self._config["anchor"]
        sf = norm_width / roi_width if roi_width > 0 else 1.0
        smin = cfg["scale_min"] * sf
        smax = cfg["scale_max"] * sf
        steps = cfg["scale_steps"]
        self._pre_scale_items(self.anchor_group.items, smin, smax, steps)
        total = sum(len(it.scaled_variants) for it in self.anchor_group.items)
        print(f"[Cache] Anchor re-scaled: roi_w={roi_width} "
              f"scale=[{smin:.3f},{smax:.3f}]x{steps} → {total} variants")

    def rescale_icon(self, resolution_scale: float) -> None:
        """Re-pre-scale icon templates for the current resolution."""
        if self.icon_group is None or not self.icon_group.items:
            return
        cfg = self._config["icon_detection"]
        smin = cfg["scale_min"] * resolution_scale
        smax = cfg["scale_max"] * resolution_scale
        steps = cfg["scale_steps"]
        self._pre_scale_items(self.icon_group.items, smin, smax, steps)
        total = sum(len(it.scaled_variants) for it in self.icon_group.items)
        print(f"[Cache] Icon re-scaled: res={resolution_scale:.2f} "
              f"scale=[{smin:.3f},{smax:.3f}]x{steps} → {total} variants")

    def rescale_keywords(self, resolution_scale: float) -> None:
        """Re-pre-scale keyword templates for the current resolution."""
        if self.keyword_group is None or not self.keyword_group.items:
            return
        cfg = self._config["keyword_detection"]
        smin = cfg["scale_min"] * resolution_scale
        smax = cfg["scale_max"] * resolution_scale
        steps = cfg["scale_steps"]
        self._pre_scale_items(self.keyword_group.items, smin, smax, steps)
        total = sum(len(it.scaled_variants) for it in self.keyword_group.items)
        print(f"[Cache] Keywords re-scaled: res={resolution_scale:.2f} "
              f"scale=[{smin:.3f},{smax:.3f}]x{steps} → {total} variants")

    # ── internal ────────────────────────────────────────────────────────

    def _load_anchor(self, config: dict) -> None:
        ac = config["anchor"]
        self.anchor_group = self._load_group(
            ac["templates"], ac["label"], ac["threshold"],
            ac["scale_min"], ac["scale_max"], ac["scale_steps"],
            ac["use_grayscale"],
            with_mask=False, pre_scale=False,
            preprocess_mode=ac.get("preprocess_mode", "none"),
            gamma=ac.get("gamma", 0.75),
        )

    def _load_icon(self, config: dict) -> None:
        """Load icon template(s) — single-scale, no pre-scaling needed."""
        ic = config.get("icon_detection", {})
        if not ic.get("enabled", False):
            self.icon_group = None
            return
        templates = ic.get("templates", [])
        if not templates:
            print("[Cache] Icon detection enabled but no templates configured")
            self.icon_group = None
            return
        self.icon_group = self._load_group(
            [templates[0]],  # single template for detection
            "icon",
            ic.get("threshold", 0.75),
            ic.get("scale_min", 0.8),
            ic.get("scale_max", 1.2),
            ic.get("scale_steps", 5),
            ic.get("use_grayscale", True),
            with_mask=False,
            pre_scale=True,
            preprocess_mode=ic.get("preprocess_mode", "none"),
            gamma=ic.get("gamma", 0.75),
        )
        if not self.icon_group or not self.icon_group.items:
            print(f"[Cache] Icon template failed to load: {templates[0]}")
        else:
            item = self.icon_group.items[0]
            print(f"[Cache] Icon template loaded: {templates[0]} "
                  f"({item.image_gray.shape[1]}x{item.image_gray.shape[0]})")

    def _load_keywords(self, config: dict) -> None:
        """Load keyword pattern templates. Each keyword has paths + output label.
        TemplateItem.label = output label so MatchResult.template_label is the result."""
        kd = config.get("keyword_detection", {})
        if not kd.get("enabled", False):
            self.keyword_group = None
            return
        tmpl_cfg = kd.get("templates", {})
        if not tmpl_cfg:
            self.keyword_group = None
            return
        all_items = []
        for keyword_name, cfg in tmpl_cfg.items():
            output_label = cfg.get("label", keyword_name)
            for p in cfg.get("paths", []):
                item = self._load_item(
                    p, label=output_label,
                    use_grayscale=kd.get("use_grayscale", True),
                    preprocess_mode=kd.get("preprocess_mode", "none"),
                    gamma=kd.get("gamma", 0.75),
                )
                if item is not None:
                    all_items.append(item)
                else:
                    print(f"[WARN] Failed to load keyword template: {p}")
        if not all_items:
            self.keyword_group = None
            print("[Cache] Keywords: none loaded")
            return
        self.keyword_group = TemplateGroup(
            label="keywords",
            threshold=kd.get("threshold", 0.80),
            scale_min=kd.get("scale_min", 0.80),
            scale_max=kd.get("scale_max", 1.20),
            scale_steps=kd.get("scale_steps", 5),
            use_grayscale=kd.get("use_grayscale", True),
            items=all_items,
            preprocess_mode=kd.get("preprocess_mode", "none"),
            gamma=kd.get("gamma", 0.75),
        )
        self._pre_scale_items(
            all_items,
            kd.get("scale_min", 0.80),
            kd.get("scale_max", 1.20),
            kd.get("scale_steps", 5),
        )
        labels = set(it.label for it in all_items)
        print(f"[Cache] Keywords loaded: {', '.join(sorted(labels))}")

    def _load_group(
        self, paths, label, threshold, scale_min, scale_max, scale_steps,
        use_grayscale, with_mask=False, pre_scale=True,
        preprocess_mode="none", gamma=0.75,
    ) -> TemplateGroup:
        group = TemplateGroup(
            label=label, threshold=threshold,
            scale_min=scale_min, scale_max=scale_max, scale_steps=scale_steps,
            use_grayscale=use_grayscale,
            preprocess_mode=preprocess_mode, gamma=gamma,
        )
        for p in paths:
            item = self._load_item(p, label, use_grayscale,
                                   with_mask, preprocess_mode, gamma)
            if item is not None:
                group.items.append(item)
            else:
                print(f"[WARN] Failed to load template: {p}")
        if pre_scale and group.items:
            self._pre_scale_items(group.items, scale_min, scale_max, scale_steps)
        return group

    def _load_item(
        self, path, label, use_grayscale, with_mask=False,
        preprocess_mode="none", gamma=0.75,
    ) -> Optional[TemplateItem]:
        img = imread_chinese(path)
        if img is None:
            return None
        gray = preprocess_image(img, use_grayscale=True,
                                preprocess_mode=preprocess_mode, gamma=gamma)
        mask = make_gaussian_mask(img.shape[1], img.shape[0]) if with_mask else None
        return TemplateItem(path=path, label=label, image_color=img,
                            image_gray=gray, mask=mask)

    @staticmethod
    def _pre_scale_items(
        items: List[TemplateItem],
        scale_min: float, scale_max: float, scale_steps: int,
    ) -> None:
        for item in items:
            item.scaled_variants.clear()
        for s in np.linspace(scale_min, scale_max, scale_steps):
            for item in items:
                sg = safe_resize_template(item.image_gray, s)
                if sg is None or sg.size == 0:
                    continue
                sm = safe_resize_mask(item.mask, s) if item.mask is not None else None
                item.scaled_variants.append(ScaledVariant(
                    scale=float(s), scaled_gray=sg, scaled_mask=sm,
                    width=sg.shape[1], height=sg.shape[0]))
