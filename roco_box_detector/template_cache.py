"""Template reading and caching. Loads all templates once at startup, pre-scales variants."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    """Loads and caches all templates at startup. Variants are pre-scaled once."""

    def __init__(self, config: dict):
        self._config = config
        self.anchor_group: Optional[TemplateGroup] = None
        self.pattern_groups: Dict[str, TemplateGroup] = {}
        self.pattern_groups_2: Dict[str, TemplateGroup] = {}
        self._load_all(config)

    # ── public ──────────────────────────────────────────────────────────

    def get_anchor_templates(self) -> Optional[TemplateGroup]:
        return self.anchor_group

    def get_pattern_groups(self) -> Dict[str, TemplateGroup]:
        return self.pattern_groups

    def get_pattern_groups_2(self) -> Dict[str, TemplateGroup]:
        return self.pattern_groups_2

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_group.items) if self.anchor_group else 0

    @property
    def pattern_count(self) -> int:
        return sum(len(g.items) for g in self.pattern_groups.values())

    @property
    def pattern_count_2(self) -> int:
        return sum(len(g.items) for g in self.pattern_groups_2.values())

    def reload(self, config: dict) -> None:
        self._config = config
        self.anchor_group = None
        self.pattern_groups.clear()
        self.pattern_groups_2.clear()
        self._load_all(config)
        cnt2 = self.pattern_count_2
        print(f"[Cache] Reloaded: {self.anchor_count} anchor templates, "
              f"{self.pattern_count} p1 templates across "
              f"{len(self.pattern_groups)} groups"
              + (f", {cnt2} p2 templates across "
                 f"{len(self.pattern_groups_2)} groups" if cnt2 else ""))

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

    # ── internal ────────────────────────────────────────────────────────

    def _load_all(self, config: dict) -> None:
        # Anchor: deferred pre-scale (scales depend on ROI)
        self.anchor_group = self._load_group(
            config["anchor"]["templates"],
            config["anchor"]["label"],
            config["anchor"]["threshold"],
            config["anchor"]["scale_min"],
            config["anchor"]["scale_max"],
            config["anchor"]["scale_steps"],
            config["anchor"]["use_grayscale"],
            with_mask=False,
            pre_scale=False,
            preprocess_mode=config["anchor"].get("preprocess_mode", "none"),
            gamma=config["anchor"].get("gamma", 0.75),
        )

        for name, pcfg in config.get("patterns", {}).items():
            group = self._load_group(
                pcfg["templates"], name, pcfg["threshold"],
                pcfg["scale_min"], pcfg["scale_max"], pcfg["scale_steps"],
                pcfg["use_grayscale"],
                with_mask=True, pre_scale=True,
            )
            self.pattern_groups[name] = group

        for name, pcfg in config.get("patterns_2", {}).items():
            group = self._load_group(
                pcfg["templates"], name, pcfg["threshold"],
                pcfg["scale_min"], pcfg["scale_max"], pcfg["scale_steps"],
                pcfg["use_grayscale"],
                with_mask=True, pre_scale=True,
            )
            self.pattern_groups_2[name] = group

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
