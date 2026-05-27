"""Generate side-by-side comparison: Original | Grayscale | Canny edge for all templates."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # resolve_path depends on CWD
sys.path.insert(0, HERE)

import cv2
from image_utils import resolve_path, preprocess_image, imread_chinese

os.makedirs("temp_preview", exist_ok=True)

with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

all_paths = []
for pname, pcfg in cfg.get("patterns", {}).items():
    for t in pcfg["templates"]:
        all_paths.append((pname, os.path.join(HERE, t.replace("/", os.sep))))
for pname, pcfg in cfg.get("patterns_2", {}).items():
    for t in pcfg["templates"]:
        all_paths.append(("p2_" + pname, os.path.join(HERE, t.replace("/", os.sep))))

for label, path in all_paths:
    if not os.path.exists(path):
        print(f"  [SKIP] {path}")
        continue
    img = imread_chinese(path)
    if img is None:
        continue

    gray = preprocess_image(img, use_grayscale=True, use_canny=False)
    canny = preprocess_image(img, use_grayscale=False, use_canny=True)

    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canny3 = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    row1 = np.hstack([img, gray3, canny3])
    h, w = row1.shape[:2]
    LH = 30
    canvas = np.zeros((h + LH, w, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)
    canvas[LH : LH + h, 0:w] = row1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (10, 22), font, 0.6, (200, 200, 200), 1)
    col2 = img.shape[1]
    cv2.putText(canvas, "Grayscale (CCOEFF_NORMED)", (col2 + 10, 22), font, 0.6, (200, 200, 200), 1)
    col3 = img.shape[1] * 2
    cv2.putText(canvas, "Canny Edge", (col3 + 10, 22), font, 0.6, (200, 200, 200), 1)

    name = os.path.splitext(os.path.basename(path))[0]
    out = f"temp_preview/{label}_{name}.png"
    # Use imencode to avoid Chinese path issue
    _, buf = cv2.imencode(".png", canvas)
    buf.tofile(out)
    print(f"  {out}")

print(f"\nDone — {len(all_paths)} files in temp_preview/")
