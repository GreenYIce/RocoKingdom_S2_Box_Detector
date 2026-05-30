"""Per-screen ROI selection via mouse drag. Works across multiple monitors."""

import tkinter as tk
import ctypes
from ctypes import wintypes
from typing import Optional


class ROISelector:
    """Screen overlay for selecting a region of interest on the monitor the mouse is on.

    Usage:
        selector = ROISelector()
        roi = selector.select()  # blocks until user selects or cancels
        # roi is {"left": x, "top": y, "width": w, "height": h} or None
    """

    def __init__(self):
        self._roi: Optional[dict] = None
        self._start_x = 0
        self._start_y = 0
        self._rect_id = None
        self._cancelled = False

    # ── per-monitor geometry ─────────────────────────────────────────

    @staticmethod
    def _get_pointer_screen():
        """Return (left, top, width, height) of the monitor under the cursor."""

        # Get all monitors via Win32
        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        monitors = []

        def callback(hMonitor, hdc, rect, _param):
            r = rect.contents
            monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
            return True

        MonitorEnumProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC,
            ctypes.POINTER(RECT), wintypes.LPARAM,
        )
        ctypes.windll.user32.EnumDisplayMonitors(
            None, None, MonitorEnumProc(callback), 0)

        # Get cursor position
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))

        # Find monitor containing cursor
        for left, top, w, h in monitors:
            if left <= pt.x < left + w and top <= pt.y < top + h:
                return left, top, w, h

        # Fallback: primary monitor
        return monitors[0] if monitors else (0, 0, 1920, 1080)

    def select(self) -> Optional[dict]:
        """Open an overlay on the monitor under the cursor. Returns ROI dict or None."""
        self._roi = None
        self._cancelled = False

        screen_left, screen_top, screen_w, screen_h = self._get_pointer_screen()

        root = tk.Tk()
        root.title("框选检测区域")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.35)
        root.configure(bg="black")

        # Position exactly on the target monitor
        geo_str = f"{screen_w}x{screen_h}+{screen_left}+{screen_top}"
        root.geometry(geo_str)

        canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Instructions — centered on THIS screen
        canvas.create_text(
            screen_w // 2, screen_h // 2 - 40,
            text="拖拽鼠标框选游戏区域\nEsc 取消 | Enter 确认 (或松手确认)",
            fill="white",
            font=("Microsoft YaHei", 18),
            justify=tk.CENTER,
        )

        def on_button_press(event):
            self._start_x = event.x
            self._start_y = event.y
            if self._rect_id is not None:
                canvas.delete(self._rect_id)

        def on_move(event):
            if self._rect_id is not None:
                canvas.delete(self._rect_id)
            self._rect_id = canvas.create_rectangle(
                self._start_x, self._start_y, event.x, event.y,
                outline="#00FF00", width=2)

        def on_button_release(event):
            x1 = min(self._start_x, event.x)
            y1 = min(self._start_y, event.y)
            x2 = max(self._start_x, event.x)
            y2 = max(self._start_y, event.y)
            w = x2 - x1
            h = y2 - y1
            if w > 10 and h > 10:
                # Map window-relative coords to absolute screen coords
                self._roi = {
                    "left": screen_left + x1,
                    "top": screen_top + y1,
                    "width": w,
                    "height": h,
                }
                root.destroy()
            else:
                if self._rect_id is not None:
                    canvas.delete(self._rect_id)
                    self._rect_id = None

        def on_key(event):
            if event.keysym == "Escape":
                self._cancelled = True
                root.destroy()
            elif event.keysym == "Return" and self._roi is not None:
                root.destroy()

        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_button_release)
        root.bind("<Key>", on_key)

        root.mainloop()

        if self._cancelled:
            return None
        return self._roi
