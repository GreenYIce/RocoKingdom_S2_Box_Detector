"""Full-screen ROI selection via mouse drag. Returns screen coordinates."""

import tkinter as tk
from typing import Optional, Tuple


class ROISelector:
    """
    Full-screen overlay for selecting a region of interest.

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

    def select(self) -> Optional[dict]:
        """Open a full-screen window for ROI selection. Returns ROI dict or None."""
        self._roi = None
        self._cancelled = False

        root = tk.Tk()
        root.title("框选检测区域 (拖拽鼠标框选，Esc取消)")
        root.attributes("-fullscreen", True)
        root.attributes("-alpha", 0.35)
        root.attributes("-topmost", True)
        root.configure(bg="black")

        canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Instructions
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        canvas.create_text(
            screen_w // 2, screen_h // 2 - 40,
            text="拖拽鼠标框选游戏区域\nEsc 取消 | Enter 确认",
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
                outline="#00FF00",
                width=2,
            )

        def on_button_release(event):
            x1 = min(self._start_x, event.x)
            y1 = min(self._start_y, event.y)
            x2 = max(self._start_x, event.x)
            y2 = max(self._start_y, event.y)
            w = x2 - x1
            h = y2 - y1
            if w > 10 and h > 10:
                self._roi = {"left": x1, "top": y1, "width": w, "height": h}
                root.destroy()
            else:
                # Too small, reset
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
