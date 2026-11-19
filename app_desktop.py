"""Standalone Native Desktop Application for Sports Action Recognition.

Runs directly on Windows (zero browser / web server required).
Features:
- File Dialog to browse and select any video (MP4, AVI, MOV, MKV)
- Quick-pick random sample from benchmark test set
- Multi-stream AI Action Classification (Visual, Pose, Object, HOI)
- Displays Top-5 predicted actions with confidence scores
- Embedded interactive OpenCV playback of annotated output video (17 skeletal joints + equipment boxes)
"""

from pathlib import Path
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import SportsActionInferenceEngine as ActionInferenceEngine


class SportsActionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sports Action Recognition & HOI Kinematics")
        self.root.geometry("780x620")
        self.root.minsize(700, 550)

        # Apply system style
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.engine: Optional[ActionInferenceEngine] = None
        self.selected_video_path: Optional[Path] = None
        self.last_result = None

        self._build_ui()
        self._init_engine_async()

    def _build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#1E3A8A", height=75)
        header.pack(fill="x", side="top")

        lbl_title = tk.Label(
            header,
            text="🏋️‍♂️ Sports Action Recognition Studio",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#1E3A8A",
        )
        lbl_title.pack(anchor="w", padx=20, pady=(12, 2))

        lbl_sub = tk.Label(
            header,
            text="Human–Object Interaction (HOI) 4-Stream Deep Learning Suite · 82 Sports Categories",
            font=("Segoe UI", 9),
            fg="#93C5FD",
            bg="#1E3A8A",
        )
        lbl_sub.pack(anchor="w", padx=20, pady=(0, 10))

        # Main Container
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill="both", expand=True)

        # File Selection Frame
        grp_file = ttk.LabelFrame(main_frame, text="1. Select Video for Analysis", padding=12)
        grp_file.pack(fill="x", pady=(0, 12))

        btn_box = ttk.Frame(grp_file)
        btn_box.pack(fill="x")

        self.btn_browse = ttk.Button(
            btn_box,
            text="📂 Browse & Upload Video File...",
            command=self._on_browse,
        )
        self.btn_browse.pack(side="left", padx=(0, 8))

        self.btn_sample = ttk.Button(
            btn_box,
            text="🎲 Test Random Benchmark Video",
            command=self._on_random_sample,
        )
        self.btn_sample.pack(side="left")

        self.lbl_selected_file = ttk.Label(
            grp_file,
            text="No video selected yet. Please upload or choose a sample.",
            font=("Segoe UI", 9, "italic"),
            foreground="#4B5563",
        )
        self.lbl_selected_file.pack(anchor="w", pady=(8, 0))

        # Run Action Section
        grp_action = ttk.Frame(main_frame)
        grp_action.pack(fill="x", pady=(0, 12))

        self.btn_run = ttk.Button(
            grp_action,
            text="🚀 Run Sports Action Recognition",
            command=self._on_run_analysis,
            state="disabled",
        )
        self.btn_run.pack(side="left", ipady=4, padx=(0, 12))

        self.progress_bar = ttk.Progressbar(grp_action, mode="indeterminate", length=250)

        self.lbl_status = ttk.Label(
            grp_action,
            text="Ready.",
            font=("Segoe UI", 9),
            foreground="#2563EB",
        )
        self.lbl_status.pack(side="left")

        # Results Display Frame
        self.grp_results = ttk.LabelFrame(main_frame, text="2. Detection & Prediction Outputs", padding=12)
        self.grp_results.pack(fill="both", expand=True)

        # Top Banner
        top_banner = tk.Frame(self.grp_results, bg="#F3F4F6", height=60, bd=1, relief="solid")
        top_banner.pack(fill="x", pady=(0, 10))

        self.lbl_pred_action = tk.Label(
            top_banner,
            text="Predicted Action: ---",
            font=("Segoe UI", 14, "bold"),
            bg="#F3F4F6",
            fg="#111827",
        )
        self.lbl_pred_action.pack(anchor="w", padx=14, pady=(8, 2))

        self.lbl_confidence = tk.Label(
            top_banner,
            text="Confidence: ---% | Inference Latency: ---s",
            font=("Segoe UI", 10),
            bg="#F3F4F6",
            fg="#059669",
        )
        self.lbl_confidence.pack(anchor="w", padx=14, pady=(0, 8))

        # Top 5 Rankings Table
        ttk.Label(
            self.grp_results,
            text="Top-5 Action Candidates:",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        columns = ("rank", "category", "prob")
        self.tree = ttk.Treeview(self.grp_results, columns=columns, show="headings", height=5)
        self.tree.heading("rank", text="Rank")
        self.tree.heading("category", text="Sport Action Category")
        self.tree.heading("prob", text="Confidence Probability")
        self.tree.column("rank", width=60, anchor="center")
        self.tree.column("category", width=360, anchor="w")
        self.tree.column("prob", width=180, anchor="center")
        self.tree.pack(fill="x", pady=(0, 10))

        # Action Buttons for Output
        bottom_box = ttk.Frame(self.grp_results)
        bottom_box.pack(fill="x")

        self.btn_play = ttk.Button(
            bottom_box,
            text="▶️ Play Annotated AI Video (Skeleton + Gear Overlay)",
            command=self._on_play_video,
            state="disabled",
        )
        self.btn_play.pack(side="left", padx=(0, 8))

    def _init_engine_async(self):
        def load_worker():
            self.lbl_status.config(text="Loading neural networks onto GPU...")
            default_pth = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pth"
            default_pt = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pt"
            ckpt = str(default_pth) if default_pth.exists() else (str(default_pt) if default_pt.exists() else None)
            self.engine = ActionInferenceEngine(checkpoint_path=ckpt)
            self.lbl_status.config(text="Models initialized and ready.")

        t = threading.Thread(target=load_worker, daemon=True)
        t.start()

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="Select Sports Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self._set_selected_video(Path(path))

    def _on_random_sample(self):
        test_dir = PROJECT_ROOT / "Model Data" / "Test"
        if not test_dir.exists():
            test_dir = PROJECT_ROOT / "ucf action"
        videos = list(test_dir.rglob("*.avi")) + list(test_dir.rglob("*.mp4"))
        if videos:
            sample = random.choice(videos)
            self._set_selected_video(sample)
        else:
            messagebox.showwarning("No Samples", "No test videos found in Model Data/Test!")

    def _set_selected_video(self, path: Path):
        self.selected_video_path = path
        self.lbl_selected_file.config(text=f"Selected: {path.parent.name}/{path.name}")
        self.btn_run.config(state="normal")
        self.lbl_pred_action.config(text="Predicted Action: ---")
        self.lbl_confidence.config(text="Confidence: ---% | Inference Latency: ---s")
        self.btn_play.config(state="disabled")
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _on_run_analysis(self):
        if not self.selected_video_path or not self.selected_video_path.exists():
            messagebox.showerror("Error", "Video file not found!")
            return

        if not self.engine:
            messagebox.showinfo("Wait", "Neural networks are still initializing. Please wait a few seconds.")
            return

        self.btn_run.config(state="disabled")
        self.btn_browse.config(state="disabled")
        self.btn_sample.config(state="disabled")
        self.progress_bar.pack(side="left", padx=(0, 10))
        self.progress_bar.start(10)
        self.lbl_status.config(text="Extracting 4 streams & running classifier...")

        def worker():
            try:
                t0 = time.time()
                result = self.engine.predict_video(self.selected_video_path, topk=5)
                elapsed = time.time() - t0
                self.root.after(0, self._on_analysis_complete, result, elapsed)
            except Exception as e:
                self.root.after(0, self._on_analysis_error, str(e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_analysis_complete(self, result, elapsed: float):
        self.last_result = result
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run.config(state="normal")
        self.btn_browse.config(state="normal")
        self.btn_sample.config(state="normal")
        self.btn_play.config(state="normal")
        self.lbl_status.config(text=f"Done in {elapsed:.2f}s!")

        # Update Top Banner
        self.lbl_pred_action.config(text=f"Predicted Action: {result['top_class']}")
        self.lbl_confidence.config(
            text=f"Confidence: {result['top_confidence']:.2f}% | Latency: {elapsed:.2f}s | Keyframes: {len(result['raw_frames'])}"
        )

        # Populate Top-5 Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        for rank, (cls_name, prob) in enumerate(result["predictions"], start=1):
            bar = "█" * int(prob / 5)
            self.tree.insert("", "end", values=(f"#{rank}", cls_name, f"{prob:5.1f}%  {bar}"))

    def _on_analysis_error(self, err_msg: str):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run.config(state="normal")
        self.btn_browse.config(state="normal")
        self.btn_sample.config(state="normal")
        self.lbl_status.config(text="Analysis failed.")
        messagebox.showerror("Inference Error", f"Failed to analyze video:\n{err_msg}")

    def _on_play_video(self):
        if not self.last_result or "annotated_frames" not in self.last_result:
            return

        frames = self.last_result["annotated_frames"]
        window_title = f"AI Annotated Output: {self.last_result['top_class']}"

        def play_loop():
            delay = 125  # ~8 FPS
            idx = 0
            total = len(frames)
            while True:
                bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
                cv2.imshow(window_title, bgr)
                key = cv2.waitKey(delay) & 0xFF
                if key in [ord("q"), 27]:
                    break
                idx = (idx + 1) % total

            cv2.destroyWindow(window_title)

        t = threading.Thread(target=play_loop, daemon=True)
        t.start()


def main():
    root = tk.Tk()
    app = SportsActionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
