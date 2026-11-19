"""Interactive Live Real-Time & Video Validation Demo for Sports Action Recognition.

Runs on local machine (VS Code / PowerShell):
- Live Webcam Scanning: python demo_realtime.py --camera 0
- Custom Video File Test: python demo_realtime.py --video path/to/sport_video.mp4
- Random Test Video Sample: python demo_realtime.py --sample
"""

import argparse
from pathlib import Path
import queue
import random
import sys
import threading
import time

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ActionInferenceEngine


def play_annotated_stream(
    annotated_frames: list,
    window_name: str = "Sports Action Real-Time Validation",
    fps: float = 10.0,
    loop: bool = True,
) -> None:
    """Plays an annotated sequence in an interactive OpenCV window."""
    delay = int(1000.0 / fps)
    print(f"\n[INFO] Playing annotated sequence in window '{window_name}'.")
    print("       Press 'q' or 'ESC' to close window | SPACE to pause/resume.")

    paused = False
    idx = 0
    total = len(annotated_frames)

    while True:
        frame_rgb = annotated_frames[idx]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, frame_bgr)

        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in [ord("q"), 27]:  # 'q' or ESC
            break
        elif key == ord(" "):       # SPACE to pause
            paused = not paused

        if not paused:
            idx += 1
            if idx >= total:
                if loop:
                    idx = 0
                else:
                    break

    cv2.destroyAllWindows()


def run_live_webcam(
    engine: ActionInferenceEngine,
    camera_id: int = 0,
    buffer_size: int = 16,
) -> None:
    """Asynchronous live scanning loop for webcam validation at smooth 30+ FPS."""
    print(f"\n--> Opening Webcam (Device Index: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}. Please check connection or permissions.")
        return

    window_name = f"Live Sports Action Scanner (Camera {camera_id})"
    print(f"--> Camera ready! Buffering {buffer_size} frames for real-time action recognition.")
    print("    Press 'q' or 'ESC' to exit | 'c' to print current prediction to console.")

    frame_buffer = []
    task_queue = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    latest_result = {
        "top_class": "Scanning motion...",
        "top_confidence": 0.0,
        "predictions": [],
        "is_inferring": False,
    }
    stop_event = threading.Event()

    def inference_worker():
        while not stop_event.is_set():
            try:
                frames_to_process = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                res = engine.predict_video(frames_to_process, topk=3)
                with result_lock:
                    latest_result["top_class"] = res["top_class"]
                    latest_result["top_confidence"] = res["top_confidence"]
                    latest_result["predictions"] = res["predictions"]
                    latest_result["is_inferring"] = False
            except Exception as e:
                with result_lock:
                    latest_result["is_inferring"] = False
            finally:
                task_queue.task_done()

    worker_thread = threading.Thread(target=inference_worker, daemon=True)
    worker_thread.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab frame from camera.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb_frame)
            if len(frame_buffer) > buffer_size * 2:
                frame_buffer = frame_buffer[-buffer_size:]

            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # Trigger background prediction when buffer has enough frames and worker is idle
            if len(frame_buffer) >= buffer_size and task_queue.empty():
                with result_lock:
                    is_busy = latest_result["is_inferring"]
                if not is_busy:
                    with result_lock:
                        latest_result["is_inferring"] = True
                    task_queue.put(list(frame_buffer[-buffer_size:]))

            # Retrieve current prediction data
            with result_lock:
                pred_class = latest_result["top_class"]
                pred_conf = latest_result["top_confidence"]
                top_preds = list(latest_result["predictions"])
                inferring = latest_result["is_inferring"]

            # Draw HUD overlay
            hud_h = 105
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, hud_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

            status_color = (0, 165, 255) if inferring else (0, 255, 0)
            cv2.putText(
                display_frame,
                f"ACTION: {pred_class}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Confidence: {pred_conf:.1f}%",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
                cv2.LINE_AA,
            )

            if top_preds:
                c_text = " | ".join([f"{c}: {p:.0f}%" for c, p in top_preds[:3]])
                cv2.putText(
                    display_frame,
                    c_text,
                    (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("c"):
                print(f"[LIVE CAPTURE] Predicted Action: {pred_class} ({pred_conf:.1f}%)")
                if top_preds:
                    for r, (c, p) in enumerate(top_preds, 1):
                        print(f"   {r}. {c}: {p:.1f}%")

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Sports Action Validation Engine")
    parser.add_argument("--video", type=str, default=None, help="Path to custom video file to test")
    parser.add_argument("--camera", type=int, default=None, help="Webcam device index (e.g. 0)")
    parser.add_argument("--sample", action="store_true", help="Pick a random test video from Model Data/Test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Optional output path to save annotated MP4")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument("--no-gui", action="store_true", help="Do not open GUI window (headless/server mode)")
    args = parser.parse_args()

    # Determine execution target
    target_video = None
    if args.sample:
        test_dir = PROJECT_ROOT / "Model Data" / "Test"
        if not test_dir.exists():
            test_dir = PROJECT_ROOT / "ucf action"
        test_videos = list(test_dir.rglob("*.avi")) + list(test_dir.rglob("*.mp4"))
        if not test_videos:
            print("[ERROR] No sample videos found in Model Data/Test!")
            return
        target_video = random.choice(test_videos)
        print(f"\n--> Random Sample Selected: {target_video.parent.name}/{target_video.name}")
    elif args.video:
        target_video = Path(args.video)
        if not target_video.exists():
            # Search for filename across Model Data and ucf action
            search_name = target_video.name
            matches = list(PROJECT_ROOT.glob(f"**/{search_name}"))
            if matches:
                target_video = matches[0]
                print(f"--> Located video file at: {target_video}")
            else:
                print(f"[ERROR] Video file not found: {target_video}")
                return

    # Checkpoint resolution
    ckpt_path = args.checkpoint
    if not ckpt_path:
        default_pth = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pth"
        default_pt = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pt"
        if default_pth.exists():
            ckpt_path = str(default_pth)
        elif default_pt.exists():
            ckpt_path = str(default_pt)

    # Initialize Engine
    print("=" * 65)
    print("SPORTS ACTION REAL-TIME VALIDATION ENGINE")
    print("=" * 65)
    engine = ActionInferenceEngine(checkpoint_path=ckpt_path)

    # 1. Live Webcam Mode
    if args.camera is not None:
        run_live_webcam(engine, camera_id=args.camera)
        return

    # 2. Video File Validation Mode
    if target_video is not None:
        print(f"\n--> Processing Video: {target_video.name}")
        t0 = time.time()
        result = engine.predict_video(target_video, topk=args.topk)
        elapsed = time.time() - t0

        print(f"\n{'='*45}")
        print(f"PREDICTED ACTION: {result['top_class']}")
        print(f"CONFIDENCE      : {result['top_confidence']:.2f}%")
        print(f"INFERENCE TIME  : {elapsed:.2f} seconds")
        print(f"{'='*45}")
        print("\nTop Predicted Candidates:")
        for rank, (cls_name, prob) in enumerate(result["predictions"], start=1):
            bar = "█" * int(prob / 5)
            print(f"  {rank}. {cls_name:<24} : {prob:5.1f}% {bar}")

        # Save annotated video if requested
        if args.output:
            out_file = engine.save_annotated_video(result["annotated_frames"], args.output)
            print(f"\n[✓] Annotated video saved to: {out_file.resolve()}")

        # Display in interactive window
        if not args.no_gui:
            play_annotated_stream(result["annotated_frames"], window_name=f"Action: {result['top_class']}")

        return

    # If neither camera nor video specified
    print("Please specify an action:")
    print("  python demo_realtime.py --camera 0                # Scan live webcam")
    print("  python demo_realtime.py --video my_sport.mp4      # Test any video")
    print("  python demo_realtime.py --sample                  # Test random sample from dataset")
    print("  python demo_realtime.py --sample --output out.mp4 # Test and save video")


if __name__ == "__main__":
    main()

