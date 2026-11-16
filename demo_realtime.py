"""Interactive Live Real-Time & Video Validation Demo for Sports Action Recognition.

Runs on local machine (VS Code / PowerShell):
- Live Webcam Scanning: python demo_realtime.py --camera 0
- Custom Video File Test: python demo_realtime.py --video path/to/sport_video.mp4
- Random Test Video Sample: python demo_realtime.py --sample
"""

import argparse
from pathlib import Path
import random
import sys
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
    """Live scanning loop for webcam validation."""
    print(f"\n--> Opening Webcam (Device Index: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}. Please check connection or permissions.")
        return

    window_name = f"Live Sports Action Scanner (Camera {camera_id})"
    print(f"--> Camera ready! Buffering {buffer_size} frames for real-time action recognition.")
    print("    Press 'q' or 'ESC' to exit | 'c' to capture & predict current action.")

    frame_buffer = []
    current_prediction = "Buffering action..."
    current_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from camera.")
            break

        # Collect buffer of frames
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb_frame)

        display_frame = frame.copy()

        # When we have enough frames, run action prediction
        if len(frame_buffer) >= buffer_size:
            # Run prediction on latest window
            result = engine.predict_video(frame_buffer[-buffer_size:], topk=3)
            current_prediction = result["top_class"]
            current_conf = result["top_confidence"]

            # Use the latest annotated frame from the inference engine
            display_frame = cv2.cvtColor(result["annotated_frames"][-1], cv2.COLOR_RGB2BGR)
            # Slide buffer
            frame_buffer = frame_buffer[-(buffer_size // 2):]
        else:
            # Show buffering status
            cv2.putText(
                display_frame,
                f"Scanning motion... [{len(frame_buffer)}/{buffer_size} frames]",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:
            break
        elif key == ord("c"):
            print(f"[LIVE CAPTURE] Predicted Action: {current_prediction} ({current_conf:.1f}%)")

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
            print(f"[ERROR] Video file not found: {target_video}")
            return

    # Initialize Engine
    print("=" * 65)
    print("SPORTS ACTION REAL-TIME VALIDATION ENGINE")
    print("=" * 65)
    engine = ActionInferenceEngine(checkpoint_path=args.checkpoint)

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
