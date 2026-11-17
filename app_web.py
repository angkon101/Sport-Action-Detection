"""Interactive Web Application for Sports Action Recognition & HOI Validation.

Works seamlessly on:
- Google Colab (with public gradio share link)
- Kaggle (with public gradio share link)
- Local Machine (localhost web browser)

Allows:
- Uploading any custom video file (MP4, AVI, MOV)
- Recording live video directly from the browser webcam
- Real-time visualization of 17-joint skeleton, sports gear bounding boxes, and HOI interaction vectors
- Interactive Top-5 prediction confidence chart
"""

import argparse
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, Tuple

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    print("\n[!] Gradio is not installed. Please install it with:")
    print("    pip install gradio\n")
    sys.exit(1)

from src.inference import ActionInferenceEngine

# Initialize Inference Engine
print("--> Initializing Sports Action Validation Engine...")
engine = ActionInferenceEngine()


def analyze_sports_video(video_path: str) -> Tuple[str, Dict[str, float], str]:
    """Processes uploaded or webcam-recorded video and returns annotated video + probabilities."""
    if not video_path:
        return None, {}, "No video uploaded."

    t0 = time.time()
    result = engine.predict_video(video_path, topk=5)
    elapsed = time.time() - t0

    # Save annotated video to temporary file
    temp_out = Path(tempfile.gettempdir()) / f"annotated_sport_{int(time.time())}.mp4"
    engine.save_annotated_video(result["annotated_frames"], temp_out, fps=8.0)

    # Format probabilities dictionary for Gradio Label
    conf_dict = {cls_name: prob / 100.0 for cls_name, prob in result["predictions"]}

    # Telemetry report
    telemetry = (
        f"Predicted Action: {result['top_class']}\n"
        f"Confidence      : {result['top_confidence']:.2f}%\n"
        f"Processing Time : {elapsed:.2f} seconds\n"
        f"Keyframes Sampled: {len(result['raw_frames'])} frames\n"
        f"Streams Active  : Visual (ResNet-50) | Pose (Keypoint R-CNN) | Object (Faster R-CNN) | HOI Kinematics"
    )

    return str(temp_out), conf_dict, telemetry


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="emerald")
    with gr.Blocks(title="Sports Action Recognition & HOI Kinematics", theme=theme) as demo:
        gr.Markdown(
            """
            # 🏋️‍♂️ Human–Object Interaction (HOI) Sports Action Recognition
            ### Real-Time Validation & Action Detection Suite (82 Sports Categories)
            
            Upload any sports video or record a short clip from your webcam to validate the **4-stream multimodal architecture**:
            - **Visual Stream**: ResNet-50 scene context
            - **Pose Stream**: Keypoint R-CNN 17 body joints
            - **Object Stream**: Faster R-CNN sports gear detection
            - **HOI Stream**: Wrist/ankle-to-gear geometric interaction vectors
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(
                    label="Input Video (Upload File or Record from Webcam)",
                    sources=["upload", "webcam"],
                )
                analyze_btn = gr.Button("🔍 Detect Sports Action", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Annotated Video (Skeleton, Sports Gear & HOI Vectors)",
                    autoplay=True,
                )
                output_label = gr.Label(num_top_classes=5, label="Action Prediction Probabilities")
                output_telemetry = gr.Textbox(label="Diagnostic Telemetry", lines=4)

        analyze_btn.click(
            fn=analyze_sports_video,
            inputs=[input_video],
            outputs=[output_video, output_label, output_telemetry],
        )

        gr.Markdown(
            """
            ---
            *Human–Object Interaction-Aware Multimodal CNN–LSTM Framework for Sports Action Recognition*
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Sports Action Recognition Web App")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link (for Colab / Kaggle)")
    parser.add_argument("--port", type=int, default=7860, help="Local port for server")
    args = parser.parse_args()

    app = build_app()
    print(f"\n--> Launching Validation Web App on port {args.port} (Public Share: {args.share})...")
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

