"""Interactive Streamlit Studio for Sports Action Recognition & HOI Kinematics.

Allows users to:
1. Upload any sports video file (MP4, AVI, MOV, MKV)
2. Or pick from preloaded benchmark test videos
3. Run 4-stream multimodal action classification (Visual, Pose, Object, HOI)
4. View the annotated video with 17-point skeletal pose estimation, sports object bounding boxes, and interaction links
5. Inspect confidence rankings, probability charts, and diagnostic telemetry.
"""

from pathlib import Path
import sys
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import SportsActionInferenceEngine as ActionInferenceEngine


@st.cache_resource(show_spinner="Loading 4-stream deep neural networks onto GPU...")
def get_inference_engine(checkpoint_path: Optional[str] = None) -> ActionInferenceEngine:
    """Cached loader for the inference engine and models."""
    default_pth = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pth"
    default_pt = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pt"

    ckpt = checkpoint_path
    if not ckpt or not Path(ckpt).exists():
        if default_pth.exists():
            ckpt = str(default_pth)
        elif default_pt.exists():
            ckpt = str(default_pt)
        else:
            ckpt = None

    return ActionInferenceEngine(checkpoint_path=ckpt)


def list_sample_videos() -> List[Path]:
    """Finds sample test videos from Model Data/Test."""
    test_dir = PROJECT_ROOT / "Model Data" / "Test"
    if not test_dir.exists():
        test_dir = PROJECT_ROOT / "ucf action"
    if not test_dir.exists():
        return []
    videos = sorted(list(test_dir.rglob("*.avi")) + list(test_dir.rglob("*.mp4")))
    return videos


def main():
    st.set_page_config(
        page_title="Sports Action Recognition Studio",
        page_icon="🏋️‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Header CSS
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F3F4F6;
            border-radius: 8px;
            padding: 12px;
            border-left: 4px solid #2563EB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🏋️‍♂️ Sports Action Recognition Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Human–Object Interaction (HOI) Multimodal AI Suite · 82 Sports Categories</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    st.sidebar.header("⚙️ Model & Evaluation Settings")

    ckpt_choice = st.sidebar.text_input(
        "Checkpoint Path",
        value="checkpoints/M7/best_model.pth",
        help="Path to trained PyTorch checkpoint",
    )
    topk = st.sidebar.slider("Top Predictions to Show", min_value=3, max_value=10, value=5)
    playback_fps = st.sidebar.slider("Annotated Video Playback FPS", min_value=4, max_value=24, value=8)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Active Model Streams:**
        * 👁️ **Visual**: ResNet-50 (2048-d)
        * 🧘 **Pose**: Keypoint R-CNN 17 Joints (85-d)
        * 🎾 **Object**: Faster R-CNN Equipment (7-d)
        * 🔗 **HOI**: Spatial Joint-Object Kinematics (16-d)
        * 🧠 **Classifier**: BiLSTM + 4-Head Attention
        """
    )

    # Load engine
    engine = get_inference_engine(ckpt_choice)

    # Video Source Selection Tabs
    tab_upload, tab_sample = st.tabs(["📤 Upload Custom Video", "🎯 Pick from Benchmark Dataset"])

    target_video_path = None

    with tab_upload:
        st.markdown("Upload any sports video file (**MP4, AVI, MOV, MKV**):")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            # Save uploaded file to temp file
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                target_video_path = Path(tmp_file.name)
            st.success(f"Video uploaded: **{uploaded_file.name}**")

    with tab_sample:
        sample_videos = list_sample_videos()
        if sample_videos:
            sample_labels = [f"{v.parent.name} / {v.name}" for v in sample_videos]
            selected_idx = st.selectbox(
                "Select a benchmark sample from Model Data/Test:",
                range(len(sample_labels)),
                format_func=lambda i: sample_labels[i],
            )
            if st.button("Load Selected Benchmark Video"):
                target_video_path = sample_videos[selected_idx]
                st.info(f"Loaded: **{target_video_path.parent.name}/{target_video_path.name}**")
        else:
            st.warning("No benchmark videos found in Model Data/Test.")

    # Execution Section
    if target_video_path and Path(target_video_path).exists():
        st.markdown("---")
        col_btn, _ = st.columns([2, 5])
        with col_btn:
            run_btn = st.button("🚀 Analyze & Classify Action", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Analyzing video across Visual, Pose, Object, and HOI streams..."):
                t0 = time.time()
                result = engine.predict_video(target_video_path, topk=topk)
                elapsed = time.time() - t0

                # Save annotated video to temporary mp4
                temp_annotated = Path(tempfile.gettempdir()) / f"annotated_{int(time.time())}.mp4"
                engine.save_annotated_video(result["annotated_frames"], temp_annotated, fps=float(playback_fps))

            st.balloons()

            # Metric Banners
            st.markdown("### 🏆 Prediction Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Action", result["top_class"])
            m2.metric("Confidence Score", f"{result['top_confidence']:.2f}%")
            m3.metric("Inference Latency", f"{elapsed:.2f}s")
            m4.metric("Frames Analyzed", f"{len(result['raw_frames'])} frames")

            st.markdown("---")

            # Side-by-Side Video Display
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### 📹 Source Video")
                try:
                    st.video(str(target_video_path))
                except Exception:
                    st.info("Source video format preview not supported directly by browser player.")

            with col_right:
                st.markdown("#### 🤖 Annotated AI Output (Pose + Gear + HOI)")
                if temp_annotated.exists():
                    st.video(str(temp_annotated))
                else:
                    st.error("Failed to render annotated preview.")

            # Top-K Ranking Chart
            st.markdown("### 📊 Top Candidate Rankings")
            chart_data = pd.DataFrame(
                result["predictions"],
                columns=["Action Category", "Probability (%)"],
            ).sort_values("Probability (%)", ascending=True)

            st.bar_chart(chart_data.set_index("Action Category"), horizontal=True)

            # Multimodal Diagnostics Accordion
            with st.expander("🔍 Deep Multimodal Diagnostics & HOI Stream Details", expanded=False):
                st.markdown(
                    f"""
                    * **Visual Stream**: ResNet-50 embedding extracted across 16 temporal segments.
                    * **Pose Stream**: Keypoint R-CNN detected 17 anatomical joints (Nose, Shoulders, Elbows, Wrists, Hips, Knees, Ankles) with velocities.
                    * **Object Stream**: Faster R-CNN scanned for sports equipment (balls, rackets, bats, gloves, skis, boards).
                    * **HOI Kinematics**: Geometric Euclidean distances and directional vectors computed between human extremity joints and equipment centroids.
                    * **Temporal Pooling**: Multi-Head Self-Attention weighted the most discriminative action frames.
                    """
                )


if __name__ == "__main__":
    main()
