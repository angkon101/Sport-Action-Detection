#!/usr/bin/env python3
"""
Generate a clean, simplified high-resolution publication figure (PNG + SVG)
matching the simplified CMAT-HOI Methodology Architecture.
Directly models Reference 7 and Reference 4 for effortless visual understanding.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def render_simplified_methodology(output_png: Path, output_svg: Path):
    fig, ax = plt.subplots(figsize=(14, 18), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Helper for rounded boxes
    def draw_card(x, y, w, h, bg, border, lw=1.5, ls="-", radius=0.03):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=bg, edgecolor=border,
            linewidth=lw, linestyle=ls
        )
        ax.add_patch(rect)
        return rect

    # -------------------------------------------------------------
    # 0. TITLE BANNER
    # -------------------------------------------------------------
    draw_card(0.5, 17.0, 13.0, 0.75, "#1E293B", "#0F172A", lw=2, radius=0.04)
    ax.text(7.0, 17.48, "CMAT-HOI: System Methodology & Architecture Diagram",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#FFFFFF")
    ax.text(7.0, 17.18, "End-to-End Multimodal Framework for Sports Action Recognition (82 Classes)",
            ha="center", va="center", fontsize=9.5, color="#94A3B8")

    # -------------------------------------------------------------
    # STEP 1: DATASET, PREPROCESSING FUNNEL & SPLIT
    # -------------------------------------------------------------
    draw_card(0.5, 13.9, 13.0, 2.9, "#F0F9FF", "#0284C7", lw=1.6, ls="--")
    ax.text(0.8, 16.55, "STEP 1: DATA COLLECTION, PREPROCESSING FUNNEL & DATASET SPLITTING",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#0369A1")

    # 1A: Dataset
    draw_card(0.8, 14.1, 2.6, 2.2, "#FFFFFF", "#38BDF8", lw=1.2)
    ax.text(2.1, 16.05, "Sports Video Dataset", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0284C7")
    ax.plot([1.0, 3.2], [15.85, 15.85], color="#BAE6FD", lw=1)
    ds_text = [
        "• UCF-101 Sports Subset",
        "• 9,151 Full Video Clips",
        "• 82 Action Classes:",
        "   - Tennis, Basketball, Cricket",
        "   - Gymnastics, Diving, Boxing",
        "   - Surfing, Skiing, Archery...",
        "• Format: RGB Video Files"
    ]
    for i, line in enumerate(ds_text):
        ax.text(0.95, 15.55 - i * 0.23, line, ha="left", va="center", fontsize=8.2, color="#334155")

    # 1B: Funnel (Clean 5 steps)
    draw_card(3.7, 14.1, 6.0, 2.2, "#FFFFFF", "#0284C7", lw=1.2)
    ax.text(6.7, 16.05, "5-Step Video Preprocessing Funnel", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0284C7")
    ax.plot([4.0, 9.4], [15.85, 15.85], color="#BAE6FD", lw=1)
    
    funnel_rows = [
        ("1. Frame Sampling", "Extract 16 equidistant frames across each video", "#DBEAFE", "#2563EB"),
        ("2. Image Resizing", "Standardize resolution to 224 × 224 pixels", "#E0E7FF", "#4F46E5"),
        ("3. Motion Velocities", "Compute speed & direction between frames", "#EDE9FE", "#7C3AED"),
        ("4. Data Cleaning", "Fill missing joint points & apparatus boxes", "#FAE8FF", "#C026D3"),
        ("5. Normalization", "Zero-center pixel values & package tensors", "#FCE7F3", "#DB2777")
    ]
    for i, (f_title, f_desc, f_bg, f_border) in enumerate(funnel_rows):
        y_f = 15.48 - i * 0.33
        draw_card(3.9, y_f, 5.6, 0.28, f_bg, f_border, lw=1)
        ax.text(4.05, y_f + 0.14, f_title + ":", ha="left", va="center", fontsize=8, fontweight="bold", color=f_border)
        ax.text(6.65, y_f + 0.14, f_desc, ha="left", va="center", fontsize=7.8, color="#1E293B")

    # 1C: Split
    draw_card(10.0, 14.1, 3.2, 2.2, "#FFFFFF", "#38BDF8", lw=1.2)
    ax.text(11.6, 16.05, "Dataset Split (No Leakage)", ha="center", va="center", fontsize=10, fontweight="bold", color="#0284C7")
    ax.plot([10.3, 12.9], [15.85, 15.85], color="#BAE6FD", lw=1)
    
    # Train box
    draw_card(10.2, 15.0, 2.8, 0.75, "#EFF6FF", "#2563EB", lw=1.2)
    ax.text(11.6, 15.52, "Training Set (70%)", ha="center", va="center", fontsize=9, fontweight="bold", color="#1D4ED8")
    ax.text(11.6, 15.25, "6,408 Video Clips", ha="center", va="center", fontsize=10, fontweight="bold", color="#1E40AF")

    # Test box
    draw_card(10.2, 14.2, 2.8, 0.70, "#F0FDF4", "#16A34A", lw=1.2)
    ax.text(11.6, 14.65, "Testing Set (30%)", ha="center", va="center", fontsize=9, fontweight="bold", color="#15803D")
    ax.text(11.6, 14.40, "2,743 Clips (Unseen Data)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#166534")

    # Arrows in Step 1
    ax.annotate("", xy=(3.7, 15.2), xytext=(3.4, 15.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))
    ax.annotate("", xy=(10.0, 15.2), xytext=(9.7, 15.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))

    # -------------------------------------------------------------
    # STEP 2: 4-STREAM FEATURE EXTRACTION
    # -------------------------------------------------------------
    draw_card(0.5, 11.75, 13.0, 1.95, "#FAF5FF", "#9333EA", lw=1.6, ls="--")
    ax.text(0.8, 13.45, "STEP 2: 4-STREAM MULTIMODAL FEATURE EXTRACTION (WHAT THE MODEL EXTRACTS)",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#6B21A8")

    streams = [
        ("1. Visual Scene", "ResNet-50", "Background context, court lines,\nclothing, outdoor environment", "#DBEAFE", "#2563EB", 0.8),
        ("2. Player Body Pose", "Keypoint R-CNN", "17 skeletal joints (arms, legs)\n+ Movement speeds & angles", "#D1FAE5", "#059669", 3.9),
        ("3. Sports Equipment", "Faster R-CNN", "Ball, tennis racket, bat,\nbarbell, skateboard locations", "#FEF3C7", "#D97706", 7.0),
        ("4. Player-Object Action", "HOI Engine", "Distance between hands/feet & gear\n+ Exact hit/release timing", "#FCE7F3", "#DB2777", 10.1)
    ]
    for stitle, smodel, sdesc, sbg, sborder, sx in streams:
        draw_card(sx, 11.9, 2.9, 1.35, "#FFFFFF", sborder, lw=1.4)
        ax.text(sx + 1.45, 13.0, stitle, ha="center", va="center", fontsize=9.5, fontweight="bold", color=sborder)
        ax.text(sx + 1.45, 12.75, f"Model: {smodel}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1E293B")
        ax.plot([sx + 0.3, sx + 2.6], [12.6, 12.6], color="#E2E8F0", lw=0.8)
        ax.text(sx + 1.45, 12.25, sdesc, ha="center", va="center", fontsize=7.5, color="#475569")

    # Arrow from train split down to streams
    ax.annotate("", xy=(7.0, 13.7), xytext=(11.6, 14.2),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#9333EA", connectionstyle="arc3,rad=-0.15"))
    ax.text(9.2, 13.9, "Extract Features from Training Clips", fontsize=8, fontweight="bold", color="#7E22CE")

    # -------------------------------------------------------------
    # STEP 3: CMAT-HOI CORE NETWORK ARCHITECTURE
    # -------------------------------------------------------------
    draw_card(0.5, 6.75, 13.0, 4.8, "#FFFDF5", "#D97706", lw=1.8)
    ax.text(0.8, 11.3, "STEP 3: CMAT-HOI CORE NETWORK ARCHITECTURE",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#92400E")

    # 3A: Modality Stems
    draw_card(0.8, 10.45, 12.4, 0.60, "#FEF3C7", "#D97706", lw=1.2)
    ax.text(7.0, 10.82, "1. Modality Alignment (Stems)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#92400E")
    ax.text(7.0, 10.60, "Transforms all 4 streams (images, body joints, equipment, interactions) into a unified matching feature size (256 dimensions)",
            ha="center", va="center", fontsize=8.2, color="#451A03")

    # 3B: Cross-Attention
    draw_card(0.8, 9.65, 12.4, 0.65, "#FDF2F8", "#BE185D", lw=1.4)
    ax.text(7.0, 10.08, "2. Cross-Modal Attention (Player Body Pose ↔ Sports Equipment)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#831843")
    ax.text(7.0, 9.82, "Directly correlates body posture with equipment location (e.g. knowing where the racket is when the arm swings)",
            ha="center", va="center", fontsize=8.2, color="#4C0519")

    # 3C: Gated Multimodal Fusion
    draw_card(0.8, 8.75, 12.4, 0.75, "#FEF9C3", "#CA8A04", lw=1.4)
    ax.text(7.0, 9.25, "3. Smart Gated Fusion (Adaptive Modality Trust)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#78350F")
    ax.text(7.0, 9.0, "Automatically weights which modality is most reliable per frame:", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#92400E")
    ax.text(7.0, 8.85, "• Body occluded? Trust equipment & scene &nbsp;|&nbsp; • Ball impact instant? Prioritize hand-object interaction!",
            ha="center", va="center", fontsize=8, color="#451A03")

    # 3D: Spatio-Temporal Transformer
    draw_card(0.8, 7.85, 12.4, 0.75, "#ECFDF5", "#059669", lw=1.4)
    ax.text(7.0, 8.35, "4. Spatio-Temporal Transformer (Sequence Pattern Learning)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#064E3B")
    ax.text(7.0, 8.12, "Reads the full 16 frames in chronological sequence using Self-Attention:", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#047857")
    ax.text(7.0, 7.97, "Understands the complete athletic motion storyline (Preparation ──► Peak Strike / Jump ──► Follow-Through)",
            ha="center", va="center", fontsize=8, color="#022C22")

    # 3E: Multi-Scale Temporal Pooling
    draw_card(0.8, 6.95, 12.4, 0.75, "#CFFAFE", "#0891B2", lw=1.4)
    ax.text(7.0, 7.45, "5. Multi-Scale Temporal Pooling (Rich Action Signature)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0E7490")
    ax.text(7.0, 7.20, "Combines 4 perspectives into one complete action vector:", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#155E75")
    ax.text(7.0, 7.05, "Global Scene Context  +  Peak Explosive Spike  +  Average Movement Flow  +  Learned Key Moments",
            ha="center", va="center", fontsize=8, color="#164E63")

    # Vertical sequence arrows in Step 3
    ax.annotate("", xy=(7.0, 10.45), xytext=(7.0, 10.3), arrowprops=dict(arrowstyle="<-", lw=1.2, color="#D97706"))
    ax.annotate("", xy=(7.0, 9.65), xytext=(7.0, 9.5), arrowprops=dict(arrowstyle="<-", lw=1.2, color="#D97706"))
    ax.annotate("", xy=(7.0, 8.75), xytext=(7.0, 8.6), arrowprops=dict(arrowstyle="<-", lw=1.2, color="#D97706"))
    ax.annotate("", xy=(7.0, 7.85), xytext=(7.0, 7.7), arrowprops=dict(arrowstyle="<-", lw=1.2, color="#D97706"))

    # -------------------------------------------------------------
    # STEP 4: MODEL TRAINING & EVALUATION LOOP
    # -------------------------------------------------------------
    draw_card(0.5, 3.85, 13.0, 2.7, "#FEF2F2", "#EF4444", lw=1.6, ls="--")
    ax.text(0.8, 6.32, "STEP 4: TRAINING OPTIMIZATION & INDEPENDENT TEST EVALUATION",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#B91C1C")

    # 4A: Training
    draw_card(0.8, 4.0, 3.7, 2.1, "#FFFFFF", "#EF4444", lw=1.2)
    ax.text(2.65, 5.85, "Model Training & Loss", ha="center", va="center", fontsize=10, fontweight="bold", color="#B91C1C")
    ax.plot([1.1, 4.2], [5.65, 5.65], color="#FECACA", lw=1)
    train_lines = [
        "• 3-Layer Classifier Head",
        "• Focal Loss with Label Smoothing",
        "   (Prevents confusion on close sports)",
        "• AdamW Optimizer (Adaptive LR)",
        "• Cosine Annealing Schedule",
        "• Multimodal Mixup Augmentation"
    ]
    for i, line in enumerate(train_lines):
        ax.text(1.0, 5.38 - i * 0.24, line, ha="left", va="center", fontsize=8, color="#1E293B")

    # 4B: Center loop
    draw_card(4.8, 4.0, 4.2, 2.1, "#FFFFFF", "#94A3B8", lw=1.2)
    ax.text(6.9, 5.85, "Iterative Training Loop", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax.plot([5.1, 8.7], [5.65, 5.65], color="#E2E8F0", lw=1)
    ax.text(6.9, 5.30, "The neural network adjusts\nits parameters epoch-by-epoch\nuntil top accuracy is achieved.",
            ha="center", va="center", fontsize=8.2, color="#475569")
    
    draw_card(5.0, 4.2, 3.8, 0.7, "#DCFCE7", "#16A34A", lw=1)
    ax.text(6.9, 4.65, "BEST MODEL CHECKPOINT", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#14532D")
    ax.text(6.9, 4.40, "Peak Epoch 23 (Saved to best_model.pt)", ha="center", va="center", fontsize=8, color="#15803D")

    # 4C: Test Evaluation
    draw_card(9.3, 4.0, 3.9, 2.1, "#FFFFFF", "#10B981", lw=1.2)
    ax.text(11.25, 5.85, "Independent Test Evaluation", ha="center", va="center", fontsize=10, fontweight="bold", color="#047857")
    ax.plot([9.6, 12.9], [5.65, 5.65], color="#A7F3D0", lw=1)
    
    draw_card(9.5, 4.85, 3.5, 0.75, "#ECFDF5", "#059669", lw=1)
    ax.text(11.25, 5.35, "Top-1 Accuracy: 90.19%", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#065F46")
    ax.text(11.25, 5.05, "Top-5 Accuracy: 99.02%", ha="center", va="center", fontsize=9, fontweight="bold", color="#065F46")

    ax.text(9.6, 4.60, "• Evaluated on 2,743 Unseen Videos", ha="left", va="center", fontsize=8, color="#1E293B")
    ax.text(9.6, 4.35, "• Lowest Loss: 0.4397 (Peak Generalization)", ha="left", va="center", fontsize=8, color="#1E293B")
    ax.text(9.6, 4.12, "• Zero data leakage between actors", ha="left", va="center", fontsize=8, color="#1E293B")

    # Arrows in Step 4
    ax.annotate("", xy=(4.8, 5.0), xytext=(4.5, 5.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#EF4444"))
    ax.annotate("", xy=(9.3, 5.0), xytext=(9.0, 5.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.annotate("", xy=(6.9, 6.1), xytext=(6.9, 6.95), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))

    # -------------------------------------------------------------
    # STEP 5: REAL-TIME DEPLOYMENT & ACTION PREDICTION
    # -------------------------------------------------------------
    draw_card(0.5, 0.75, 13.0, 2.9, "#ECFDF5", "#059669", lw=1.8)
    ax.text(0.8, 3.4, "STEP 5: REAL-TIME DEPLOYMENT & ACTION PREDICTION INTERFACES",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#047857")

    # 5A: Live Input
    draw_card(0.8, 0.95, 3.6, 2.2, "#FFFFFF", "#059669", lw=1.2)
    ax.text(2.6, 2.9, "Live Video / Camera Input", ha="center", va="center", fontsize=10, fontweight="bold", color="#047857")
    ax.plot([1.1, 4.1], [2.7, 2.7], color="#A7F3D0", lw=1)
    live_lines = [
        "• 30+ FPS Live Camera Feed",
        "• Or Unseen Match Video (.mp4)",
        "• Continuous 16-frame rolling queue",
        "• Instant real-time video streaming"
    ]
    for i, line in enumerate(live_lines):
        ax.text(1.1, 2.35 - i * 0.28, line, ha="left", va="center", fontsize=8.2, color="#1E293B")

    # 5B: ActionInferenceEngine
    draw_card(4.7, 0.95, 4.3, 2.2, "#FFFFFF", "#059669", lw=1.2)
    ax.text(6.85, 2.9, "Trained CMAT-HOI Engine", ha="center", va="center", fontsize=10, fontweight="bold", color="#047857")
    ax.plot([5.0, 8.7], [2.7, 2.7], color="#A7F3D0", lw=1)
    engine_lines = [
        "• Loads weights from best_model.pt",
        "• High-Speed GPU Forward Pass",
        "• Inference Speed: ~18.4 ms / clip",
        "• End-to-End Latency < 33 ms",
        "• True 30+ FPS Real-Time Execution"
    ]
    for i, line in enumerate(engine_lines):
        ax.text(5.0, 2.35 - i * 0.28, line, ha="left", va="center", fontsize=8.2, color="#1E293B")

    # 5C: User Interfaces & Output
    draw_card(9.3, 0.95, 3.9, 2.2, "#FFFFFF", "#059669", lw=1.2)
    ax.text(11.25, 2.9, "Action Predictions & UI", ha="center", va="center", fontsize=10, fontweight="bold", color="#047857")
    ax.plot([9.6, 12.9], [2.7, 2.7], color="#A7F3D0", lw=1)
    ui_lines = [
        "• Output: Top-1 & Top-5 Probabilities",
        "   (e.g. TennisSwing 94.6%)",
        "• Web App: Streamlit Dashboard",
        "• Desktop App: Tkinter Native GUI",
        "• Live Camera HUD with Joint Skeletons"
    ]
    for i, line in enumerate(ui_lines):
        ax.text(9.6, 2.35 - i * 0.28, line, ha="left", va="center", fontsize=8.2, color="#1E293B")

    # Arrows in Step 5
    ax.annotate("", xy=(4.7, 2.05), xytext=(4.4, 2.05), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.annotate("", xy=(9.3, 2.05), xytext=(9.0, 2.05), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.annotate("", xy=(6.85, 3.15), xytext=(6.9, 4.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.text(7.05, 3.6, "Deploy Saved Model", fontsize=8, fontweight="bold", color="#047857")

    # -------------------------------------------------------------
    # CAPTION
    # -------------------------------------------------------------
    ax.text(7.0, 0.40, "Fig. 1. Simplified System Architecture Diagram: CMAT-HOI End-to-End Sports Action Recognition Framework.",
            ha="center", va="center", fontsize=10, fontfamily="serif", fontstyle="italic", color="#475569")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved simplified PNG to: {output_png}")
    print(f"Saved simplified SVG to: {output_svg}")


if __name__ == "__main__":
    out_dir_m7 = Path("diagrams/M7")
    out_dir_paper = Path("Paper writing/figures")

    render_simplified_methodology(
        out_dir_m7 / "methodology_architecture_diagram.png",
        out_dir_m7 / "methodology_architecture_diagram.svg"
    )
    render_simplified_methodology(
        out_dir_paper / "methodology_architecture_diagram.png",
        out_dir_paper / "methodology_architecture_diagram.svg"
    )
