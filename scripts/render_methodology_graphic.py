#!/usr/bin/env python3
"""
Generate a high-resolution companion publication figure (PNG + SVG)
matching the complete 5-phase CMAT-HOI Methodology Architecture Diagram.
This serves as an instant visual companion to the .drawio file.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def render_methodology_diagram(output_png: Path, output_svg: Path):
    fig, ax = plt.subplots(figsize=(16, 22), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.axis("off")

    # Background
    fig.patch.set_facecolor("#FFFFFF")

    # Helper function for fancy boxes
    def draw_box(x, y, w, h, bg, border, lw=1.5, ls="-", radius=0.04):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=bg, edgecolor=border,
            linewidth=lw, linestyle=ls
        )
        ax.add_patch(rect)
        return rect

    # -------------------------------------------------------------
    # 0. HEADER BANNER
    # -------------------------------------------------------------
    draw_box(0.6, 20.8, 14.8, 0.9, "#0F172A", "#1E293B", lw=2, radius=0.06)
    ax.text(8.0, 21.38, "METHODOLOGY ARCHITECTURAL DIAGRAM: CMAT-HOI SPORTS ACTION RECOGNITION",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#FFFFFF")
    ax.text(8.0, 20.98, "End-to-End Multimodal Cross-Modal Attention Transformer with Human-Object Interaction Modeling (82 Classes)",
            ha="center", va="center", fontsize=10, color="#94A3B8")

    # -------------------------------------------------------------
    # PHASE 1: DATA INGESTION & PREPROCESSING FUNNEL
    # -------------------------------------------------------------
    draw_box(0.6, 17.0, 14.8, 3.55, "#F0F7FF", "#2563EB", lw=1.8, ls="--", radius=0.05)
    ax.text(1.0, 20.25, "PHASE 1: DATASET INGESTION, PREPROCESSING FUNNEL & DISJOINT SPLITTING",
            ha="left", va="center", fontsize=12, fontweight="bold", color="#1E3A8A")

    # Phase 1 Left: Data Source
    draw_box(0.9, 17.25, 3.4, 2.7, "#FFFFFF", "#3B82F6", lw=1.5)
    ax.text(2.6, 19.65, "UCF-101 Sports Repository", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E40AF")
    ax.plot([1.2, 4.0], [19.4, 19.4], color="#BFDBFE", lw=1)
    
    repo_lines = [
        "• Total Clips: 9,151 Videos",
        "• Categories: 82 Sports Classes",
        "• Domains:",
        "   - Ball & Racquet (Tennis, Cricket)",
        "   - Track & Field (HighJump, Vault)",
        "   - Water & Snow (Surfing, Skiing)",
        "   - Combat & Sport (Boxing, Archery)",
        "• Format: RGB .avi / .mp4, 25-30 FPS",
        "• Severe occlusions & high speeds"
    ]
    for i, line in enumerate(repo_lines):
        ax.text(1.1, 19.1 - i * 0.22, line, ha="left", va="center", fontsize=8.2, color="#334155")

    # Phase 1 Center: 5-Stage Funnel (Reference 7 style)
    draw_box(4.6, 19.5, 6.8, 0.45, "#EFF6FF", "#2563EB", lw=1.2)
    ax.text(8.0, 19.72, "5-Stage Preprocessing & Conditioning Funnel", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E40AF")

    funnel_items = [
        ("Step 1: Uniform Temporal Sampling", "Extract T = 16 equidistant frames spanning clip", "#DBEAFE", "#2563EB"),
        ("Step 2: Spatial Resizing & Normalization", "Resize to 224×224 px; ImageNet standard normalization (μ, σ)", "#E0E7FF", "#4F46E5"),
        ("Step 3: Kinematic Velocity Computation", "First-order temporal displacement Δp_t = p_t - p_(t-1)", "#EDE9FE", "#7C3AED"),
        ("Step 4: Apparatus Imputation & Filtering", "Zero-fill missing gear bboxes; Kalman smoothing on joints", "#FAE8FF", "#C026D3"),
        ("Step 5: Pre-Flight Tensor Sanitization", "FP32 casting, NaN/Inf elimination, multimodal alignment", "#FCE7F3", "#DB2777")
    ]
    for i, (f_title, f_desc, f_bg, f_border) in enumerate(funnel_items):
        y_pos = 18.95 - i * 0.40
        draw_box(4.6, y_pos, 6.8, 0.35, f_bg, f_border, lw=1.2)
        ax.text(4.8, y_pos + 0.175, f"{f_title}:", ha="left", va="center", fontsize=8.5, fontweight="bold", color=f_border)
        ax.text(8.2, y_pos + 0.175, f_desc, ha="left", va="center", fontsize=8, color="#1E293B")

    # Phase 1 Right: Disjoint Split
    draw_box(11.7, 17.25, 3.4, 2.7, "#FFFFFF", "#3B82F6", lw=1.5)
    ax.text(13.4, 19.65, "Subject-Disjoint Split", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E40AF")
    ax.plot([12.0, 14.8], [19.4, 19.4], color="#BFDBFE", lw=1)
    
    # Train box
    draw_box(11.9, 18.3, 3.0, 0.95, "#EFF6FF", "#3B82F6", lw=1.2)
    ax.text(13.4, 19.0, "Training Split (70%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1D4ED8")
    ax.text(13.4, 18.68, "6,408 Video Clips", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E40AF")
    ax.text(13.4, 18.42, "Mixup + Kinematic Jitter", ha="center", va="center", fontsize=7.5, color="#64748B")

    # Test box
    draw_box(11.9, 17.4, 3.0, 0.75, "#F0FDF4", "#16A34A", lw=1.2)
    ax.text(13.4, 17.92, "Validation & Test (30%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#15803D")
    ax.text(13.4, 17.65, "2,743 Clips (Unseen Subjects)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#166534")

    # Arrows in Phase 1
    ax.annotate("", xy=(4.6, 18.6), xytext=(4.3, 18.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563EB"))
    ax.annotate("", xy=(11.7, 18.6), xytext=(11.4, 18.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563EB"))

    # -------------------------------------------------------------
    # PHASE 2: SYNCHRONIZED MULTIMODAL FEATURE EXTRACTION
    # -------------------------------------------------------------
    draw_box(0.6, 13.9, 14.8, 2.9, "#FAF5FF", "#9333EA", lw=1.8, ls="--", radius=0.05)
    ax.text(1.0, 16.55, "PHASE 2: SYNCHRONIZED MULTIMODAL FEATURE EXTRACTION (4 PARALLEL TEMPORAL STREAMS, T = 16)",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color="#581C87")

    streams = [
        ("Visual Appearance", "ResNet-50 (Pretrained ImageNet)", "X_vis ∈ ℝ^(16 × 2048)", "Scene context, court surfaces, attire", "#DBEAFE", "#2563EB", 0.9),
        ("Skeletal Kinematics", "Keypoint R-CNN (ResNet-50-FPN)", "X_pose ∈ ℝ^(16 × 85)", "17 Joints (x, y, c) + Velocities Δ(x, y)", "#D1FAE5", "#059669", 4.6),
        ("Sports Apparatus", "Faster R-CNN (FPN Backbone)", "X_obj ∈ ℝ^(16 × 7)", "[x1, y1, x2, y2, score, cls, aspect]", "#FEF3C7", "#D97706", 8.3),
        ("Geometric HOI", "Kinematic HOI Engine", "X_hoi ∈ ℝ^(16 × 16)", "Wrists/Ankles ↔ Gear vectors & distances", "#FCE7F3", "#DB2777", 12.0)
    ]
    for s_name, s_backbone, s_tensor, s_desc, s_bg, s_border, s_x in streams:
        draw_box(s_x, 14.1, 3.1, 2.2, "#FFFFFF", s_border, lw=1.5)
        ax.text(s_x + 1.55, 16.05, s_name, ha="center", va="center", fontsize=10.5, fontweight="bold", color=s_border)
        ax.text(s_x + 1.55, 15.65, s_backbone, ha="center", va="center", fontsize=8.5, fontweight="semibold", color="#334155")
        
        draw_box(s_x + 0.2, 14.9, 2.7, 0.45, s_bg, s_border, lw=1)
        ax.text(s_x + 1.55, 15.12, s_tensor, ha="center", va="center", fontsize=9, fontweight="bold", color=s_border)
        
        ax.text(s_x + 1.55, 14.5, s_desc, ha="center", va="center", fontsize=7.8, color="#64748B", wrap=True)

    # Arrow from Train Split down to Streams
    ax.annotate("", xy=(8.0, 16.8), xytext=(13.4, 17.4),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#9333EA", connectionstyle="arc3,rad=-0.15"))
    ax.text(10.8, 17.05, "Train Batches (6,408 clips)", fontsize=8, fontweight="bold", color="#7E22CE")

    # -------------------------------------------------------------
    # PHASE 3: CMAT-HOI CORE NETWORK ARCHITECTURE
    # -------------------------------------------------------------
    draw_box(0.6, 6.7, 14.8, 7.0, "#FFFDF5", "#D97706", lw=2, radius=0.05)
    ax.text(1.0, 13.45, "PHASE 3: CMAT-HOI CORE ARCHITECTURE (CROSS-MODAL ATTENTION TRANSFORMER & GATED FUSION)",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color="#78350F")

    # 3A: Modality Stems
    stems = [
        ("Visual Stem", "1D Conv + LN + GELU", "E_vis ∈ ℝ^(16 × 256)", "#DBEAFE", "#1D4ED8", 0.9),
        ("Pose Stem", "1D Conv + LN + GELU", "E_pose ∈ ℝ^(16 × 256)", "#D1FAE5", "#047857", 4.6),
        ("Object Stem", "1D Conv + LN + GELU", "E_obj ∈ ℝ^(16 × 256)", "#FEF3C7", "#B45309", 8.3),
        ("HOI Stem", "1D Conv + LN + GELU", "E_hoi ∈ ℝ^(16 × 256)", "#FCE7F3", "#BE185D", 12.0)
    ]
    for st_name, st_op, st_out, st_bg, st_border, st_x in stems:
        draw_box(st_x, 12.6, 3.1, 0.65, st_bg, st_border, lw=1.2)
        ax.text(st_x + 1.55, 13.05, f"{st_name} (d=256)", ha="center", va="center", fontsize=8.5, fontweight="bold", color=st_border)
        ax.text(st_x + 1.55, 12.8, st_out, ha="center", va="center", fontsize=8, fontweight="bold", color="#1E293B")
        # Arrow down from extraction stream
        ax.annotate("", xy=(st_x + 1.55, 13.25), xytext=(st_x + 1.55, 14.1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748B"))

    # 3B: Bi-Directional Cross Attention (Pose <-> Object)
    draw_box(3.5, 11.2, 9.0, 1.1, "#FDF2F8", "#BE185D", lw=1.8)
    ax.text(8.0, 12.05, "Bi-Directional Keypoint–Equipment Cross-Attention Module (h = 4 Heads, d_k = 64)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#831843")
    
    # Left sub-equation
    draw_box(3.8, 11.35, 4.1, 0.55, "#FFFFFF", "#F472B6", lw=1)
    ax.text(5.85, 11.7, "Pose Queries Equipment (Pose Enhancement)", ha="center", va="center", fontsize=8, fontweight="bold", color="#9D174D")
    ax.text(5.85, 11.48, "P_enh = LN(E_pose + MHA(Q=E_p, K=E_o, V=E_o))", ha="center", va="center", fontsize=7.5, color="#1E293B")

    # Right sub-equation
    draw_box(8.1, 11.35, 4.1, 0.55, "#FFFFFF", "#F472B6", lw=1)
    ax.text(10.15, 11.7, "Equipment Queries Pose (Object Enhancement)", ha="center", va="center", fontsize=8, fontweight="bold", color="#9D174D")
    ax.text(10.15, 11.48, "O_enh = LN(E_obj + MHA(Q=E_o, K=E_p, V=E_p))", ha="center", va="center", fontsize=7.5, color="#1E293B")

    # Arrows into Cross-Attention
    ax.annotate("", xy=(5.85, 12.3), xytext=(6.15, 12.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#BE185D"))
    ax.annotate("", xy=(10.15, 12.3), xytext=(9.85, 12.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#BE185D"))

    # 3C: Gated Multimodal Fusion Unit (GMFU)
    draw_box(1.5, 9.9, 13.0, 0.95, "#FEF9C3", "#D97706", lw=1.8)
    ax.text(8.0, 10.6, "Gated Multimodal Fusion Unit (GMFU) — Dynamic Frame-Level Modality Attention Weighting",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#78350F")
    ax.text(8.0, 10.3, "Modality Gates: g_t = Softmax( W_g · [ E_vis,t ; P_enh,t ; O_enh,t ; E_hoi,t ] + b_g ) ∈ ℝ^4",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#92400E")
    ax.text(8.0, 10.05, "Fused Token: F_t = g_t,1·E_vis,t + g_t,2·P_enh,t + g_t,3·O_enh,t + g_t,4·E_hoi,t  ──►  F ∈ ℝ^(16 × 256)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#B45309")

    # Arrows into GMFU
    ax.annotate("", xy=(2.45, 10.85), xytext=(2.45, 12.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))
    ax.annotate("", xy=(8.0, 10.85), xytext=(8.0, 11.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))
    ax.annotate("", xy=(13.55, 10.85), xytext=(13.55, 12.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))

    # 3D: Spatio-Temporal Transformer Encoder
    draw_box(1.5, 8.5, 13.0, 1.15, "#ECFDF5", "#059669", lw=1.8)
    ax.text(8.0, 9.42, "Spatio-Temporal Transformer Encoder (Pre-LayerNorm, 3 Stacked Layers)",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#064E3B")
    ax.text(8.0, 9.15, "Token Sequence: Z_0 = [ z_CLS ; F_1 ; F_2 ; ... ; F_16 ] + E_pos ∈ ℝ^(17 × 256)   (Prepend Learnable [z_CLS] + Sinusoidal Positional Encoding)",
            ha="center", va="center", fontsize=8.5, color="#047857")
    ax.text(8.0, 8.85, "3 Encoder Blocks  |  h = 8 Multi-Head Attention Heads  |  d_model = 256  |  d_ff = 1024  |  GELU  |  Dropout p = 0.1",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#065F46")
    ax.text(8.0, 8.65, "Output Sequence: Z_final = [z_CLS ; z_1 ; ... ; z_16] ∈ ℝ^(17 × 256)  (Captures Long-Range Temporal Dynamics)",
            ha="center", va="center", fontsize=8, color="#047857")

    ax.annotate("", xy=(8.0, 9.65), xytext=(8.0, 9.9), arrowprops=dict(arrowstyle="->", lw=1.8, color="#059669"))

    # 3E: Multi-Scale 4-Way Temporal Pooling Head
    draw_box(1.5, 6.9, 13.0, 1.35, "#CFFAFE", "#0891B2", lw=1.8)
    ax.text(8.0, 8.05, "Multi-Scale 4-Way Temporal Pooling Mechanism (Comprehensive Action Representation)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#155E75")

    pools = [
        ("[z_CLS] Token (256-d)", "Global Contextual Anchor", 1.8),
        ("Temporal Attention (256-d)", "Learned Saliency α_t Weighting", 4.9),
        ("Global Max Pool (256-d)", "Peak Explosive Athletic Spike", 8.0),
        ("Global Avg Pool (256-d)", "Sustained Baseline Motion", 11.1)
    ]
    for p_name, p_desc, p_x in pools:
        draw_box(p_x, 7.35, 2.9, 0.5, "#FFFFFF", "#0891B2", lw=1)
        ax.text(p_x + 1.45, 7.67, p_name, ha="center", va="center", fontsize=8, fontweight="bold", color="#0E7490")
        ax.text(p_x + 1.45, 7.47, p_desc, ha="center", va="center", fontsize=7.2, color="#475569")

    # Unified 1024-d Concat vector
    draw_box(3.5, 6.98, 9.0, 0.32, "#E0F2FE", "#0284C7", lw=1.2)
    ax.text(8.0, 7.14, "Concatenated Sports Action Descriptor: Z_pool = [ z_CLS ; z_Attn ; z_Max ; z_Avg ] ∈ ℝ^1024",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0369A1")

    ax.annotate("", xy=(8.0, 8.25), xytext=(8.0, 8.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#0891B2"))

    # -------------------------------------------------------------
    # PHASE 4: OPTIMIZATION, LOSS FORMULATION & RIGOROUS EVALUATION
    # -------------------------------------------------------------
    draw_box(0.6, 3.4, 14.8, 3.1, "#FEF2F2", "#DC2626", lw=1.8, ls="--", radius=0.05)
    ax.text(1.0, 6.25, "PHASE 4: OPTIMIZATION, LOSS FORMULATION & RIGOROUS MODEL EVALUATION",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color="#991B1B")

    # 4A: MLP Classifier & Augmentation
    draw_box(0.9, 3.55, 4.4, 2.45, "#FFFFFF", "#DC2626", lw=1.5)
    ax.text(3.1, 5.8, "3-Layer MLP Classifier", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#991B1B")
    ax.plot([1.2, 5.0], [5.6, 5.6], color="#FCA5A5", lw=1)
    
    mlp_lines = [
        "• Linear(1024 ──► 512) + LN + GELU + Drop(0.3)",
        "• Linear(512 ──► 256) + LN + GELU + Drop(0.2)",
        "• Linear(256 ──► 82) ──► Logits ŷ ∈ ℝ^82",
        "• AdamW Optimizer (lr = 1e-4, wd = 1e-2)",
        "• CosineAnnealingLR (T_max = 30) + 5 Warmup",
        "• Multimodal Mixup (β = 0.4)",
        "• Kinematic Jitter (σ = 0.008 Gaussian)",
        "• Temporal Frame Dropout (p = 0.1)"
    ]
    for i, line in enumerate(mlp_lines):
        ax.text(1.1, 5.35 - i * 0.22, line, ha="left", va="center", fontsize=8, color="#1E293B")

    # 4B: Focal Loss Formulation
    draw_box(5.6, 3.55, 4.8, 2.45, "#FFF1F2", "#E11D48", lw=1.5)
    ax.text(8.0, 5.8, "Focal Loss with Label Smoothing", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#9F1239")
    ax.plot([5.9, 10.1], [5.6, 5.6], color="#FECDD3", lw=1)

    draw_box(5.8, 5.05, 4.4, 0.45, "#FFFFFF", "#E11D48", lw=1.2)
    ax.text(8.0, 5.27, "L_focal = - ∑_(c=1)^82 α_c · (1 - p_c)^γ · y_c^(smooth) · log(p_c)",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#BE123C")

    loss_lines = [
        "• γ = 1.5 (Focusing parameter): suppresses easy classes",
        "• α_c = 0.08: handles class frequency imbalance",
        "• ε = 0.08 (Label Smoothing): y_c^smooth = (1-ε)y_c + ε/82",
        "• Prevents overconfidence on visually close sports actions",
        "• Backpropagation updates Transformer and Stems"
    ]
    for i, line in enumerate(loss_lines):
        ax.text(5.8, 4.8 - i * 0.23, line, ha="left", va="center", fontsize=8, color="#334155")

    # 4C: Model Evaluation & Peak Milestone Checkpoint
    draw_box(10.7, 3.55, 4.4, 2.45, "#F0FDF4", "#16A34A", lw=1.5)
    ax.text(12.9, 5.8, "Model Evaluation & Peak Milestone", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#15803D")
    ax.plot([11.0, 14.8], [5.6, 5.6], color="#BBF7D0", lw=1)
    
    # Milestone badge
    draw_box(10.9, 4.5, 4.0, 1.0, "#DCFCE7", "#16A34A", lw=1.2)
    ax.text(12.9, 5.28, "PEAK VALIDATION (EPOCH 23)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#14532D")
    ax.text(11.9, 4.95, "Top-1: 90.19%", ha="center", va="center", fontsize=10, fontweight="bold", color="#15803D")
    ax.text(13.9, 4.95, "Top-5: 99.02%", ha="center", va="center", fontsize=10, fontweight="bold", color="#15803D")
    ax.text(11.9, 4.68, "Val Loss: 0.4397", ha="center", va="center", fontsize=8.5, color="#166534")
    ax.text(13.9, 4.68, "Macro F1: 90.19%", ha="center", va="center", fontsize=8.5, color="#166534")

    ax.text(11.0, 4.25, "• Evaluated on 2,743 Disjoint Test Clips (Zero Leakage)", ha="left", va="center", fontsize=7.8, color="#1E293B")
    ax.text(11.0, 4.0, "• 82×82 Normalized Confusion Matrix verification", ha="left", va="center", fontsize=7.8, color="#1E293B")
    ax.text(11.0, 3.75, "• Serializes best_model.pt for production deployment", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#15803D")

    # Arrows in Phase 4
    # Concat down to MLP
    ax.annotate("", xy=(3.1, 6.0), xytext=(5.5, 6.98), arrowprops=dict(arrowstyle="->", lw=1.5, color="#DC2626"))
    # MLP to Loss
    ax.annotate("", xy=(5.6, 4.75), xytext=(5.3, 4.75), arrowprops=dict(arrowstyle="->", lw=1.5, color="#DC2626"))
    # Loss backpropagation loop back up
    ax.annotate("", xy=(8.0, 6.7), xytext=(8.0, 6.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#DC2626"))
    ax.text(8.2, 6.35, "∇ Loss (Backpropagation)", fontsize=8, fontweight="bold", color="#DC2626")
    # Test split into evaluation
    ax.annotate("", xy=(12.9, 6.0), xytext=(14.2, 17.25),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#16A34A", connectionstyle="arc3,rad=-0.1"))
    ax.text(14.3, 10.5, "30% Test Set\n(2,743 clips)", fontsize=8, fontweight="bold", color="#15803D")

    # -------------------------------------------------------------
    # PHASE 5: REAL-TIME DEPLOYMENT & INFERENCE ENGINE
    # -------------------------------------------------------------
    draw_box(0.6, 0.7, 14.8, 2.5, "#F0FDFA", "#0D9488", lw=2, radius=0.05)
    ax.text(1.0, 3.0, "PHASE 5: REAL-TIME INFERENCE PIPELINE & MULTI-PLATFORM DEPLOYMENT",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color="#115E59")

    # 5A: Real-time source & circular queue
    draw_box(0.9, 0.85, 4.2, 1.85, "#FFFFFF", "#0D9488", lw=1.5)
    ax.text(3.0, 2.45, "Live Stream & Circular Queue", ha="center", va="center", fontsize=10, fontweight="bold", color="#0F766E")
    ax.plot([1.2, 4.8], [2.3, 2.3], color="#99F6E4", lw=1)
    
    live_lines = [
        "• 30+ FPS Live Camera / Video (.mp4)",
        "• Circular Sliding Window: T = 16 Frames",
        "• Stride = 1 Frame (Continuous Stream)",
        "• Thread-safe asynchronous buffer",
        "• Zero frame drops with hardware decode"
    ]
    for i, line in enumerate(live_lines):
        ax.text(1.1, 2.05 - i * 0.25, line, ha="left", va="center", fontsize=8, color="#1E293B")

    # 5B: ActionInferenceEngine
    draw_box(5.4, 0.85, 4.9, 1.85, "#FFFFFF", "#0D9488", lw=1.5)
    ax.text(7.85, 2.45, "ActionInferenceEngine (Runtime)", ha="center", va="center", fontsize=10, fontweight="bold", color="#0F766E")
    ax.plot([5.7, 10.0], [2.3, 2.3], color="#99F6E4", lw=1)

    engine_lines = [
        "• Core Script: src/inference/engine.py",
        "• Checkpoint: Serialized best_model.pt (Epoch 23)",
        "• Latency: ~18.4 ms / forward pass on GPU",
        "• Full End-to-End: < 33.3 ms (30+ FPS Real-Time)",
        "• Real-time Temporal Attention Saliency (α_t)"
    ]
    for i, line in enumerate(engine_lines):
        ax.text(5.6, 2.05 - i * 0.25, line, ha="left", va="center", fontsize=8, color="#1E293B")

    # 5C: Multi-Platform UI & Prediction
    draw_box(10.6, 0.85, 4.5, 1.85, "#FFFFFF", "#0D9488", lw=1.5)
    ax.text(12.85, 2.45, "Multi-Platform Deployment & HUD", ha="center", va="center", fontsize=10, fontweight="bold", color="#0F766E")
    ax.plot([10.9, 14.8], [2.3, 2.3], color="#99F6E4", lw=1)

    ui_lines = [
        "• Outputs: Top-1 & Top-5 Probabilities (82 Classes)",
        "• Streamlit Web Dashboard: app.py",
        "• Native Desktop GUI: src/ui/desktop_app.py",
        "• Live Camera Stream HUD: src/ui/video_stream.py",
        "• Real-time confidence bar & joint skeleton HUD"
    ]
    for i, line in enumerate(ui_lines):
        ax.text(10.8, 2.05 - i * 0.25, line, ha="left", va="center", fontsize=8, color="#1E293B")

    # Arrows in Phase 5
    # Live to Engine
    ax.annotate("", xy=(5.4, 1.77), xytext=(5.1, 1.77), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0D9488"))
    # Engine to UI
    ax.annotate("", xy=(10.6, 1.77), xytext=(10.3, 1.77), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0D9488"))
    # Best model to Engine
    ax.annotate("", xy=(7.85, 2.7), xytext=(12.9, 3.55),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#0D9488", connectionstyle="arc3,rad=0.15"))
    ax.text(9.7, 3.2, "Serialized best_model.pt", fontsize=8, fontweight="bold", color="#0F766E")

    # -------------------------------------------------------------
    # CAPTION
    # -------------------------------------------------------------
    ax.text(8.0, 0.35, "Fig. 1. End-to-End Methodology Architecture Diagram of the Proposed CMAT-HOI Multimodal Sports Action Recognition Framework.",
            ha="center", va="center", fontsize=10.5, fontfamily="serif", fontstyle="italic", color="#334155")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG to: {output_png}")
    print(f"Saved SVG to: {output_svg}")


if __name__ == "__main__":
    out_dir_m7 = Path("diagrams/M7")
    out_dir_paper = Path("Paper writing/figures")

    render_methodology_diagram(
        out_dir_m7 / "methodology_architecture_diagram.png",
        out_dir_m7 / "methodology_architecture_diagram.svg"
    )
    render_methodology_diagram(
        out_dir_paper / "methodology_architecture_diagram.png",
        out_dir_paper / "methodology_architecture_diagram.svg"
    )
