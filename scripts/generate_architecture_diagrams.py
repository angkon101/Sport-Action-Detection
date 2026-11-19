import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_system_architecture(output_path: Path):
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Colors
    c_bg = "#F8FAFC"
    c_input = "#E0E7FF"
    c_input_border = "#4338CA"
    c_stem = "#EDE9FE"
    c_stem_border = "#6D28D9"
    c_cross = "#FCE7F3"
    c_cross_border = "#BE185D"
    c_gated = "#FEF3C7"
    c_gated_border = "#B45309"
    c_trans = "#D1FAE5"
    c_trans_border = "#047857"
    c_pool = "#CFFAFE"
    c_pool_border = "#0E7490"
    c_head = "#FEE2E2"
    c_head_border = "#B91C1C"

    # Title
    ax.text(8, 10.5, "CMAT-HOI: Cross-Modal Attention Transformer Architecture", 
            ha="center", va="center", fontsize=18, fontweight="bold", color="#0F172A")
    ax.text(8, 10.15, "End-to-End Multimodal Framework for Fine-Grained Sports Action Recognition (82 Classes)", 
            ha="center", va="center", fontsize=12, color="#475569")

    # 1. Raw Input
    rect_raw = patches.FancyBboxPatch((4.5, 9.2), 7, 0.65, boxstyle="round,pad=0.08", 
                                      facecolor=c_input, edgecolor=c_input_border, linewidth=2)
    ax.add_patch(rect_raw)
    ax.text(8, 9.525, "Raw Sports Video Input (T = 16 Uniformly Sampled Frames)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1E1B4B")

    # Arrows down from Raw to 4 Streams
    stream_x = [2.2, 6.0, 10.0, 13.8]
    stream_names = [
        ("Visual Stream", "ResNet-50", "[16, 2048]", "Scene context & background"),
        ("Pose Stream", "Keypoint R-CNN", "[16, 85]", "17 skeleton joints + velocities"),
        ("Object Stream", "Faster R-CNN", "[16, 7]", "Sports equipment bboxes & conf"),
        ("HOI Stream", "Kinematic Engine", "[16, 16]", "Wrist/ankle-gear vectors & dist")
    ]

    for sx in stream_x:
        ax.annotate("", xy=(sx, 8.5), xytext=(8, 9.2),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748B", connectionstyle="arc3,rad=0"))

    # 2. 4 Stream Feature Boxes
    for i, (sx, (sname, extractor, sdim, sdesc)) in enumerate(zip(stream_x, stream_names)):
        r = patches.FancyBboxPatch((sx - 1.65, 7.35), 3.3, 1.15, boxstyle="round,pad=0.08", 
                                   facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.5)
        ax.add_patch(r)
        ax.text(sx, 8.2, sname, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")
        ax.text(sx, 7.85, f"{extractor}  •  {sdim}", ha="center", va="center", fontsize=8.5, fontweight="semibold", color="#2563EB")
        ax.text(sx, 7.55, sdesc, ha="center", va="center", fontsize=7.5, color="#64748B")

        # Arrow down to Stem
        ax.annotate("", xy=(sx, 6.7), xytext=(sx, 7.35),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748B"))

    # 3. Modality Stems
    for sx in stream_x:
        r = patches.FancyBboxPatch((sx - 1.65, 5.9), 3.3, 0.8, boxstyle="round,pad=0.08", 
                                   facecolor=c_stem, edgecolor=c_stem_border, linewidth=1.5)
        ax.add_patch(r)
        ax.text(sx, 6.45, "Unified Modality Stem", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#4C1D95")
        ax.text(sx, 6.1, "Conv1D + LayerNorm + GELU (d=256)", ha="center", va="center", fontsize=8, color="#5B21B6")

    # 4. Bi-Directional Cross-Attention (Pose <-> Object)
    r_cross = patches.FancyBboxPatch((4.2, 4.65), 7.6, 0.85, boxstyle="round,pad=0.08", 
                                    facecolor=c_cross, edgecolor=c_cross_border, linewidth=2)
    ax.add_patch(r_cross)
    ax.text(8, 5.2, "Bi-Directional Cross-Attention Mechanism (Multi-Head Attn, h=4)", 
            ha="center", va="center", fontsize=10, fontweight="bold", color="#831843")
    ax.text(8, 4.85, "Pose queries Equipment (Q_p, K_o, V_o)  ◄──►  Equipment queries Pose (Q_o, K_p, V_p)", 
            ha="center", va="center", fontsize=8.5, color="#9D174D")

    # Arrows into Cross-Attention from Pose and Object stems
    ax.annotate("", xy=(6.5, 5.5), xytext=(6.0, 5.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="#BE185D"))
    ax.annotate("", xy=(9.5, 5.5), xytext=(10.0, 5.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="#BE185D"))

    # 5. Gated Multimodal Fusion Unit (GMFU)
    r_gmfu = patches.FancyBboxPatch((2.0, 3.45), 12.0, 0.8, boxstyle="round,pad=0.08", 
                                    facecolor=c_gated, edgecolor=c_gated_border, linewidth=2)
    ax.add_patch(r_gmfu)
    ax.text(8, 4.0, "Gated Multimodal Fusion Unit (GMFU)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="#78350F")
    ax.text(8, 3.65, "Softmax Modality Gates g_t = Softmax(W_g [V; P_enh; O_enh; H])  •  Dynamic Modality Weighting (d=256)", 
            ha="center", va="center", fontsize=8.5, color="#92400E")

    # Arrows into GMFU
    # Visual stem straight down
    ax.annotate("", xy=(3.0, 4.25), xytext=(2.2, 5.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="#B45309"))
    # Cross-attention down
    ax.annotate("", xy=(8.0, 4.25), xytext=(8.0, 4.65), arrowprops=dict(arrowstyle="->", lw=1.5, color="#B45309"))
    # HOI stem down
    ax.annotate("", xy=(13.0, 4.25), xytext=(13.8, 5.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="#B45309"))

    # 6. Spatio-Temporal Positional Encoding & Transformer Encoder
    ax.annotate("", xy=(8, 3.05), xytext=(8, 3.45), arrowprops=dict(arrowstyle="->", lw=2, color="#047857"))

    r_trans = patches.FancyBboxPatch((2.5, 1.85), 11.0, 1.2, boxstyle="round,pad=0.08", 
                                     facecolor=c_trans, edgecolor=c_trans_border, linewidth=2)
    ax.add_patch(r_trans)
    ax.text(8, 2.75, "Spatio-Temporal Transformer Encoder (Pre-LayerNorm)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="#064E3B")
    ax.text(8, 2.4, "+ Sinusoidal Positional Encoding & Learnable [CLS] Token", 
            ha="center", va="center", fontsize=9, color="#065F46")
    ax.text(8, 2.05, "3 Encoder Layers  •  8 Attention Heads  •  d_model = 256  •  d_ff = 1024  •  GELU", 
            ha="center", va="center", fontsize=8.5, fontweight="semibold", color="#047857")

    # 7. Multi-Scale 4-Way Temporal Pooling
    ax.annotate("", xy=(8, 1.45), xytext=(8, 1.85), arrowprops=dict(arrowstyle="->", lw=2, color="#0E7490"))

    pool_x = [3.2, 6.4, 9.6, 12.8]
    pool_labels = [
        ("[CLS] Token", "Contextual Anchor (256-d)"),
        ("Temporal Attention", "Learned α_t Saliency (256-d)"),
        ("Global Max Pooling", "Peak Motion Frame (256-d)"),
        ("Global Avg Pooling", "Mean Trajectory (256-d)")
    ]
    for px, (plabel, psub) in zip(pool_x, pool_labels):
        r = patches.FancyBboxPatch((px - 1.4, 0.85), 2.8, 0.6, boxstyle="round,pad=0.06", 
                                   facecolor=c_pool, edgecolor=c_pool_border, linewidth=1.2)
        ax.add_patch(r)
        ax.text(px, 1.25, plabel, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#155E75")
        ax.text(px, 1.0, psub, ha="center", va="center", fontsize=7.2, color="#0E7490")

    # 8. Classification Head & 82-Class Softmax Output
    r_head = patches.FancyBboxPatch((4.0, 0.05), 8.0, 0.65, boxstyle="round,pad=0.08", 
                                    facecolor=c_head, edgecolor=c_head_border, linewidth=2)
    ax.add_patch(r_head)
    ax.text(8, 0.5, "Concatenated Multimodal Vector (1024-d) ──► 3-Layer MLP Classifier", 
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#7F1D1D")
    ax.text(8, 0.22, "1024 ──► 512 ──► 256 ──► 82 Fine-Grained Sports Action Probabilities (Top-1: 90.19%, Top-5: 99.02%)", 
            ha="center", va="center", fontsize=8.2, fontweight="semibold", color="#991B1B")

    # Connect pool boxes down to classification head
    for px in pool_x:
        ax.annotate("", xy=(8, 0.7), xytext=(px, 0.85), 
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#B91C1C", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved architecture diagram to:", output_path)

def draw_dataset_split(output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Pie chart
    labels = ['Training Set (70% Stratified)', 'Testing Set (30% Stratified)']
    sizes = [6408, 2743]
    colors = ['#3B82F6', '#10B981']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140, pctdistance=0.75,
        textprops=dict(color="#1E293B", fontsize=10, fontweight="bold")
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(11)
        at.set_fontweight('bold')
        
    centre_circle = plt.Circle((0,0), 0.50, fc='white')
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, "9,151\nVideos", ha='center', va='center', fontsize=12, fontweight='bold', color="#0F172A")
    ax1.set_title("Strictly Disjoint Split (82 Classes)", fontsize=13, pad=12, fontweight="bold")

    # Bar chart of modalities
    streams = ['Visual (ResNet-50)', 'Pose (Keypoint R-CNN)', 'Object (Faster R-CNN)', 'HOI Kinematics']
    dims = [2048, 85, 7, 16]
    bcolors = ['#6366F1', '#EC4899', '#F59E0B', '#14B8A6']
    
    bars = ax2.barh(streams, dims, color=bcolors, height=0.55, edgecolor="black")
    ax2.set_xscale('log')
    ax2.set_xlabel("Feature Vector Dimension per Frame (Log Scale)", fontsize=10, fontweight="bold")
    ax2.set_title("Input Feature Stream Dimensionality (T=16)", fontsize=13, pad=12, fontweight="bold")
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar, d in zip(bars, dims):
        ax2.text(d * 1.25, bar.get_y() + bar.get_height() / 2, f"{d}-d", va='center', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved dataset split diagram to:", output_path)

if __name__ == "__main__":
    out_dir = Path("diagrams/M7")
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_system_architecture(out_dir / "system_architecture_diagram.png")
    draw_dataset_split(out_dir / "dataset_split_distribution.png")
