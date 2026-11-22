#!/usr/bin/env python3
"""
Research-backed System Model Architecture Diagram Generator for CMAT-HOI.

Synthesizes:
  - Empirical Transformer HOI Literature (HOTR, STIP, MSTR, Video Swin)
  - Codebase implementation in `src/models/hoi_transformer.py`
  - Guidelines from `Skills/` (figure-spec, latex-paper-skills)

Outputs:
  - `Paper writing/CMAT_HOI_Methodology_Architecture.drawio`
  - `diagrams/M7/CMAT_HOI_Methodology_Architecture.drawio`
  - `diagrams/M7/figure_spec.json`
  - `Paper writing/figures/methodology_architecture_diagram.svg`
  - `Paper writing/figures/methodology_architecture_diagram.png`
"""

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_drawio_xml(output_path: Path):
    root = ET.Element("mxfile", host="app.diagrams.net", version="24.0.0", type="device")
    diagram = ET.SubElement(root, "diagram", id="CMAT_HOI_System_Architecture", name="System Architecture")
    graph = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1400",
        dy="1000",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="1200",
        pageHeight="1150",
        math="0",
        shadow="0"
    )
    root_cells = ET.SubElement(graph, "root")
    ET.SubElement(root_cells, "mxCell", id="0")
    ET.SubElement(root_cells, "mxCell", id="1", parent="0")

    def add_node(nid, val, style, x, y, w, h):
        cell = ET.SubElement(root_cells, "mxCell", id=nid, value=val, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return cell

    def add_edge(eid, src, tgt, style, val="", exit_x=None, exit_y=None, entry_x=None, entry_y=None):
        edge_style = style
        if exit_x is not None and exit_y is not None:
            edge_style += f";exitX={exit_x};exitY={exit_y};exitDx=0;exitDy=0"
        if entry_x is not None and entry_y is not None:
            edge_style += f";entryX={entry_x};entryY={entry_y};entryDx=0;entryDy=0"
        cell = ET.SubElement(root_cells, "mxCell", id=eid, value=val, style=edge_style, edge="1", parent="1", source=src, target=tgt)
        ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        return cell

    # Base Styles
    st_banner = "rounded=1;arcSize=6;fillColor=#0F172A;strokeColor=#1E293B;strokeWidth=2;html=1;align=center;verticalAlign=middle;"
    st_group = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#94A3B8;strokeWidth=1.5;dashed=1;verticalAlign=top;spacingTop=8;fontStyle=1;fontSize=12;fontColor=#334155;container=1;collapsible=0;"
    st_group_model = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FFFDF5;strokeColor=#D97706;strokeWidth=2;verticalAlign=top;spacingTop=8;fontStyle=1;fontSize=13;fontColor=#92400E;container=1;collapsible=0;"
    st_block = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeWidth=1.5;align=center;verticalAlign=middle;shadow=1;"
    st_arr = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#475569;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#334155;"

    # Banner
    banner_val = (
        "<b style='font-size: 17px; color: #FFFFFF;'>CMAT-HOI: System Model Architecture Diagram</b><br>"
        "<span style='font-size: 11px; color: #94A3B8;'>Cross-Modal Attention Transformer with Human-Object Interaction for Sports Action Recognition (82 Classes)</span>"
    )
    add_node("banner", banner_val, st_banner, 40, 20, 1120, 55)

    # -------------------------------------------------------------
    # 1. VIDEO INGESTION & TEMPORAL SAMPLING
    # -------------------------------------------------------------
    add_node("g1", "<b>1. VIDEO INGESTION &amp; TEMPORAL SAMPLING</b>", st_group, 40, 90, 1120, 115)

    add_node("b_raw", "<b style='font-size:12px; color:#1E40AF;'>Raw Video Input</b><br><span style='font-size:10px; color:#475569;'>UCF-101 Subset (9,151 clips, 82 classes)</span>",
             st_block + "fillColor=#EFF6FF;strokeColor=#3B82F6;", 60, 125, 290, 60)

    add_node("b_sample", "<b style='font-size:12px; color:#1E40AF;'>Uniform Temporal Sampling</b><br><span style='font-size:10px; color:#475569;'>Extract T = 16 Equidistant Frames (224 × 224 px)</span>",
             st_block + "fillColor=#EFF6FF;strokeColor=#3B82F6;", 410, 125, 360, 60)

    add_node("b_seq", "<b style='font-size:12px; color:#1E40AF;'>Synchronized Clip Sequence</b><br><span style='font-size:10px; color:#475569;'>Sequence: { I_1, I_2, ..., I_16 } ∈ ℝ^(16 × 3 × 224 × 224)</span>",
             st_block + "fillColor=#EFF6FF;strokeColor=#3B82F6;", 830, 125, 310, 60)

    add_edge("e1_1", "b_raw", "b_sample", st_arr)
    add_edge("e1_2", "b_sample", "b_seq", st_arr)

    # -------------------------------------------------------------
    # 2. MULTIMODAL FEATURE EXTRACTION (4 PARALLEL STREAMS)
    # -------------------------------------------------------------
    add_node("g2", "<b>2. MULTIMODAL FEATURE EXTRACTION (4 SYNCHRONIZED PARALLEL STREAMS)</b>", st_group, 40, 220, 1120, 135)

    add_node("b_vis", "<b style='font-size:12px; color:#1D4ED8;'>Visual Appearance Stream</b><br><span style='font-size:10.5px; font-weight:bold; color:#1E293B;'>ResNet-50 (avgpool)</span><br><font style='font-size:10px; color:#2563EB;'><b>X_vis ∈ ℝ^(16 × 2048)</b></font>",
             st_block + "fillColor=#DBEAFE;strokeColor=#2563EB;", 60, 255, 255, 80)

    add_node("b_pose", "<b style='font-size:12px; color:#047857;'>Skeletal Kinematics Stream</b><br><span style='font-size:10.5px; font-weight:bold; color:#1E293B;'>Keypoint R-CNN (17 joints + vel)</span><br><font style='font-size:10px; color:#059669;'><b>X_pose ∈ ℝ^(16 × 85)</b></font>",
             st_block + "fillColor=#D1FAE5;strokeColor=#059669;", 345, 255, 255, 80)

    add_node("b_obj", "<b style='font-size:12px; color:#B45309;'>Sports Apparatus Stream</b><br><span style='font-size:10.5px; font-weight:bold; color:#1E293B;'>Faster R-CNN (FPN Detection)</span><br><font style='font-size:10px; color:#D97706;'><b>X_obj ∈ ℝ^(16 × 7)</b></font>",
             st_block + "fillColor=#FEF3C7;strokeColor=#D97706;", 630, 255, 255, 80)

    add_node("b_hoi", "<b style='font-size:12px; color:#BE185D;'>Geometric HOI Stream</b><br><span style='font-size:10.5px; font-weight:bold; color:#1E293B;'>Kinematic Interaction Engine</span><br><font style='font-size:10px; color:#DB2777;'><b>X_hoi ∈ ℝ^(16 × 16)</b></font>",
             st_block + "fillColor=#FCE7F3;strokeColor=#DB2777;", 915, 255, 225, 80)

    add_edge("e_seq_ext", "b_seq", "b_obj", st_arr, "Extract Streams", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # -------------------------------------------------------------
    # 3. PROPOSED CMAT-HOI NETWORK ARCHITECTURE
    # -------------------------------------------------------------
    add_node("g3", "<b>3. PROPOSED CMAT-HOI NETWORK ARCHITECTURE</b>", st_group_model, 40, 370, 1120, 480)

    add_node("b_stems", "<b style='font-size:12.5px; color:#92400E;'>Unified Modality Alignment Stems (d = 256)</b><br><span style='font-size:10px; color:#451A03;'>Linear(in_dim → 256) + LayerNorm + 1D Conv(k=3, p=1) + GELU ──► <b>E_vis, E_pose, E_obj, E_hoi ∈ ℝ^(16 × 256)</b></span>",
             st_block + "fillColor=#FEF9C3;strokeColor=#D97706;", 80, 410, 1040, 60)

    add_node("b_cross", "<b style='font-size:12.5px; color:#831843;'>Bi-Directional Keypoint–Equipment Cross-Attention Module (h = 4 Heads, d_k = 64)</b><br><span style='font-size:10px; color:#4C0519;'>Pose queries Equipment: <b>P_enh = LN(E_pose + MHA(Q_p, K_o, V_o))</b> &nbsp;◄──►&nbsp; Equipment queries Pose: <b>O_enh = LN(E_obj + MHA(Q_o, K_p, V_p))</b></span>",
             st_block + "fillColor=#FDF2F8;strokeColor=#BE185D;", 80, 500, 1040, 60)

    add_node("b_gmfu", "<b style='font-size:12.5px; color:#78350F;'>Gated Multimodal Fusion Unit (GMFU) — Dynamic Modality Attention Weighting</b><br><span style='font-size:10px; color:#451A03;'>Gating Vector: <b>g_t = Softmax(W_g · [E_vis ; P_enh ; O_enh ; E_hoi] + b_g) ∈ ℝ^4</b> &nbsp;►&nbsp; Fused Token: <b>F_t = ∑ g_t,m · E_m,t ∈ ℝ^256</b></span>",
             st_block + "fillColor=#FEF3C7;strokeColor=#B45309;", 80, 590, 1040, 60)

    add_node("b_trans", "<b style='font-size:12.5px; color:#064E3B;'>Spatio-Temporal Transformer Encoder (3 Layers, Pre-LayerNorm, 8 Heads, d_ff = 1024)</b><br><span style='font-size:10px; color:#022C22;'>Token Input Sequence: <b>Z_0 = [ z_CLS ; F_1 ; F_2 ; ... ; F_16 ] + E_pos ∈ ℝ^(17 × 256)</b> &nbsp;(Captures Full Temporal Motion Dynamics)</span>",
             st_block + "fillColor=#ECFDF5;strokeColor=#059669;", 80, 680, 1040, 60)

    add_node("b_pool", "<b style='font-size:12.5px; color:#0E7490;'>Multi-Scale 4-Way Temporal Pooling Head</b><br><span style='font-size:10px; color:#164E63;'>Concatenation: <b>Z_pool = [ z_CLS (256-d) ; z_Attn (256-d) ; z_Max (256-d) ; z_Avg (256-d) ] ∈ ℝ^1024</b> &nbsp;(Unified Action Descriptor)</span>",
             st_block + "fillColor=#CFFAFE;strokeColor=#0891B2;", 80, 770, 1040, 60)

    add_edge("e_s_stems1", "b_pose", "b_stems", st_arr, "", exit_x=0.5, exit_y=1.0, entry_x=0.3, entry_y=0.0)
    add_edge("e_s_stems2", "b_obj", "b_stems", st_arr, "", exit_x=0.5, exit_y=1.0, entry_x=0.7, entry_y=0.0)
    add_edge("e_stems_cross", "b_stems", "b_cross", st_arr)
    add_edge("e_cross_gmfu", "b_cross", "b_gmfu", st_arr)
    add_edge("e_gmfu_trans", "b_gmfu", "b_trans", st_arr)
    add_edge("e_trans_pool", "b_trans", "b_pool", st_arr)

    # -------------------------------------------------------------
    # 4. CLASSIFICATION HEAD, LOSS FORMULATION & RECOGNITION
    # -------------------------------------------------------------
    add_node("g4", "<b>4. CLASSIFICATION HEAD, LOSS FORMULATION &amp; RECOGNITION OUTPUT</b>", st_group, 40, 865, 1120, 195)

    add_node("b_mlp", "<b style='font-size:12px; color:#B91C1C;'>3-Layer MLP Classifier Head</b><br><span style='font-size:10px; color:#334155;'>Linear(1024 → 512) + LN + GELU + Drop(0.3)<br>──► Linear(512 → 256) ──► Linear(256 → 82)<br><b>Logits: ŷ ∈ ℝ^82</b></span>",
             st_block + "fillColor=#FEE2E2;strokeColor=#DC2626;", 60, 905, 330, 95)

    add_node("b_loss", "<b style='font-size:12px; color:#9F1239;'>Focal Loss with Label Smoothing</b><br><span style='font-size:10px; color:#334155;'><b>L_focal = - ∑ α_c · (1 - p_c)^γ · y_c^(smooth) · log(p_c)</b><br>Focusing γ = 1.5, Balancing α_c = 0.08, Smoothing ε = 0.08<br>AdamW (lr=1e-4) + Cosine Annealing</span>",
             st_block + "fillColor=#FFF1F2;strokeColor=#E11D48;", 420, 905, 340, 95)

    add_node("b_eval", "<b style='font-size:12px; color:#15803D;'>82-Class Action Recognition</b><br><span style='font-size:11.5px; font-weight:bold; color:#14532D;'>Top-1 Accuracy: 90.19%<br>Top-5 Accuracy: 99.02%</span><br><font style='font-size:9.5px; color:#166534;'>Evaluated on 2,743 Disjoint Test Clips (Peak Epoch 23)</font>",
             st_block + "fillColor=#DCFCE7;strokeColor=#16A34A;", 790, 905, 350, 95)

    add_edge("e_pool_mlp", "b_pool", "b_mlp", st_arr, "Z_pool (1024-d)", exit_x=0.2, exit_y=1.0, entry_x=0.5, entry_y=0.0)
    add_edge("e_mlp_loss", "b_mlp", "b_loss", st_arr, "Logits ŷ")
    add_edge("e_loss_eval", "b_loss", "b_eval", st_arr, "Validation")
    add_edge("e_loss_backprop", "b_loss", "b_trans", st_arr, "∇ Loss (Backpropagation)", exit_x=0.5, exit_y=0.0, entry_x=0.5, entry_y=1.0)

    # Caption
    caption = "<b>Fig. 1.</b> System Model Architecture of the Proposed CMAT-HOI Framework for Sports Action Recognition (82 Classes)."
    add_node("cap", caption, "text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#475569;fontStyle=2;", 40, 1075, 1120, 25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    print(f"Generated Draw.io diagram at: {output_path}")


def generate_figure_spec(output_json: Path):
    spec = {
        "title": "CMAT-HOI System Model Architecture Diagram",
        "canvas": {"width": 1000, "height": 920},
        "style": {
            "font_family": "Arial, Helvetica, sans-serif",
            "font_size": 13,
            "bg_color": "#FFFFFF",
            "palette": ["#2563EB", "#059669", "#D97706", "#DB2777", "#DC2626", "#16A34A"]
        },
        "nodes": [
            # Group 1: Data Ingestion
            {"id": "raw", "label": "Raw Video Input", "sublabel": "UCF-101 (9,151 clips, 82 classes)", "x": 160, "y": 80, "width": 240, "height": 55, "shape": "rounded", "fill": "#EFF6FF", "stroke": "#3B82F6"},
            {"id": "samp", "label": "Uniform Temporal Sampling", "sublabel": "Extract T = 16 Frames (224×224 px)", "x": 480, "y": 80, "width": 260, "height": 55, "shape": "rounded", "fill": "#EFF6FF", "stroke": "#3B82F6"},
            {"id": "seq", "label": "Synchronized Sequence", "sublabel": "{I_1, ..., I_16} ∈ ℝ^(16×3×224×224)", "x": 800, "y": 80, "width": 260, "height": 55, "shape": "rounded", "fill": "#EFF6FF", "stroke": "#3B82F6"},

            # Group 2: Extraction
            {"id": "vis", "label": "Visual Appearance", "sublabel": "ResNet-50 [16 × 2048]", "x": 140, "y": 195, "width": 190, "height": 55, "shape": "rounded", "fill": "#DBEAFE", "stroke": "#2563EB"},
            {"id": "pose", "label": "Skeletal Kinematics", "sublabel": "Keypoint R-CNN [16 × 85]", "x": 380, "y": 195, "width": 190, "height": 55, "shape": "rounded", "fill": "#D1FAE5", "stroke": "#059669"},
            {"id": "obj", "label": "Sports Apparatus", "sublabel": "Faster R-CNN [16 × 7]", "x": 620, "y": 195, "width": 190, "height": 55, "shape": "rounded", "fill": "#FEF3C7", "stroke": "#D97706"},
            {"id": "hoi", "label": "Geometric HOI", "sublabel": "Kinematic Engine [16 × 16]", "x": 860, "y": 195, "width": 190, "height": 55, "shape": "rounded", "fill": "#FCE7F3", "stroke": "#DB2777"},

            # Group 3: Core Model
            {"id": "stems", "label": "Unified Modality Alignment Stems", "sublabel": "Linear + LayerNorm + 1D Conv (d = 256 for all 4 streams)", "x": 500, "y": 300, "width": 800, "height": 50, "shape": "rounded", "fill": "#FEF9C3", "stroke": "#D97706"},
            {"id": "cross", "label": "Bi-Directional Keypoint–Equipment Cross-Attention", "sublabel": "Pose Queries Equipment (P_enh) ◄──► Equipment Queries Pose (O_enh)", "x": 500, "y": 375, "width": 800, "height": 50, "shape": "rounded", "fill": "#FDF2F8", "stroke": "#BE185D"},
            {"id": "gmfu", "label": "Gated Multimodal Fusion Unit (GMFU)", "sublabel": "Dynamic Modality Weights g_t = Softmax(W_g · [V; P; O; H]) ──► Fused F_t", "x": 500, "y": 450, "width": 800, "height": 50, "shape": "rounded", "fill": "#FEF3C7", "stroke": "#B45309"},
            {"id": "trans", "label": "Spatio-Temporal Transformer Encoder", "sublabel": "3 Layers, 8 Attention Heads, Positional Encoding + Learnable [z_CLS]", "x": 500, "y": 525, "width": 800, "height": 50, "shape": "rounded", "fill": "#ECFDF5", "stroke": "#059669"},
            {"id": "pool", "label": "Multi-Scale 4-Way Temporal Pooling Head", "sublabel": "Z_pool = [ z_CLS ; Attention Saliency ; Max Pool ; Avg Pool ] ∈ ℝ^1024", "x": 500, "y": 600, "width": 800, "height": 50, "shape": "rounded", "fill": "#CFFAFE", "stroke": "#0891B2"},

            # Group 4: Classification & Evaluation
            {"id": "mlp", "label": "3-Layer MLP Classifier", "sublabel": "Linear(1024 → 512 → 256 → 82)", "x": 200, "y": 720, "width": 260, "height": 60, "shape": "rounded", "fill": "#FEE2E2", "stroke": "#DC2626"},
            {"id": "loss", "label": "Focal Loss & Label Smoothing", "sublabel": "γ = 1.5, α = 0.08, ε = 0.08", "x": 500, "y": 720, "width": 260, "height": 60, "shape": "rounded", "fill": "#FFF1F2", "stroke": "#E11D48"},
            {"id": "eval", "label": "82-Class Action Recognition", "sublabel": "Top-1: 90.19% | Top-5: 99.02% (Epoch 23)", "x": 800, "y": 720, "width": 260, "height": 60, "shape": "rounded", "fill": "#DCFCE7", "stroke": "#16A34A"}
        ],
        "edges": [
            {"from": "raw", "to": "samp", "label": "Video"},
            {"from": "samp", "to": "seq", "label": "T=16"},
            {"from": "seq", "to": "obj", "label": "Extract"},
            {"from": "stems", "to": "cross"},
            {"from": "cross", "to": "gmfu"},
            {"from": "gmfu", "to": "trans"},
            {"from": "trans", "to": "pool"},
            {"from": "pool", "to": "mlp", "label": "1024-d"},
            {"from": "mlp", "to": "loss", "label": "Logits"},
            {"from": "loss", "to": "eval", "label": "Epoch 23"}
        ],
        "groups": [
            {"id": "grp_data", "label": "1. VIDEO INGESTION & TEMPORAL SAMPLING", "node_ids": ["raw", "samp", "seq"], "fill": "#F8FAFC", "stroke": "#94A3B8"},
            {"id": "grp_ext", "label": "2. MULTIMODAL FEATURE EXTRACTION BACKBONES", "node_ids": ["vis", "pose", "obj", "hoi"], "fill": "#FAF5FF", "stroke": "#9333EA"},
            {"id": "grp_model", "label": "3. PROPOSED CMAT-HOI NETWORK ARCHITECTURE", "node_ids": ["stems", "cross", "gmfu", "trans", "pool"], "fill": "#FFFDF5", "stroke": "#D97706"},
            {"id": "grp_opt", "label": "4. CLASSIFICATION HEAD, LOSS FORMULATION & RECOGNITION", "node_ids": ["mlp", "loss", "eval"], "fill": "#FEF2F2", "stroke": "#EF4444"}
        ]
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"Generated FigureSpec JSON at: {output_json}")


def render_with_figure_spec_skill(spec_json: Path, output_svg: Path):
    renderer_script = Path("Skills/Auto-claude-code-research-in-sleep/skills/figure-spec/scripts/figure_renderer.py")
    if renderer_script.exists():
        cmd = ["python", str(renderer_script), "render", str(spec_json), "--output", str(output_svg)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("FigureSpec Renderer Output:\n", res.stdout)
        if res.stderr:
            print("FigureSpec Renderer Stderr:\n", res.stderr)


def render_companion_png(output_png: Path):
    fig, ax = plt.subplots(figsize=(12, 11.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    def card(x, y, w, h, bg, stroke, lw=1.5, ls="-"):
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", facecolor=bg, edgecolor=stroke, linewidth=lw, linestyle=ls)
        ax.add_patch(r)
        return r

    # Title
    card(0.4, 10.7, 11.2, 0.6, "#0F172A", "#1E293B", lw=2)
    ax.text(6.0, 11.08, "CMAT-HOI: System Model Architecture Diagram", ha="center", va="center", fontsize=13, fontweight="bold", color="#FFFFFF")
    ax.text(6.0, 10.84, "Cross-Modal Attention Transformer with Human-Object Interaction for Sports Action Recognition (82 Classes)", ha="center", va="center", fontsize=8.2, color="#94A3B8")

    # 1. Video Ingestion & Temporal Sampling
    card(0.4, 9.2, 11.2, 1.35, "#F8FAFC", "#94A3B8", lw=1.2, ls="--")
    ax.text(0.7, 10.35, "1. VIDEO INGESTION & TEMPORAL SAMPLING", fontsize=9.5, fontweight="bold", color="#334155")
    
    card(0.6, 9.35, 3.2, 0.8, "#EFF6FF", "#3B82F6")
    ax.text(2.2, 9.85, "Raw Sports Video Input", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(2.2, 9.55, "UCF-101 (82 Classes, 9,151 clips)", ha="center", fontsize=8, color="#475569")

    card(4.3, 9.35, 3.4, 0.8, "#EFF6FF", "#3B82F6")
    ax.text(6.0, 9.85, "Uniform Temporal Sampling", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(6.0, 9.55, "Extract T = 16 Frames (224×224 px)", ha="center", fontsize=8, color="#475569")

    card(8.2, 9.35, 3.2, 0.8, "#EFF6FF", "#3B82F6")
    ax.text(9.8, 9.85, "Synchronized Clip Sequence", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(9.8, 9.55, "{I_1, ..., I_16} Normalized Tensors", ha="center", fontsize=8, color="#475569")

    ax.annotate("", xy=(4.3, 9.75), xytext=(3.8, 9.75), arrowprops=dict(arrowstyle="->", lw=1.5, color="#3B82F6"))
    ax.annotate("", xy=(8.2, 9.75), xytext=(7.7, 9.75), arrowprops=dict(arrowstyle="->", lw=1.5, color="#3B82F6"))

    # 2. 4-Stream Feature Extraction
    card(0.4, 7.6, 11.2, 1.45, "#FAF5FF", "#9333EA", lw=1.2, ls="--")
    ax.text(0.7, 8.85, "2. MULTIMODAL FEATURE EXTRACTION BACKBONES", fontsize=9.5, fontweight="bold", color="#6B21A8")

    streams = [
        ("Visual Appearance", "ResNet-50 (avgpool)", "X_vis ∈ ℝ^(16 × 2048)", "#DBEAFE", "#2563EB", 0.6),
        ("Skeletal Kinematics", "Keypoint R-CNN", "X_pose ∈ ℝ^(16 × 85)", "#D1FAE5", "#059669", 3.4),
        ("Sports Apparatus", "Faster R-CNN (FPN)", "X_obj ∈ ℝ^(16 × 7)", "#FEF3C7", "#D97706", 6.2),
        ("Geometric HOI", "Kinematic Engine", "X_hoi ∈ ℝ^(16 × 16)", "#FCE7F3", "#DB2777", 9.0)
    ]
    for stitle, smodel, sdim, sfill, sstroke, sx in streams:
        card(sx, 7.75, 2.4, 0.85, sfill, sstroke)
        ax.text(sx + 1.2, 8.32, stitle, ha="center", fontsize=8.5, fontweight="bold", color=sstroke)
        ax.text(sx + 1.2, 8.08, smodel, ha="center", fontsize=7.8, color="#1E293B")
        ax.text(sx + 1.2, 7.88, sdim, ha="center", fontsize=8, fontweight="bold", color=sstroke)

    ax.annotate("", xy=(6.0, 9.2), xytext=(9.8, 9.35), arrowprops=dict(arrowstyle="->", lw=1.2, color="#9333EA", connectionstyle="arc3,rad=-0.15"))

    # 3. CMAT-HOI Core Transformer Network
    card(0.4, 2.7, 11.2, 4.7, "#FFFDF5", "#D97706", lw=1.8)
    ax.text(0.7, 7.18, "3. PROPOSED CMAT-HOI NETWORK ARCHITECTURE", fontsize=10, fontweight="bold", color="#92400E")

    modules = [
        ("1. Unified Modality Alignment Stems", "Linear + LayerNorm + 1D Conv (k=3, p=1) + GELU ──► E_vis, E_pose, E_obj, E_hoi ∈ ℝ^(16 × 256)", "#FEF9C3", "#D97706", 6.35),
        ("2. Bi-Directional Keypoint–Equipment Cross-Attention", "Pose Queries Equipment (P_enh) ◄──► Equipment Queries Pose (O_enh) (h = 4 Heads, d_k = 64)", "#FDF2F8", "#BE185D", 5.45),
        ("3. Gated Multimodal Fusion Unit (GMFU)", "Dynamic Modality Gating: g_t = Softmax(W_g · [V; P; O; H]) ──► Fused Token: F_t = ∑ g_t,m · E_m,t ∈ ℝ^256", "#FEF3C7", "#B45309", 4.55),
        ("4. Spatio-Temporal Transformer Encoder", "3 Layers, Pre-LN, 8 Attention Heads, Positional Encoding + Learnable [z_CLS] Token", "#ECFDF5", "#059669", 3.65),
        ("5. Multi-Scale 4-Way Temporal Pooling Head", "Concatenation: Z_pool = [ z_CLS ; Attention Saliency ; Max Pool ; Avg Pool ] ∈ ℝ^1024", "#CFFAFE", "#0891B2", 2.75)
    ]
    for mtitle, mdesc, mfill, mstroke, my in modules:
        card(0.8, my + 0.1, 10.4, 0.65, mfill, mstroke)
        ax.text(6.0, my + 0.52, mtitle, ha="center", fontsize=9, fontweight="bold", color=mstroke)
        ax.text(6.0, my + 0.28, mdesc, ha="center", fontsize=8, color="#1E293B")

    for i in range(len(modules) - 1):
        y_top = modules[i][4] + 0.1
        y_bot = modules[i+1][4] + 0.75
        ax.annotate("", xy=(6.0, y_bot), xytext=(6.0, y_top), arrowprops=dict(arrowstyle="->", lw=1.2, color="#D97706"))

    # 4. Classification & Evaluation
    card(0.4, 0.9, 11.2, 1.6, "#FEF2F2", "#DC2626", lw=1.4)
    ax.text(0.7, 2.3, "4. CLASSIFICATION HEAD, LOSS FORMULATION & RECOGNITION OUTPUT", fontsize=9.5, fontweight="bold", color="#991B1B")

    card(0.6, 1.05, 3.4, 1.0, "#FEE2E2", "#DC2626")
    ax.text(2.3, 1.75, "3-Layer MLP Classifier", ha="center", fontsize=9, fontweight="bold", color="#B91C1C")
    ax.text(2.3, 1.5, "Linear(1024 → 512 → 256 → 82)", ha="center", fontsize=7.8, color="#1E293B")
    ax.text(2.3, 1.25, "Logits: ŷ ∈ ℝ^82 (Probabilities)", ha="center", fontsize=7.5, fontweight="bold", color="#991B1B")

    card(4.3, 1.05, 3.4, 1.0, "#FFF1F2", "#E11D48")
    ax.text(6.0, 1.75, "Focal Loss & Label Smoothing", ha="center", fontsize=9, fontweight="bold", color="#9F1239")
    ax.text(6.0, 1.5, "L_focal (γ = 1.5, α_c = 0.08, ε = 0.08)", ha="center", fontsize=7.8, fontweight="bold", color="#BE123C")
    ax.text(6.0, 1.25, "AdamW Optimizer + Cosine Annealing", ha="center", fontsize=7.2, color="#475569")

    card(8.0, 1.05, 3.4, 1.0, "#DCFCE7", "#16A34A")
    ax.text(9.7, 1.8, "82-Class Action Recognition", ha="center", fontsize=9, fontweight="bold", color="#15803D")
    ax.text(9.7, 1.55, "Top-1: 90.19% | Top-5: 99.02%", ha="center", fontsize=9.5, fontweight="bold", color="#14532D")
    ax.text(9.7, 1.25, "2,743 Disjoint Test Clips (Epoch 23)", ha="center", fontsize=7.5, color="#166534")

    ax.annotate("", xy=(2.3, 2.05), xytext=(2.3, 2.85), arrowprops=dict(arrowstyle="->", lw=1.2, color="#DC2626"))
    ax.annotate("", xy=(4.3, 1.55), xytext=(4.0, 1.55), arrowprops=dict(arrowstyle="->", lw=1.2, color="#DC2626"))
    ax.annotate("", xy=(8.0, 1.55), xytext=(7.7, 1.55), arrowprops=dict(arrowstyle="->", lw=1.2, color="#16A34A"))

    # Caption
    ax.text(6.0, 0.45, "Fig. 1. System Model Architecture of the Proposed CMAT-HOI Framework for Sports Action Recognition (82 Classes).",
            ha="center", va="center", fontsize=9, fontfamily="serif", fontstyle="italic", color="#475569")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated companion PNG at: {output_png}")


if __name__ == "__main__":
    out_drawio_paper = Path("Paper writing/CMAT_HOI_Methodology_Architecture.drawio")
    out_drawio_m7 = Path("diagrams/M7/CMAT_HOI_Methodology_Architecture.drawio")
    generate_drawio_xml(out_drawio_paper)
    generate_drawio_xml(out_drawio_m7)

    spec_json = Path("diagrams/M7/figure_spec.json")
    generate_figure_spec(spec_json)

    out_svg_paper = Path("Paper writing/figures/methodology_architecture_diagram.svg")
    out_svg_m7 = Path("diagrams/M7/methodology_architecture_diagram.svg")
    render_with_figure_spec_skill(spec_json, out_svg_m7)
    render_with_figure_spec_skill(spec_json, out_svg_paper)

    out_png_paper = Path("Paper writing/figures/methodology_architecture_diagram.png")
    out_png_m7 = Path("diagrams/M7/methodology_architecture_diagram.png")
    render_companion_png(out_png_paper)
    render_companion_png(out_png_m7)
