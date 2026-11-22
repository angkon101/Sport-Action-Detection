#!/usr/bin/env python3
"""
Finetuned Reference-7 System Model Architecture Diagram Generator for CMAT-HOI.

Strictly aligned with:
  - Reference 7 (Real Inverted Trapezoid Funnel, Central Training Model with drawn neural network graph,
                 side-by-side loops for Model Evaluation & Hyperparameter Optimization,
                 bottom shaded container for Unseen Data -> Trained Model -> Action Prediction).
  - CMAT-HOI Mathematical Specifications in `Paper writing/METHODOLOGY_SPECIFICATION.md`
  - Implementation in `src/models/hoi_transformer.py`

Produces:
  - `Paper writing/CMAT_HOI_Methodology_Architecture.drawio`
  - `diagrams/M7/CMAT_HOI_Methodology_Architecture.drawio`
  - `Paper writing/figures/methodology_architecture_diagram.png`
  - `Paper writing/figures/methodology_architecture_diagram.svg`
  - `diagrams/M7/methodology_architecture_diagram.png`
  - `diagrams/M7/methodology_architecture_diagram.svg`
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def build_finetuned_drawio_xml(output_path: Path):
    root = ET.Element("mxfile", host="app.diagrams.net", version="24.0.0", type="device")
    diagram = ET.SubElement(root, "diagram", id="CMAT_HOI_Finetuned_Architecture", name="System Architecture")
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
        pageWidth="1360",
        pageHeight="1850",
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

    # Styles
    st_frame = "rounded=1;arcSize=3;fillColor=#F4F9FD;strokeColor=#2980B9;strokeWidth=2;html=1;container=1;collapsible=0;"
    st_dashed_card = "rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#64748B;strokeWidth=1.5;dashed=1;html=1;container=1;collapsible=0;"
    st_center_model = "rounded=1;arcSize=5;fillColor=#FEFDF8;strokeColor=#D4AC0D;strokeWidth=2;html=1;container=1;collapsible=0;shadow=1;"
    st_card_white = "rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;shadow=1;"
    st_arr = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1E293B;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;"
    st_arr_loop = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1E293B;strokeWidth=2;startArrow=classic;endArrow=classic;startFill=1;endFill=1;fontSize=10;"

    # Outer Frame Container (Reference 7 style)
    add_node("frame", "", st_frame, 20, 20, 1320, 1790)

    # -------------------------------------------------------------
    # 1. TOP DASHED BOX: DATA INGESTION & DATASET REPOSITORY
    # -------------------------------------------------------------
    add_node("box_top_coll", "", st_dashed_card, 160, 45, 1020, 95)

    add_node("n_coll", "<b>🏃 Video Acquisition</b><br><span style='font-size:10px; color:#475569;'>Sports Action Capture</span>",
             st_card_white + "fillColor=#EFF6FF;strokeColor=#3B82F6;", 190, 62, 220, 60)

    add_node("n_repo", "<b>📁 UCF-101 Benchmark</b><br><span style='font-size:10px; color:#475569;'>Action Dataset Repository</span>",
             st_card_white + "fillColor=#E0F2FE;strokeColor=#0284C7;", 560, 62, 240, 60)

    add_node("n_dataset", "<b>📊 Curated Dataset</b><br><span style='font-size:10px; color:#1E40AF;'><b>9,151 Videos &nbsp;•&nbsp; 82 Classes</b></span>",
             st_card_white + "fillColor=#DBEAFE;strokeColor=#2563EB;", 930, 62, 220, 60)

    add_edge("e_c1", "n_coll", "n_repo", st_arr)
    add_edge("e_c2", "n_repo", "n_dataset", st_arr)

    # -------------------------------------------------------------
    # 2. INVERTED PREPROCESSING FUNNEL (Reference 7 Exact Style)
    # -------------------------------------------------------------
    # Purple Side Arrow Badge
    add_node("funnel_badge", "<b>Data Preprocessing<br>&amp; Filtering</b>",
             "shape=singleArrow;direction=east;whiteSpace=wrap;html=1;fillColor=#E8DAEF;strokeColor=#8E44AD;strokeWidth=1.5;fontStyle=1;fontSize=11;fontColor=#4A235A;align=center;",
             90, 215, 195, 55)

    # Funnel Bands (Narrowing downwards)
    add_node("funnel_b1", "<b>1. Temporal Uniform Sampling:</b> T = 16 equidistant frames (224 × 224 px)",
             "rounded=1;arcSize=4;fillColor=#D5F5E3;strokeColor=#27AE60;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#145A32;",
             300, 165, 740, 32)

    add_node("funnel_b2", "<b>2. Missing Value Imputation:</b> Zero-fill missing joints &amp; Kalman gear tracking",
             "rounded=1;arcSize=4;fillColor=#FCF3CF;strokeColor=#F39C12;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#7D6608;",
             345, 202, 650, 32)

    add_node("funnel_b3", "<b>3. Kinematic Feature Extraction:</b> Displacement velocities Δp_t = p_t - p_(t-1)",
             "rounded=1;arcSize=4;fillColor=#D4E6F1;strokeColor=#2980B9;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#1B4F72;",
             395, 239, 550, 32)

    add_node("funnel_b4", "<b>4. Apparatus Quality Gating:</b> Gear confidence s &gt; 0.5 &amp; aspect verification",
             "rounded=1;arcSize=4;fillColor=#D1F2EB;strokeColor=#1ABC9C;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#0E6251;",
             450, 276, 440, 32)

    add_node("funnel_b5", "<b>5. Feature Normalization:</b> ImageNet zero-center standardization &amp; FP32 tensors",
             "rounded=1;arcSize=4;fillColor=#FADBD8;strokeColor=#E74C3C;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#78281F;",
             505, 313, 330, 32)

    # Funnel Stem Downward Flex Arrow
    add_node("funnel_arrow", "",
             "shape=flexArrow;direction=south;fillColor=#475569;strokeColor=#1E293B;strokeWidth=1.5;endArrow=classic;",
             645, 350, 50, 42)

    # Clean Orthogonal Connector from Dataset to Funnel
    add_edge("e_ds_funnel", "n_dataset", "funnel_b1", st_arr, "", exit_x=0.5, exit_y=1.0, entry_x=0.85, entry_y=0.0)

    # -------------------------------------------------------------
    # 3. SPLIT: TEST DATA (LEFT) & TRAIN DATA (RIGHT)
    # -------------------------------------------------------------
    add_node("card_test_data", "<b>🎯 Test Data (30%)</b><br><span style='font-size:12px; font-weight:bold; color:#1E40AF;'>2,743 Video Clips</span><br><font style='font-size:9px; color:#475569;'>Strictly Disjoint Unseen Subjects</font>",
             st_card_white + "fillColor=#FEF9E7;strokeColor=#D4AC0D;", 60, 415, 230, 75)

    add_node("card_train_data", "<b>🏋️ Train Data (70%)</b><br><span style='font-size:12px; font-weight:bold; color:#1D4ED8;'>6,408 Video Clips</span><br><font style='font-size:9px; color:#475569;'>Subject-Independent Training</font>",
             st_card_white + "fillColor=#FEF9E7;strokeColor=#D4AC0D;", 1050, 415, 230, 75)

    add_edge("e_sp_left", "funnel_arrow", "card_test_data", st_arr, "30% Disjoint Split", exit_x=0.2, exit_y=1.0, entry_x=0.5, entry_y=0.0)
    add_edge("e_sp_right", "funnel_arrow", "card_train_data", st_arr, "70% Training Split", exit_x=0.8, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # -------------------------------------------------------------
    # 4. CENTER CARD: TRAINING MODEL (CMAT-HOI FRAMEWORK)
    # -------------------------------------------------------------
    # Center Model Card (x=320, w=700)
    add_node("center_model", "<b style='font-size:14px; color:#78350F;'>Training Model: CMAT-HOI Framework</b>",
             st_center_model, 320, 415, 700, 1080)

    # 4.1 Multimodal Input Streams (Inner width = 580 px, centered at x=360 to 940)
    add_node("lbl_streams", "<b>Multimodal Input Streams (T = 16 Equidistant Frames)</b>",
             "text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#78350F;fontStyle=1;", 360, 448, 580, 20)

    stream_boxes = [
        ("m_vis", "Visual Appearance<br><b>ResNet-50</b><br><font color='#1D4ED8'><b>X_V ∈ ℝ^(16 × 2048)</b></font>", "#DBEAFE", "#2563EB", 360),
        ("m_pose", "Kinematic Pose<br><b>Keypoint R-CNN</b><br><font color='#047857'><b>X_P ∈ ℝ^(16 × 85)</b></font>", "#D1FAE5", "#059669", 510),
        ("m_obj", "Sports Equipment<br><b>Faster R-CNN</b><br><font color='#B45309'><b>X_O ∈ ℝ^(16 × 7)</b></font>", "#FEF3C7", "#D97706", 660),
        ("m_hoi", "Geometric HOI<br><b>Kinematic Engine</b><br><font color='#BE185D'><b>X_H ∈ ℝ^(16 × 16)</b></font>", "#FCE7F3", "#DB2777", 810)
    ]
    for mid, mlabel, mbg, mstroke, mx in stream_boxes:
        add_node(mid, f"<span style='font-size:9.5px;'>{mlabel}</span>",
                 f"rounded=1;arcSize=6;fillColor={mbg};strokeColor={mstroke};strokeWidth=1.2;html=1;align=center;verticalAlign=middle;",
                 mx, 472, 130, 55)

    # 4.2 Modality Alignment Stems
    stems_val = (
        "<b>Dimension-Aligned Modality Stems (d = 256)</b><br>"
        "<span style='font-size:10px; color:#451A03;'>"
        "Linear(D_m → 256) + LayerNorm + Conv1D(k=3, p=1) + GELU ──► <b>H_V, H_P, H_O, H_H ∈ ℝ^(16 × 256)</b>"
        "</span>"
    )
    add_node("box_stems", stems_val,
             "rounded=1;arcSize=6;fillColor=#FEF9C3;strokeColor=#D97706;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 545, 580, 50)

    for mid in ["m_vis", "m_pose", "m_obj", "m_hoi"]:
        add_edge(f"e_s_{mid}", mid, "box_stems", st_arr, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # 4.3 Bi-Directional Cross-Attention (Pose <-> Equipment)
    cross_val = (
        "<b>Bi-Directional Keypoint–Equipment Cross-Attention Module (h = 4 Heads, d_k = 64)</b><br>"
        "<span style='font-size:9.5px; color:#831843;'>"
        "Pose queries Equipment: <b>P_enh = LayerNorm(H_P + MHA(Q_p, K_o, V_o))</b><br>"
        "Equipment queries Pose: <b>O_enh = LayerNorm(H_O + MHA(Q_o, K_p, V_p))</b>"
        "</span>"
    )
    add_node("box_cross", cross_val,
             "rounded=1;arcSize=6;fillColor=#FDF2F8;strokeColor=#BE185D;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 615, 580, 62)
    add_edge("e_stems_cross", "box_stems", "box_cross", st_arr)

    # 4.4 Gated Multimodal Fusion Unit (GMFU)
    gmfu_val = (
        "<b>Gated Multimodal Fusion Unit (GMFU) — Dynamic Modality Gating</b><br>"
        "<span style='font-size:9.5px; color:#78350F;'>"
        "Joint Token: <b>U_t = [H_V ; P_enh ; O_enh ; H_H] ∈ ℝ^1024</b> ──► Gating MLP ──► <b>g_t = Softmax(s_t) ∈ ℝ^4</b><br>"
        "Dynamic Fused Frame Representation: <b>F_t = LayerNorm( ∑ g_t^(m) · X_t^(m) ) ∈ ℝ^256</b>"
        "</span>"
    )
    add_node("box_gmfu", gmfu_val,
             "rounded=1;arcSize=6;fillColor=#FEF3C7;strokeColor=#B45309;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 695, 580, 58)
    add_edge("e_cross_gmfu", "box_cross", "box_gmfu", st_arr)

    # 4.5 Spatio-Temporal Transformer Encoder
    trans_val = (
        "<b>Spatio-Temporal Transformer Encoder (3 Layers, Pre-LayerNorm, 8 Heads)</b><br>"
        "<span style='font-size:9.5px; color:#064E3B;'>"
        "Input Sequence: <b>Z_0 = [ z_CLS ; F_1 ; F_2 ; ... ; F_16 ] + E_pos ∈ ℝ^(17 × 256)</b><br>"
        "Multi-Head Self-Attention (h = 8, d_k = 32) + FFN (d_ff = 1024, GELU) with Pre-LN Residual Skips"
        "</span>"
    )
    add_node("box_trans", trans_val,
             "rounded=1;arcSize=6;fillColor=#ECFDF5;strokeColor=#059669;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 772, 580, 60)
    add_edge("e_gmfu_trans", "box_gmfu", "box_trans", st_arr)

    # 4.6 Multi-Scale 4-Way Temporal Pooling Head
    pool_val = (
        "<b>Multi-Scale 4-Way Temporal Pooling Head (D_pooled = 1024)</b><br>"
        "<span style='font-size:9.5px; color:#0E7490;'>"
        "• <b>z_CLS</b> (256-d): Global Context &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; "
        "• <b>z_Attn</b> (256-d): Saliency ∑ α_t z_t<br>"
        "• <b>z_Max</b> (256-d): Peak Motion &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; "
        "• <b>z_Avg</b> (256-d): Baseline Flow<br>"
        "<b>Concatenated Vector: Z_video = [ z_CLS ; z_Attn ; z_Max ; z_Avg ] ∈ ℝ^1024</b>"
        "</span>"
    )
    add_node("box_pool", pool_val,
             "rounded=1;arcSize=6;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 850, 580, 68)
    add_edge("e_trans_pool", "box_trans", "box_pool", st_arr)

    # 4.7 Neural Network Graph Schematic (Exact Reference 5 & 7 style!)
    add_node("box_nn_graph", "",
             "rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.5;html=1;shadow=1;",
             360, 935, 580, 220)

    add_node("lbl_nn_head", "<b>3-Layer Multi-Layer Perceptron (MLP) Classifier Architecture</b>",
             "text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#1E293B;fontStyle=1;", 370, 940, 560, 20)

    # Layer Labels
    add_node("lbl_l1", "<b>Input Layer</b><br>1024-d", "text;html=1;align=center;fontSize=9;fontColor=#DC3545;fontStyle=1;", 385, 965, 70, 25)
    add_node("lbl_l2", "<b>Hidden 1</b><br>512-d", "text;html=1;align=center;fontSize=9;fontColor=#E67E22;fontStyle=1;", 510, 965, 60, 25)
    add_node("lbl_l3", "<b>Hidden 2</b><br>256-d", "text;html=1;align=center;fontSize=9;fontColor=#27AE60;fontStyle=1;", 640, 965, 60, 25)
    add_node("lbl_l4", "<b>Output Layer</b><br>82 Classes", "text;html=1;align=center;fontSize=9;fontColor=#2980B9;fontStyle=1;", 770, 965, 75, 25)

    # Neurons (Input: 5, Hidden1: 4, Hidden2: 3, Output: 3)
    for i in range(5):
        add_node(f"node_in_{i}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#F8D7DA;strokeColor=#DC3545;strokeWidth=1.2;", 410, 995 + i*23, 18, 18)
    for j in range(4):
        add_node(f"node_h1_{j}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#FDEBD0;strokeColor=#E67E22;strokeWidth=1.2;", 530, 1005 + j*24, 18, 18)
    for k in range(3):
        add_node(f"node_h2_{k}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#D4EFDF;strokeColor=#27AE60;strokeWidth=1.2;", 660, 1017 + k*26, 18, 18)
    for m in range(3):
        add_node(f"node_out_{m}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#D6EAF8;strokeColor=#2980B9;strokeWidth=1.2;", 795, 1017 + m*26, 18, 18)

    # Synaptic connecting lines
    st_syn = "edgeStyle=none;rounded=0;html=1;strokeColor=#CBD5E1;strokeWidth=0.8;endArrow=none;"
    for i in range(5):
        for j in range(4):
            add_edge(f"syn_1_{i}_{j}", f"node_in_{i}", f"node_h1_{j}", st_syn)
    for j in range(4):
        for k in range(3):
            add_edge(f"syn_2_{j}_{k}", f"node_h1_{j}", f"node_h2_{k}", st_syn)
    for k in range(3):
        for m in range(3):
            add_edge(f"syn_3_{k}_{m}", f"node_h2_{k}", f"node_out_{m}", st_syn)

    add_node("lbl_nn_annot", "<font color='#64748B' style='font-size:9.5px;'>Linear(1024→512) + LN + GELU + Dropout(0.3) &nbsp;──►&nbsp; Linear(512→256) &nbsp;──►&nbsp; Linear(256→82 Classes)</font>",
             "text;html=1;align=center;verticalAlign=middle;", 370, 1122, 560, 20)

    add_edge("e_pool_nn", "box_pool", "box_nn_graph", st_arr, "Z_video (1024-d)")

    # 4.8 Training Objective inside Model: Focal Loss
    loss_val = (
        "<b>Training Objective: Focal Loss with Label Smoothing Regularization</b><br>"
        "<span style='font-size:11px; font-weight:bold; color:#B91C1C;'>"
        "L_Focal = - (1/B) ∑_(i=1)^B ∑_(c=1)^82 α_c · (1 - p_i,c)^γ · y_i,c^(smooth) · log(p_i,c)"
        "</span><br>"
        "<span style='font-size:9.5px; color:#475569;'>"
        "γ = 1.5 (Focusing Parameter) &nbsp;|&nbsp; α_c = 0.08 (Class Balancing) &nbsp;|&nbsp; ε = 0.08 (Label Smoothing)"
        "</span>"
    )
    add_node("box_loss", loss_val,
             "rounded=1;arcSize=6;fillColor=#FFF1F2;strokeColor=#E11D48;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 1175, 580, 68)

    add_edge("e_nn_loss", "box_nn_graph", "box_loss", st_arr, "Action Logits ŷ ∈ ℝ^82", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # Backprop upward arrow within Model (dedicated right margin: x=980)
    add_node("lbl_backprop", "<b>∇ L_Focal (Backpropagation Gradient Flow)</b>",
             "text;html=1;align=center;verticalAlign=middle;fontSize=9;fontColor=#E11D48;fontStyle=1;rotation=270;", 955, 780, 50, 240)
    add_edge("e_loss_backprop", "box_loss", "box_stems",
             "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#E11D48;strokeWidth=2;dashed=1;endArrow=block;endFill=1;fontSize=10;",
             "", exit_x=1.0, exit_y=0.5, entry_x=1.0, entry_y=0.5)

    add_edge("e_train_in", "card_train_data", "center_model", st_arr, "Mini-Batches (6,408 clips)", exit_x=0.0, exit_y=0.5, entry_x=1.0, entry_y=0.15)

    # -------------------------------------------------------------
    # 5. LEFT BOX: MODEL EVALUATION & GENERALIZATION (Ref 7 style)
    # -------------------------------------------------------------
    eval_val = (
        "<b style='font-size:13px; color:#15803D;'>Model Evaluation</b><br>"
        "<hr style='border:0; border-top:1px solid #BBF7D0; margin:6px 0;'>"
        "<span style='font-size:10px; color:#334155;'>Independent Generalization:</span><br>"
        "<div style='background:#DCFCE7; border:1px solid #16A34A; border-radius:5px; padding:6px; margin:6px 0; text-align:center;'>"
        "<b style='color:#14532D; font-size:13px;'>Top-1 Accuracy: 90.19%</b><br>"
        "<b style='color:#14532D; font-size:11.5px;'>Top-5 Accuracy: 99.02%</b>"
        "</div>"
        "<table width='100%' style='font-size:9.5px; color:#475569; text-align:left; line-height:1.6;'>"
        "<tr><td>• <b>Val Loss:</b> 0.4397</td><td>• <b>Epoch:</b> 23</td></tr>"
        "<tr><td>• <b>Macro F1:</b> 90.19%</td><td>• <b>Clips:</b> 2,743</td></tr>"
        "</table><br>"
        "<span style='font-size:9px; color:#64748B;'>82×82 Normalized Confusion Matrix verification</span>"
    )
    add_node("box_eval_left", eval_val,
             st_card_white + "fillColor=#F0FDF4;strokeColor=#16A34A;", 60, 680, 230, 270)

    add_edge("e_test_eval", "card_test_data", "box_eval_left", st_arr, "Test Input", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
    add_edge("e_model_eval_top", "center_model", "box_eval_left", st_arr_loop, "Validation Loop", exit_x=0.0, exit_y=0.35, entry_x=1.0, entry_y=0.25)
    add_edge("e_model_eval_bot", "box_eval_left", "center_model", st_arr, "Best Checkpoint", exit_x=1.0, exit_y=0.75, entry_x=0.0, entry_y=0.55)

    # -------------------------------------------------------------
    # 6. RIGHT BOX: HYPERPARAMETER OPTIMIZATION (Ref 7 style)
    # -------------------------------------------------------------
    hyper_val = (
        "<b style='font-size:13px; color:#92400E;'>Hyperparameter<br>Optimization</b><br>"
        "<hr style='border:0; border-top:1px solid #FDE68A; margin:6px 0;'>"
        "<table width='100%' style='font-size:10px; color:#334155; text-align:left; line-height:1.7;'>"
        "<tr><td>• <b>Optimizer:</b> AdamW</td></tr>"
        "<tr><td>• <b>Peak LR:</b> 4 × 10^-4</td></tr>"
        "<tr><td>• <b>Weight Decay:</b> 1e-4</td></tr>"
        "<tr><td>• <b>Scheduler:</b> CosineAnnealing</td></tr>"
        "<tr><td>• <b>Linear Warmup:</b> 5 Epochs</td></tr>"
        "<tr><td>• <b>Multimodal Mixup:</b> β = 0.4</td></tr>"
        "<tr><td>• <b>Kinematic Jitter:</b> σ = 0.008</td></tr>"
        "<tr><td>• <b>Frame Dropout:</b> p = 0.25</td></tr>"
        "</table>"
    )
    add_node("box_hyper_right", hyper_val,
             st_card_white + "fillColor=#FEF9E7;strokeColor=#D4AC0D;", 1050, 680, 230, 270)

    add_edge("e_hyper_loop1", "center_model", "box_hyper_right", st_arr_loop, "Adaptive Tuning", exit_x=1.0, exit_y=0.35, entry_x=0.0, entry_y=0.25)
    add_edge("e_hyper_loop2", "box_hyper_right", "center_model", st_arr, "Updated Weights", exit_x=0.0, exit_y=0.75, entry_x=1.0, entry_y=0.55)

    # -------------------------------------------------------------
    # 7. BOTTOM CONTAINER: UNSEEN DATA -> TRAINED MODEL -> PREDICTION (Exact Ref 7 Style!)
    # -------------------------------------------------------------
    st_bottom_box = "rounded=1;arcSize=6;fillColor=#F0FDFA;strokeColor=#0D9488;strokeWidth=2;html=1;container=1;collapsible=0;shadow=1;"
    add_node("box_bottom_inf", "", st_bottom_box, 60, 1530, 1220, 150)

    add_node("lbl_inf_head", "<b style='font-size:12px; color:#0F766E;'>Unseen Generalization &amp; Action Prediction (Offline Evaluation Inference)</b>",
             "text;html=1;align=center;verticalAlign=middle;", 60, 1535, 1220, 20)

    add_node("n_unseen", "<b>🎥 Unseen Sports Action</b><br><span style='font-size:10px; color:#134E4A;'><b>2,743 Disjoint Test Clips</b><br>(Subject-Independent)</span>",
             st_card_white + "fillColor=#CCFBF1;strokeColor=#14B8A6;", 100, 1570, 260, 75)

    add_node("n_trained_model", "<b>🧠 Trained CMAT-HOI Model</b><br><span style='font-size:10px; color:#065F46;'><b>Optimal Checkpoint @ Epoch 23</b><br>5.2M Frozen Parameters</span>",
             st_card_white + "fillColor=#D1FAE5;strokeColor=#10B981;", 510, 1570, 320, 75)

    add_node("n_pred_out", "<b>🏆 Action Prediction (82 Classes)</b><br><span style='font-size:10px; color:#1E3A8A;'><b>Top-1: 90.19% &nbsp;|&nbsp; Top-5: 99.02%</b><br>e.g. <i>TennisSwing 94.6%</i>, <i>HighJump 92.1%</i></span>",
             st_card_white + "fillColor=#DBEAFE;strokeColor=#3B82F6;", 980, 1570, 270, 75)

    add_edge("e_inf_1", "n_unseen", "n_trained_model", st_arr, "Test Clip Streams")
    add_edge("e_inf_2", "n_trained_model", "n_pred_out", st_arr, "Softmax Probabilities")

    # Connector from Training Model to Trained Model
    add_edge("e_model_to_trained", "center_model", "box_bottom_inf",
             "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#0D9488;strokeWidth=2.5;endArrow=block;endFill=1;fontSize=10;",
             "Optimal Model Weights (Epoch 23 Checkpoint)", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # Caption at bottom
    caption = "<b>Fig. 1.</b> System Model Architecture of the Proposed CMAT-HOI Framework for Sports Action Recognition (82 Classes)."
    add_node("cap", caption,
             "text;html=1;align=center;verticalAlign=middle;fontSize=12;fontColor=#475569;fontStyle=2;",
             20, 1720, 1320, 25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    print(f"Generated finetuned Draw.io XML at: {output_path}")


def render_finetuned_companion_graphic(output_png: Path, output_svg: Path):
    fig, ax = plt.subplots(figsize=(13.6, 18.5), dpi=300)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 18.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    def card(x, y, w, h, bg, stroke, lw=1.5, ls="-", radius=0.03):
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}", facecolor=bg, edgecolor=stroke, linewidth=lw, linestyle=ls)
        ax.add_patch(r)
        return r

    # Outer Frame Container (Reference 7 style)
    card(0.2, 0.2, 13.2, 18.0, "#F4F9FD", "#2980B9", lw=2, radius=0.02)

    # 1. TOP DASHED BOX: DATA INGESTION
    card(1.6, 16.6, 10.2, 1.15, "#FFFFFF", "#64748B", lw=1.5, ls="--")
    
    card(1.9, 16.75, 2.2, 0.8, "#EFF6FF", "#3B82F6")
    ax.text(3.0, 17.25, "Video Acquisition", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(3.0, 16.95, "Sports Action Capture", ha="center", fontsize=7.5, color="#475569")

    card(5.6, 16.75, 2.4, 0.8, "#E0F2FE", "#0284C7")
    ax.text(6.8, 17.25, "UCF-101 Benchmark", ha="center", fontsize=9.5, fontweight="bold", color="#0369A1")
    ax.text(6.8, 16.95, "Action Dataset Repository", ha="center", fontsize=7.5, color="#475569")

    card(9.3, 16.75, 2.2, 0.8, "#DBEAFE", "#2563EB")
    ax.text(10.4, 17.25, "Curated Dataset", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(10.4, 16.95, "9,151 Videos • 82 Classes", ha="center", fontsize=8, fontweight="bold", color="#1D4ED8")

    ax.annotate("", xy=(5.6, 17.15), xytext=(4.1, 17.15), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))
    ax.annotate("", xy=(9.3, 17.15), xytext=(8.0, 17.15), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))

    # Orthogonal arrow from Curated Dataset to Funnel Top
    ax.plot([10.4, 10.4, 6.7], [16.75, 16.15, 16.15], color="#1E293B", lw=1.5)
    ax.annotate("", xy=(6.7, 15.95), xytext=(6.7, 16.15), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))

    # 2. INVERTED PREPROCESSING FUNNEL (Real Continuous Funnel Trapezoid!)
    # Purple Side Arrow Badge (Pointing to the left side of funnel)
    p_arrow = patches.Polygon([
        [0.85, 15.0], [2.35, 15.0], [2.85, 14.65], [2.35, 14.3], [0.85, 14.3]
    ], closed=True, facecolor="#E8DAEF", edgecolor="#8E44AD", linewidth=1.5)
    ax.add_patch(p_arrow)
    ax.text(1.70, 14.75, "Data Preprocessing", ha="center", fontsize=7.8, fontweight="bold", color="#4A235A")
    ax.text(1.70, 14.48, "& Filtering", ha="center", fontsize=7.8, fontweight="bold", color="#4A235A")

    # Funnel Geometry: Trapezoid narrowing from top (w=7.6, y=15.95) to bottom (w=3.4, y=13.95)
    # Center = 6.7
    # 5 bands: each height = 0.40
    funnel_layers = [
        # (name, bg, edge, text_color, y_top, y_bot, w_top, w_bot)
        ("1. Temporal Uniform Sampling (T = 16 equidistant frames, 224 × 224 px)",
         "#D5F5E3", "#27AE60", "#145A32", 15.95, 15.55, 7.6, 6.76),
        ("2. Missing Value Imputation (Zero-fill missing joints & Kalman gear tracking)",
         "#FCF3CF", "#F39C12", "#7D6608", 15.55, 15.15, 6.76, 5.92),
        ("3. Kinematic Feature Extraction (Displacement velocities Δp_t = p_t - p_t-1)",
         "#D4E6F1", "#2980B9", "#1B4F72", 15.15, 14.75, 5.92, 5.08),
        ("4. Apparatus Quality Gating (Gear confidence s > 0.5 & aspect check)",
         "#D1F2EB", "#1ABC9C", "#0E6251", 14.75, 14.35, 5.08, 4.24),
        ("5. Feature Normalization (ImageNet standardization & FP32 tensors)",
         "#FADBD8", "#E74C3C", "#78281F", 14.35, 13.95, 4.24, 3.40)
    ]

    for text, bg, edge, tc, yt, yb, wt, wb in funnel_layers:
        poly_pts = [
            [6.7 - wt/2, yt],
            [6.7 + wt/2, yt],
            [6.7 + wb/2, yb],
            [6.7 - wb/2, yb]
        ]
        layer_poly = patches.Polygon(poly_pts, closed=True, facecolor=bg, edgecolor=edge, linewidth=1.2)
        ax.add_patch(layer_poly)
        ax.text(6.7, (yt + yb)/2, text, ha="center", va="center", fontsize=7.6, fontweight="bold", color=tc)

    # Dashed Funnel Top Opening (like Ref 7)
    ax.plot([6.7 - 3.8, 6.7 + 3.8], [15.95, 15.95], color="#64748B", lw=1.5, linestyle="--")

    # Big Downward Funnel Stem Arrow
    ax.annotate("", xy=(6.7, 13.3), xytext=(6.7, 13.95), arrowprops=dict(arrowstyle="->", lw=4, color="#34495E"))

    # 3. SPLIT: TEST DATA (LEFT) & TRAIN DATA (RIGHT)
    card(0.6, 12.2, 2.3, 0.85, "#FEF9E7", "#D4AC0D", lw=1.5)
    ax.text(1.75, 12.8, "Test Data (30%)", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(1.75, 12.55, "2,743 Video Clips", ha="center", fontsize=10, fontweight="bold", color="#15803D")
    ax.text(1.75, 12.32, "Strictly Disjoint Split", ha="center", fontsize=7.2, color="#475569")

    card(10.7, 12.2, 2.3, 0.85, "#FEF9E7", "#D4AC0D", lw=1.5)
    ax.text(11.85, 12.8, "Train Data (70%)", ha="center", fontsize=9.5, fontweight="bold", color="#1D4ED8")
    ax.text(11.85, 12.55, "6,408 Video Clips", ha="center", fontsize=10, fontweight="bold", color="#1E40AF")
    ax.text(11.85, 12.32, "Subject-Independent", ha="center", fontsize=7.2, color="#475569")

    # Clean non-crossing split arrows
    ax.plot([6.3, 1.75, 1.75], [13.45, 13.45, 13.05], color="#1E293B", lw=1.5)
    ax.annotate("", xy=(1.75, 13.05), xytext=(1.75, 13.15), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))
    ax.text(3.6, 13.60, "30% Disjoint Split", ha="center", fontsize=7.8, fontweight="bold", color="#1E40AF")

    ax.plot([7.1, 11.85, 11.85], [13.45, 13.45, 13.05], color="#1E293B", lw=1.5)
    ax.annotate("", xy=(11.85, 13.05), xytext=(11.85, 13.15), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))
    ax.text(9.8, 13.60, "70% Training Split", ha="center", fontsize=7.8, fontweight="bold", color="#1D4ED8")

    # -------------------------------------------------------------
    # 4. CENTER CARD: TRAINING MODEL (CMAT-HOI FRAMEWORK)
    # -------------------------------------------------------------
    # Center Model Card (x=3.2 to 10.2, w=7.0)
    card(3.2, 3.4, 7.0, 9.6, "#FEFDF8", "#D4AC0D", lw=2, radius=0.03)
    ax.text(6.6, 12.75, "Training Model: CMAT-HOI Framework", ha="center", fontsize=11.5, fontweight="bold", color="#78350F")

    # Multimodal Inputs (Inner cards span x=3.7 to 9.3, w=5.6, center at 6.5)
    ax.text(6.5, 12.45, "Multimodal Input Streams (T = 16 Equidistant Frames)", ha="center", fontsize=8.5, fontweight="bold", color="#78350F")
    sboxes = [
        ("Visual Appearance\nResNet-50\nX_V ∈ ℝ^(16 × 2048)", 3.7, "#DBEAFE", "#2563EB"),
        ("Kinematic Pose\nKeypoint R-CNN\nX_P ∈ ℝ^(16 × 85)", 5.15, "#D1FAE5", "#059669"),
        ("Sports Equipment\nFaster R-CNN\nX_O ∈ ℝ^(16 × 7)", 6.6, "#FEF3C7", "#D97706"),
        ("Geometric HOI\nKinematic Engine\nX_H ∈ ℝ^(16 × 16)", 8.05, "#FCE7F3", "#DB2777")
    ]
    for stext, sx, sbg, sstroke in sboxes:
        card(sx, 11.6, 1.25, 0.7, sbg, sstroke, lw=1.2)
        ax.text(sx + 0.625, 11.95, stext, ha="center", va="center", fontsize=6.6, fontweight="bold", color="#1E293B")

    # Modality Stems
    card(3.7, 10.8, 5.6, 0.55, "#FEF9C3", "#D97706", lw=1.2)
    ax.text(6.5, 11.15, "Dimension-Aligned Modality Stems (d = 256)", ha="center", fontsize=8.5, fontweight="bold", color="#92400E")
    ax.text(6.5, 10.95, "Linear(D_m → 256) + LayerNorm + Conv1D(k=3, p=1) + GELU ──► H_V, H_P, H_O, H_H ∈ ℝ^(16 × 256)", ha="center", fontsize=7.1, color="#451A03")

    for sx in [4.325, 5.775, 7.225, 8.675]:
        ax.annotate("", xy=(sx, 11.35), xytext=(sx, 11.6), arrowprops=dict(arrowstyle="->", lw=1, color="#D97706"))

    # Bi-Directional Cross-Attention (Neatly formatted on two lines)
    card(3.7, 9.9, 5.6, 0.68, "#FDF2F8", "#BE185D", lw=1.4)
    ax.text(6.5, 10.38, "Bi-Directional Keypoint–Equipment Cross-Attention Module (h = 4 Heads, d_k = 64)", ha="center", fontsize=8.5, fontweight="bold", color="#831843")
    ax.text(6.5, 10.16, "Pose queries Equipment: P_enh = LayerNorm(H_P + MHA(Q_p, K_o, V_o))", ha="center", fontsize=7.3, color="#4C0519")
    ax.text(6.5, 9.98, "Equipment queries Pose: O_enh = LayerNorm(H_O + MHA(Q_o, K_p, V_p))", ha="center", fontsize=7.3, color="#4C0519")
    ax.annotate("", xy=(6.5, 10.58), xytext=(6.5, 10.8), arrowprops=dict(arrowstyle="->", lw=1.2, color="#BE185D"))

    # Gated Multimodal Fusion Unit (GMFU) (Neatly formatted on two lines)
    card(3.7, 9.05, 5.6, 0.65, "#FEF3C7", "#B45309", lw=1.4)
    ax.text(6.5, 9.50, "Gated Multimodal Fusion Unit (GMFU) — Dynamic Modality Gating", ha="center", fontsize=8.5, fontweight="bold", color="#78350F")
    ax.text(6.5, 9.30, "Joint Token: U_t = [H_V ; P_enh ; O_enh ; H_H] ∈ ℝ^1024 ──► Gating MLP ──► g_t = Softmax(s_t) ∈ ℝ^4", ha="center", fontsize=7.1, color="#451A03")
    ax.text(6.5, 9.13, "Dynamic Fused Frame Representation: F_t = LayerNorm( ∑ g_t^(m) · X_t^(m) ) ∈ ℝ^256", ha="center", fontsize=7.1, color="#451A03")
    ax.annotate("", xy=(6.5, 9.7), xytext=(6.5, 9.9), arrowprops=dict(arrowstyle="->", lw=1.2, color="#B45309"))

    # Spatio-Temporal Transformer Encoder
    card(3.7, 8.15, 5.6, 0.7, "#ECFDF5", "#059669", lw=1.4)
    ax.text(6.5, 8.62, "Spatio-Temporal Transformer Encoder (3 Layers, Pre-LayerNorm, 8 Heads)", ha="center", fontsize=8.5, fontweight="bold", color="#064E3B")
    ax.text(6.5, 8.40, "Input Sequence: Z_0 = [ z_CLS ; F_1 ; F_2 ; ... ; F_16 ] + E_pos ∈ ℝ^(17 × 256)", ha="center", fontsize=7.5, color="#047857")
    ax.text(6.5, 8.24, "Multi-Head Self-Attention (h = 8, d_k = 32) + FFN (d_ff = 1024, GELU) with Pre-LN Residual Skips", ha="center", fontsize=7.0, color="#022C22")
    ax.annotate("", xy=(6.5, 8.85), xytext=(6.5, 9.05), arrowprops=dict(arrowstyle="->", lw=1.2, color="#059669"))

    # Multi-Scale 4-Way Temporal Pooling Head
    card(3.7, 7.25, 5.6, 0.72, "#CFFAFE", "#0891B2", lw=1.4)
    ax.text(6.5, 7.78, "Multi-Scale 4-Way Temporal Pooling Head (D_pooled = 1024)", ha="center", fontsize=8.5, fontweight="bold", color="#0E7490")
    ax.text(6.5, 7.58, "• z_CLS (256-d): Global Context     • z_Attn (256-d): Saliency ∑ α_t z_t", ha="center", fontsize=7.3, color="#164E63")
    ax.text(6.5, 7.42, "• z_Max (256-d): Peak Motion        • z_Avg (256-d): Baseline Flow", ha="center", fontsize=7.3, color="#164E63")
    ax.text(6.5, 7.28, "Concatenated: Z_video = [ z_CLS ; z_Attn ; z_Max ; z_Avg ] ∈ ℝ^1024", ha="center", fontsize=7.5, fontweight="bold", color="#0E7490")
    ax.annotate("", xy=(6.5, 7.97), xytext=(6.5, 8.15), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0891B2"))

    # Neural Network Graph Schematic (Exact Reference 5 & 7 style!)
    card(3.7, 4.6, 5.6, 2.45, "#FFFFFF", "#CBD5E1", lw=1.5, radius=0.02)
    ax.text(6.5, 6.85, "3-Layer Multi-Layer Perceptron (MLP) Classifier Architecture", ha="center", fontsize=9, fontweight="bold", color="#1E293B")

    ax.text(4.2, 6.58, "Input Layer\n(1024-d)", ha="center", fontsize=7.5, fontweight="bold", color="#DC3545")
    ax.text(5.4, 6.58, "Hidden 1\n(512-d)", ha="center", fontsize=7.5, fontweight="bold", color="#E67E22")
    ax.text(6.6, 6.58, "Hidden 2\n(256-d)", ha="center", fontsize=7.5, fontweight="bold", color="#27AE60")
    ax.text(7.9, 6.58, "Output Layer\n(82 Classes)", ha="center", fontsize=7.5, fontweight="bold", color="#2980B9")

    in_nodes = [(4.2, 6.22 - i*0.32) for i in range(5)]
    h1_nodes = [(5.4, 6.07 - j*0.35) for j in range(4)]
    h2_nodes = [(6.6, 5.92 - k*0.38) for k in range(3)]
    out_nodes = [(7.9, 5.92 - m*0.38) for m in range(3)]

    for p1 in in_nodes:
        for p2 in h1_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#CBD5E1", lw=0.6, zorder=1)
    for p1 in h1_nodes:
        for p2 in h2_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#CBD5E1", lw=0.6, zorder=1)
    for p1 in h2_nodes:
        for p2 in out_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#CBD5E1", lw=0.6, zorder=1)

    for x, y in in_nodes:
        ax.scatter(x, y, s=140, color="#F8D7DA", edgecolors="#DC3545", linewidths=1.5, zorder=2)
    for x, y in h1_nodes:
        ax.scatter(x, y, s=140, color="#FDEBD0", edgecolors="#E67E22", linewidths=1.5, zorder=2)
    for x, y in h2_nodes:
        ax.scatter(x, y, s=140, color="#D4EFDF", edgecolors="#27AE60", linewidths=1.5, zorder=2)
    for x, y in out_nodes:
        ax.scatter(x, y, s=140, color="#D6EAF8", edgecolors="#2980B9", linewidths=1.5, zorder=2)

    ax.text(6.5, 4.75, "Linear(1024→512) + LN + GELU + Dropout(0.3)  ──►  Linear(512→256)  ──►  Linear(256→82)",
            ha="center", fontsize=7.2, color="#64748B")

    ax.annotate("", xy=(6.5, 7.05), xytext=(6.5, 7.25), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0891B2"))

    # Training Objective: Focal Loss with Label Smoothing Regularization
    card(3.7, 3.6, 5.6, 0.82, "#FFF1F2", "#E11D48", lw=1.5)
    ax.text(6.5, 4.25, "Training Objective: Focal Loss with Label Smoothing Regularization", ha="center", fontsize=8.5, fontweight="bold", color="#9F1239")
    ax.text(6.5, 4.02, "L_Focal = - (1/B) ∑_(i=1)^B ∑_(c=1)^82 α_c · (1 - p_i,c)^γ · y_i,c^(smooth) · log(p_i,c)", ha="center", fontsize=7.8, fontweight="bold", color="#B91C1C")
    ax.text(6.5, 3.78, "γ = 1.5 (Focusing Parameter)  •  α_c = 0.08 (Class Balancing)  •  ε = 0.08 (Label Smoothing)", ha="center", fontsize=7.2, color="#475569")

    ax.annotate("", xy=(6.5, 4.42), xytext=(6.5, 4.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#E11D48"))
    ax.text(6.5, 4.51, "Action Logits ŷ ∈ ℝ^82", ha="center", fontsize=7.0, fontweight="bold", color="#9F1239", backgroundcolor="#FFFFFF")

    # Upward Backpropagation Gradient Flow (Completely in dedicated right margin: x=9.7, free of collisions!)
    ax.plot([9.7, 9.7], [4.0, 11.0], color="#E11D48", lw=1.8, linestyle="--")
    ax.annotate("", xy=(9.7, 11.0), xytext=(9.7, 10.9), arrowprops=dict(arrowstyle="->", lw=1.8, color="#E11D48"))
    ax.text(9.7, 7.5, "∇ L_Focal (Backpropagation Gradient Flow)", ha="center", va="center", rotation=270,
            fontsize=7.8, fontweight="bold", color="#E11D48", backgroundcolor="#FEFDF8")

    # Train Data Input arrow into Model
    ax.plot([11.85, 11.85, 10.2], [12.2, 11.95, 11.95], color="#1E293B", lw=1.5)
    ax.annotate("", xy=(10.2, 11.95), xytext=(10.4, 11.95), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))
    ax.text(11.0, 12.08, "Train Batches (6,408)", ha="center", fontsize=7.2, color="#1E293B")

    # -------------------------------------------------------------
    # 5. LEFT BOX: MODEL EVALUATION
    # -------------------------------------------------------------
    card(0.6, 6.8, 2.3, 3.8, "#F0FDF4", "#16A34A", lw=1.5)
    ax.text(1.75, 10.3, "Model Evaluation", ha="center", fontsize=10.5, fontweight="bold", color="#15803D")
    ax.plot([0.8, 2.7], [10.1, 10.1], color="#BBF7D0", lw=1)
    
    card(0.75, 9.0, 2.0, 0.9, "#DCFCE7", "#16A34A", lw=1.2)
    ax.text(1.75, 9.6, "Top-1: 90.19%", ha="center", fontsize=10.5, fontweight="bold", color="#14532D")
    ax.text(1.75, 9.25, "Top-5: 99.02%", ha="center", fontsize=9.5, fontweight="bold", color="#14532D")

    eval_text = [
        "• Val Loss: 0.4397",
        "• Peak Epoch: 23",
        "• Macro F1: 90.19%",
        "• 2,743 Disjoint Test Clips",
        "• 82×82 Confusion Matrix"
    ]
    for i, line in enumerate(eval_text):
        ax.text(0.8, 8.6 - i * 0.32, line, ha="left", fontsize=7.8, color="#1E293B")

    ax.annotate("", xy=(1.75, 10.6), xytext=(1.75, 12.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))
    ax.text(1.75, 11.4, "Test Input", ha="center", fontsize=7.2, color="#1E293B", backgroundcolor="#FFFFFF")

    # Clean Bidirectional Evaluation Loop (with label above)
    ax.annotate("", xy=(3.2, 8.8), xytext=(2.9, 8.8), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#1E293B"))
    ax.text(3.05, 9.05, "Evaluation", ha="center", fontsize=6.8, color="#1E293B")

    # -------------------------------------------------------------
    # 6. RIGHT BOX: HYPERPARAMETER OPTIMIZATION
    # -------------------------------------------------------------
    card(10.7, 6.8, 2.3, 3.8, "#FEF9E7", "#D4AC0D", lw=1.5)
    ax.text(11.85, 10.3, "Hyperparameter", ha="center", fontsize=10.5, fontweight="bold", color="#92400E")
    ax.text(11.85, 10.0, "Optimization", ha="center", fontsize=10.5, fontweight="bold", color="#92400E")
    ax.plot([10.9, 12.8], [9.85, 9.85], color="#FDE68A", lw=1)

    hyper_text = [
        "• Optimizer: AdamW",
        "• Peak LR: 4 × 10^-4",
        "• Weight Decay: 1e-4",
        "• CosineAnnealingLR",
        "• Linear Warmup (5 ep)",
        "• Multimodal Mixup: 0.4",
        "• Kinematic Jitter: σ = 0.008",
        "• Frame Dropout: p = 0.25"
    ]
    for i, line in enumerate(hyper_text):
        ax.text(10.9, 9.55 - i * 0.32, line, ha="left", fontsize=7.8, color="#1E293B")

    # Clean Bidirectional Optimization Loop (with label above)
    ax.annotate("", xy=(10.7, 8.8), xytext=(10.2, 8.8), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#1E293B"))
    ax.text(10.45, 9.05, "Tuning", ha="center", fontsize=6.8, color="#1E293B")

    # -------------------------------------------------------------
    # 7. BOTTOM SHADED CONTAINER: UNSEEN DATA -> TRAINED MODEL -> PREDICTION (Ref 7 exact style)
    # -------------------------------------------------------------
    card(0.6, 1.0, 12.4, 1.8, "#F0FDFA", "#0D9488", lw=2, radius=0.03)
    ax.text(6.8, 2.58, "Unseen Generalization & Action Prediction (Offline Evaluation Inference)",
            ha="center", fontsize=10.5, fontweight="bold", color="#0F766E")

    card(1.0, 1.25, 2.7, 0.95, "#CCFBF1", "#14B8A6", lw=1.3)
    ax.text(2.35, 1.85, "Unseen Sports Action", ha="center", fontsize=9.5, fontweight="bold", color="#115E59")
    ax.text(2.35, 1.62, "2,743 Disjoint Test Clips", ha="center", fontsize=8.5, fontweight="bold", color="#0D9488")
    ax.text(2.35, 1.42, "Subject-Independent Split", ha="center", fontsize=7.2, color="#475569")

    card(5.0, 1.25, 3.6, 0.95, "#D1FAE5", "#10B981", lw=1.3)
    ax.text(6.8, 1.85, "Trained CMAT-HOI Model", ha="center", fontsize=9.5, fontweight="bold", color="#065F46")
    ax.text(6.8, 1.62, "Optimal Checkpoint @ Epoch 23 (Val Loss: 0.4397)", ha="center", fontsize=8.0, fontweight="bold", color="#047857")
    ax.text(6.8, 1.42, "Frozen Parameters (Θ = 5.2M) • Multi-Scale Pooling", ha="center", fontsize=7.2, color="#475569")

    card(9.9, 1.25, 2.7, 0.95, "#DBEAFE", "#3B82F6", lw=1.3)
    ax.text(11.25, 1.85, "Action Prediction (82 Classes)", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(11.25, 1.62, "Top-1: 90.19%  |  Top-5: 99.02%", ha="center", fontsize=8.2, fontweight="bold", color="#1D4ED8")
    ax.text(11.25, 1.42, "TennisSwing, HighJump, Dunk...", ha="center", fontsize=7.2, color="#475569")

    ax.annotate("", xy=(5.0, 1.72), xytext=(3.7, 1.72), arrowprops=dict(arrowstyle="->", lw=1.8, color="#0D9488"))
    ax.text(4.35, 1.85, "Test Clips", ha="center", fontsize=7.2, color="#0D9488")

    ax.annotate("", xy=(9.9, 1.72), xytext=(8.6, 1.72), arrowprops=dict(arrowstyle="->", lw=1.8, color="#0D9488"))
    ax.text(9.25, 1.85, "Logits / Softmax", ha="center", fontsize=7.2, color="#0D9488")

    # Downward arrow from Training Model to Bottom Box
    ax.annotate("", xy=(6.6, 2.8), xytext=(6.6, 3.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#0D9488"))
    ax.text(6.6, 3.1, "Optimal Trained Weights Checkpoint (Epoch 23)", ha="center", fontsize=7.5, fontweight="bold", color="#0F766E", backgroundcolor="#FFFFFF")

    # Caption at bottom
    ax.text(6.8, 0.5, "Fig. 1. System Model Architecture of the Proposed CMAT-HOI Framework for Sports Action Recognition (82 Classes).",
            ha="center", va="center", fontsize=9.5, fontfamily="serif", fontstyle="italic", color="#334155")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()
    print(f"Generated finetuned companion PNG at: {output_png}")
    print(f"Generated finetuned companion SVG at: {output_svg}")


if __name__ == "__main__":
    out_drawio_paper = Path("Paper writing/CMAT_HOI_Methodology_Architecture.drawio")
    out_drawio_m7 = Path("diagrams/M7/CMAT_HOI_Methodology_Architecture.drawio")
    build_finetuned_drawio_xml(out_drawio_paper)
    build_finetuned_drawio_xml(out_drawio_m7)

    out_png_paper = Path("Paper writing/figures/methodology_architecture_diagram.png")
    out_svg_paper = Path("Paper writing/figures/methodology_architecture_diagram.svg")
    out_png_m7 = Path("diagrams/M7/methodology_architecture_diagram.png")
    out_svg_m7 = Path("diagrams/M7/methodology_architecture_diagram.svg")

    render_finetuned_companion_graphic(out_png_paper, out_svg_paper)
    render_finetuned_companion_graphic(out_png_m7, out_svg_m7)
