#!/usr/bin/env python3
"""
Generate the Reference-7 style System Architecture Diagram for CMAT-HOI.
Directly models Reference 7 (media_1795360450914_42db0c78.png):
  - Top: Data Collection & Dataset Ingestion card
  - Funnel: Inverted 5-stage Data Preprocessing Funnel (Trapezoid / multi-band)
  - Split: Left to Test Data (30%), Right to Train Data (70%)
  - Center: Training Model card featuring:
      * 4 Multimodal Input Streams
      * Modality Stems (d=256)
      * Bi-Directional Cross-Attention (Pose <-> Equipment)
      * Gated Multimodal Fusion Unit (GMFU)
      * Spatio-Temporal Transformer Encoder
      * Multi-Scale 4-Way Temporal Pooling
      * Visual Neural Network Graph (drawn circular neurons & synaptic weights like Ref 5 & 7)
  - Left: Model Evaluation card (90.19% Top-1, 99.02% Top-5) with bidirectional loop
  - Right: Hyperparameter Optimization card (AdamW, Cosine, Mixup) with bidirectional loop
  - Bottom: Focal Loss with Label Smoothing + 82-Class Action Output
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def build_drawio_xml(output_path: Path):
    root = ET.Element("mxfile", host="app.diagrams.net", version="24.0.0", type="device")
    diagram = ET.SubElement(root, "diagram", id="CMAT_HOI_Ref7_Architecture", name="System Architecture")
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
        pageWidth="1320",
        pageHeight="1680",
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

    # Styles matching Reference 7
    st_outer_container = "rounded=1;arcSize=4;fillColor=#EBF5FB;strokeColor=#2980B9;strokeWidth=2;html=1;container=1;collapsible=0;"
    st_top_dashed = "rounded=1;arcSize=6;fillColor=#F8FAFC;strokeColor=#475569;strokeWidth=1.5;dashed=1;html=1;container=1;collapsible=0;"
    st_center_model = "rounded=1;arcSize=6;fillColor=#FEF9E7;strokeColor=#B7950B;strokeWidth=2;html=1;container=1;collapsible=0;shadow=1;"
    st_card_white = "rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;shadow=1;"
    st_arr = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1E293B;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;"
    st_arr_loop = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1E293B;strokeWidth=2;startArrow=classic;endArrow=classic;startFill=1;endFill=1;fontSize=10;"

    # Outer Container Frame (Reference 7 style)
    add_node("frame", "", st_outer_container, 20, 20, 1280, 1620)

    # -------------------------------------------------------------
    # TOP DASHED BOX: DATA COLLECTION & DATASET REPOSITORY
    # -------------------------------------------------------------
    add_node("box_top_coll", "", st_top_dashed, 160, 50, 1000, 100)

    # Left: Data Collection Icon & Card
    add_node("n_coll", "<b>🏃 Data Collection</b><br><span style='font-size:10px; color:#475569;'>Sports Video Ingestion</span>",
             st_card_white + "fillColor=#EFF6FF;strokeColor=#3B82F6;", 200, 70, 190, 60)

    # Center: Kaggle / UCF-101 Repository
    add_node("n_repo", "<b>📁 UCF-101 Repository</b><br><span style='font-size:10px; color:#475569;'>Sports Action Benchmark</span>",
             st_card_white + "fillColor=#E0F2FE;strokeColor=#0284C7;", 540, 70, 240, 60)

    # Right: Dataset Specs
    add_node("n_dataset", "<b>📊 Data Set</b><br><span style='font-size:10px; color:#1E40AF;'><b>9,151 Clips &nbsp;•&nbsp; 82 Classes</b></span>",
             st_card_white + "fillColor=#DBEAFE;strokeColor=#2563EB;", 920, 70, 200, 60)

    add_edge("e_c1", "n_coll", "n_repo", st_arr)
    add_edge("e_c2", "n_repo", "n_dataset", st_arr)

    # -------------------------------------------------------------
    # PREPROCESSING FUNNEL (Reference 7 Inverted Funnel)
    # -------------------------------------------------------------
    # Side badge arrow pointing into funnel
    add_node("funnel_badge", "<b>Data Filtering &amp;<br>Conditioning</b>",
             "shape=singleArrow;direction=east;whiteSpace=wrap;html=1;fillColor=#E8DAEF;strokeColor=#8E44AD;strokeWidth=1.5;fontStyle=1;fontSize=11;fontColor=#4A235A;align=center;",
             100, 220, 170, 60)

    # Funnel Bands (Narrowing downwards)
    # Band 1 (Top/Widest): Data Cleaning & Frame Sampling
    add_node("funnel_b1", "<b>Data Cleaning &amp; Temporal Sampling:</b> Extract T = 16 equidistant frames (224 × 224 px)",
             "rounded=1;arcSize=4;fillColor=#D5F5E3;strokeColor=#27AE60;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#145A32;",
             310, 175, 700, 34)

    # Band 2: Missing Keypoint & Equipment Imputation
    add_node("funnel_b2", "<b>Missing Value Handling:</b> Zero-filling missing joints &amp; Kalman smoothing on apparatus bboxes",
             "rounded=1;arcSize=4;fillColor=#FCF3CF;strokeColor=#F39C12;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#7D6608;",
             355, 214, 610, 34)

    # Band 3: Kinematic Velocity Extraction
    add_node("funnel_b3", "<b>Kinematic Extraction:</b> Compute 1st-order temporal displacement velocities (Δx, Δy)",
             "rounded=1;arcSize=4;fillColor=#D4E6F1;strokeColor=#2980B9;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#1B4F72;",
             405, 253, 510, 34)

    # Band 4: Outlier & Aspect Ratio Filtering
    add_node("funnel_b4", "<b>Outlier Handling:</b> Keypoint confidence thresholding (c &gt; 0.3) &amp; aspect ratio validation",
             "rounded=1;arcSize=4;fillColor=#D1F2EB;strokeColor=#1ABC9C;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#0E6251;",
             455, 292, 410, 34)

    # Band 5 (Bottom/Narrowest): Standardization & Packaging
    add_node("funnel_b5", "<b>Standardization:</b> Zero-center normalization &amp; FP32 tensor packaging",
             "rounded=1;arcSize=4;fillColor=#FADBD8;strokeColor=#E74C3C;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#78281F;",
             505, 331, 310, 34)

    # Big Downward Funnel Stem Arrow
    add_node("funnel_arrow", "",
             "shape=flexArrow;direction=south;fillColor=#5D6D7E;strokeColor=#34495E;strokeWidth=1.5;endArrow=classic;",
             635, 370, 50, 45)

    # Connect Dataset into Funnel
    add_edge("e_ds_funnel", "n_dataset", "funnel_b1", st_arr, "", exit_x=0.5, exit_y=1.0, entry_x=0.85, entry_y=0.0)

    # -------------------------------------------------------------
    # SPLIT: TEST DATA (LEFT) & TRAIN DATA (RIGHT) (Reference 7 style)
    # -------------------------------------------------------------
    # Left Split Card: Test Data
    add_node("card_test_data", "<b>🎯 Test Data (30%)</b><br><span style='font-size:12px; font-weight:bold; color:#1E40AF;'>2,743 Video Clips</span><br><font style='font-size:9.5px; color:#475569;'>Strictly Disjoint Unseen Subjects</font>",
             st_card_white + "fillColor=#FEF9E7;strokeColor=#D4AC0D;", 70, 435, 240, 75)

    # Right Split Card: Train Data
    add_node("card_train_data", "<b>🏋️ Train Data (70%)</b><br><span style='font-size:12px; font-weight:bold; color:#1D4ED8;'>6,408 Video Clips</span><br><font style='font-size:9.5px; color:#475569;'>Subject-Independent Training</font>",
             st_card_white + "fillColor=#FEF9E7;strokeColor=#D4AC0D;", 1010, 435, 240, 75)

    # Branch arrows from funnel bottom arrow
    add_edge("e_sp_left", "funnel_arrow", "card_test_data", st_arr, "30% Disjoint Split", exit_x=0.2, exit_y=1.0, entry_x=0.5, entry_y=0.0)
    add_edge("e_sp_right", "funnel_arrow", "card_train_data", st_arr, "70% Training Split", exit_x=0.8, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # -------------------------------------------------------------
    # CENTER CARD: TRAINING MODEL (CMAT-HOI FRAMEWORK)
    # -------------------------------------------------------------
    add_node("center_model", "<b style='font-size:14px; color:#78350F;'>Training Model: CMAT-HOI Framework</b>",
             st_center_model, 340, 435, 640, 940)

    # --- 1. Multimodal Feature Input Streams ---
    add_node("lbl_streams", "<b>Multimodal Input Streams (T = 16 Frames)</b>",
             "text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#78350F;fontStyle=1;", 360, 470, 600, 20)

    stream_boxes = [
        ("m_vis", "Visual Stream<br><b>ResNet-50</b><br><font color='#1D4ED8'>[16, 2048]</font>", "#DBEAFE", "#2563EB", 360),
        ("m_pose", "Skeletal Pose<br><b>Keypoint R-CNN</b><br><font color='#047857'>[16, 85]</font>", "#D1FAE5", "#059669", 515),
        ("m_obj", "Equipment<br><b>Faster R-CNN</b><br><font color='#B45309'>[16, 7]</font>", "#FEF3C7", "#D97706", 670),
        ("m_hoi", "HOI Kinematics<br><b>Engine</b><br><font color='#BE185D'>[16, 16]</font>", "#FCE7F3", "#DB2777", 825)
    ]
    for mid, mlabel, mbg, mstroke, mx in stream_boxes:
        add_node(mid, f"<span style='font-size:9.5px;'>{mlabel}</span>",
                 f"rounded=1;arcSize=6;fillColor={mbg};strokeColor={mstroke};strokeWidth=1.2;html=1;align=center;verticalAlign=middle;",
                 mx, 495, 140, 55)

    # --- 2. Unified Modality Stems ---
    add_node("box_stems", "<b>Modality Alignment Stems (Conv1D + LayerNorm)</b><br><span style='font-size:10px; color:#451A03;'>Unified Latent Dimension: E_vis, E_pose, E_obj, E_hoi ∈ ℝ^(16 × 256)</span>",
             "rounded=1;arcSize=6;fillColor=#FEF9C3;strokeColor=#D97706;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 570, 600, 48)

    # Connect streams to stems
    for mid in ["m_vis", "m_pose", "m_obj", "m_hoi"]:
        add_edge(f"e_s_{mid}", mid, "box_stems", st_arr, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # --- 3. Bi-Directional Cross-Attention ---
    cross_val = (
        "<b>Bi-Directional Keypoint–Equipment Cross-Attention Module</b><br>"
        "<span style='font-size:9.5px; color:#831843;'>Multi-Head Attention (h = 4 Heads, d_k = 64)<br>"
        "Pose queries Equipment (P_enh) &nbsp;◄──►&nbsp; Equipment queries Pose (O_enh)</span>"
    )
    add_node("box_cross", cross_val,
             "rounded=1;arcSize=6;fillColor=#FDF2F8;strokeColor=#BE185D;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 640, 600, 55)
    add_edge("e_stems_cross", "box_stems", "box_cross", st_arr)

    # --- 4. Gated Multimodal Fusion Unit (GMFU) ---
    gmfu_val = (
        "<b>Gated Multimodal Fusion Unit (GMFU)</b><br>"
        "<span style='font-size:9.5px; color:#78350F;'>Softmax Modality Gating Vector: g_t ∈ ℝ^4 ──► Fused Token: F_t = ∑ g_t,m · E_m,t ∈ ℝ^256</span>"
    )
    add_node("box_gmfu", gmfu_val,
             "rounded=1;arcSize=6;fillColor=#FEF3C7;strokeColor=#B45309;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 715, 600, 50)
    add_edge("e_cross_gmfu", "box_cross", "box_gmfu", st_arr)

    # --- 5. Spatio-Temporal Transformer Encoder ---
    trans_val = (
        "<b>Spatio-Temporal Transformer Encoder (3 Layers, 8 Attention Heads)</b><br>"
        "<span style='font-size:9.5px; color:#064E3B;'>Input: Z_0 = [ z_CLS ; F_1 ; F_2 ; ... ; F_16 ] + E_pos ∈ ℝ^(17 × 256)<br>"
        "Multi-Head Self-Attention + Pre-LayerNorm + Feed-Forward (d_ff = 1024) + Residual Skips</span>"
    )
    add_node("box_trans", trans_val,
             "rounded=1;arcSize=6;fillColor=#ECFDF5;strokeColor=#059669;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 785, 600, 60)
    add_edge("e_gmfu_trans", "box_gmfu", "box_trans", st_arr)

    # --- 6. Multi-Scale 4-Way Temporal Pooling Head ---
    pool_val = (
        "<b>Multi-Scale 4-Way Temporal Pooling Head</b><br>"
        "<span style='font-size:9.5px; color:#0E7490;'>Concatenation: Z_pool = [ z_CLS (256) ; z_Attn (256) ; z_Max (256) ; z_Avg (256) ] ∈ ℝ^1024</span>"
    )
    add_node("box_pool", pool_val,
             "rounded=1;arcSize=6;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             360, 865, 600, 50)
    add_edge("e_trans_pool", "box_trans", "box_pool", st_arr)

    # --- 7. Neural Network Graph Schematic (Exact Reference 5 & 7 style!) ---
    # Container for Neural Net Graph
    add_node("box_nn_graph", "",
             "rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.5;html=1;shadow=1;",
             360, 935, 600, 240)

    add_node("lbl_nn_head", "<b>3-Layer Multi-Layer Perceptron (MLP) Classifier Architecture</b>",
             "text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#1E293B;fontStyle=1;", 370, 940, 580, 20)

    # Draw Layers: Input (5 nodes), Hidden 1 (4 nodes), Hidden 2 (3 nodes), Output (3 nodes)
    # Layer 1: Input (x=410)
    add_node("lbl_l1", "<b>Input</b><br>1024-d", "text;html=1;align=center;fontSize=9;fontColor=#475569;", 385, 965, 50, 25)
    for i in range(5):
        add_node(f"node_in_{i}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#F8D7DA;strokeColor=#DC3545;strokeWidth=1.2;", 400, 995 + i*26, 20, 20)

    # Layer 2: Hidden 1 (x=530)
    add_node("lbl_l2", "<b>Hidden 1</b><br>512-d", "text;html=1;align=center;fontSize=9;fontColor=#475569;", 510, 965, 50, 25)
    for j in range(4):
        add_node(f"node_h1_{j}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#FDEBD0;strokeColor=#E67E22;strokeWidth=1.2;", 525, 1005 + j*28, 20, 20)

    # Layer 3: Hidden 2 (x=670)
    add_node("lbl_l3", "<b>Hidden 2</b><br>256-d", "text;html=1;align=center;fontSize=9;fontColor=#475569;", 655, 965, 50, 25)
    for k in range(3):
        add_node(f"node_h2_{k}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#D4EFDF;strokeColor=#27AE60;strokeWidth=1.2;", 670, 1018 + k*30, 20, 20)

    # Layer 4: Output (x=820)
    add_node("lbl_l4", "<b>Output</b><br>82 Classes", "text;html=1;align=center;fontSize=9;fontColor=#475569;", 800, 965, 60, 25)
    for m in range(3):
        add_node(f"node_out_{m}", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#D6EAF8;strokeColor=#2980B9;strokeWidth=1.2;", 820, 1018 + m*30, 20, 20)

    # Draw Synaptic Lines between layers
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

    # Annotation at bottom of NN graph
    add_node("lbl_nn_annot", "<font color='#64748B' style='font-size:9.5px;'>Linear(1024→512) + LN + GELU + Drop &nbsp;──►&nbsp; Linear(512→256) &nbsp;──►&nbsp; Linear(256→82 Classes)</font>",
             "text;html=1;align=center;verticalAlign=middle;", 370, 1140, 580, 20)

    # Connect Pooling to NN Graph
    add_edge("e_pool_nn", "box_pool", "box_nn_graph", st_arr, "1024-d Descriptor")

    # Connect Train Data into Training Model
    add_edge("e_train_in", "card_train_data", "center_model", st_arr, "Mini-Batches (6,408 clips)", exit_x=0.0, exit_y=0.5, entry_x=1.0, entry_y=0.15)

    # -------------------------------------------------------------
    # LEFT BOX: MODEL EVALUATION (Reference 7 style)
    # -------------------------------------------------------------
    eval_val = (
        "<b style='font-size:13px; color:#15803D;'>Model Evaluation</b><br>"
        "<hr style='border:0; border-top:1px solid #BBF7D0; margin:6px 0;'>"
        "<span style='font-size:10px; color:#334155;'>Independent Generalization:</span><br>"
        "<div style='background:#DCFCE7; border:1px solid #16A34A; border-radius:5px; padding:6px; margin:6px 0; text-align:center;'>"
        "<b style='color:#14532D; font-size:13px;'>Top-1 Accuracy: 90.19%</b><br>"
        "<b style='color:#14532D; font-size:11.5px;'>Top-5 Accuracy: 99.02%</b>"
        "</div>"
        "<table width='100%' style='font-size:9.5px; color:#475569; text-align:left;'>"
        "<tr><td>• <b>Val Loss:</b> 0.4397</td><td>• <b>Epoch:</b> 23</td></tr>"
        "<tr><td>• <b>Macro F1:</b> 90.19%</td><td>• <b>Clips:</b> 2,743</td></tr>"
        "</table><br>"
        "<span style='font-size:9px; color:#64748B;'>82×82 Normalized Confusion Matrix verification</span>"
    )
    add_node("box_eval_left", eval_val,
             st_card_white + "fillColor=#F0FDF4;strokeColor=#16A34A;", 70, 680, 240, 260)

    # Connect Test Data to Model Evaluation
    add_edge("e_test_eval", "card_test_data", "box_eval_left", st_arr, "Test Input", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # Bidirectional Validation Loops between Training Model & Model Evaluation (Reference 7 style)
    add_edge("e_model_eval_top", "center_model", "box_eval_left", st_arr_loop, "Accuracy & Loss Checks", exit_x=0.0, exit_y=0.35, entry_x=1.0, entry_y=0.25)
    add_edge("e_model_eval_bot", "box_eval_left", "center_model", st_arr, "Best Weights Checkpoint", exit_x=1.0, exit_y=0.75, entry_x=0.0, entry_y=0.55)

    # -------------------------------------------------------------
    # RIGHT BOX: HYPERPARAMETER OPTIMIZATION (Reference 7 style)
    # -------------------------------------------------------------
    hyper_val = (
        "<b style='font-size:13px; color:#92400E;'>Hyperparameter<br>Optimization</b><br>"
        "<hr style='border:0; border-top:1px solid #FDE68A; margin:6px 0;'>"
        "<table width='100%' style='font-size:10px; color:#334155; text-align:left; line-height:1.6;'>"
        "<tr><td>• <b>Optimizer:</b> AdamW</td></tr>"
        "<tr><td>• <b>Base LR:</b> 1e-4</td></tr>"
        "<tr><td>• <b>Weight Decay:</b> 1e-2</td></tr>"
        "<tr><td>• <b>Scheduler:</b> CosineAnnealing</td></tr>"
        "<tr><td>• <b>Warmup:</b> 5 Epochs</td></tr>"
        "<tr><td>• <b>Multimodal Mixup:</b> β = 0.4</td></tr>"
        "<tr><td>• <b>Kinematic Jitter:</b> σ = 0.008</td></tr>"
        "<tr><td>• <b>Frame Dropout:</b> p = 0.1</td></tr>"
        "</table>"
    )
    add_node("box_hyper_right", hyper_val,
             st_card_white + "fillColor=#FEF9E7;strokeColor=#D4AC0D;", 1010, 680, 240, 260)

    # Bidirectional Hyperparameter Training loops (Reference 7 style)
    add_edge("e_hyper_loop1", "center_model", "box_hyper_right", st_arr_loop, "Adaptive Tuning", exit_x=1.0, exit_y=0.35, entry_x=0.0, entry_y=0.25)
    add_edge("e_hyper_loop2", "box_hyper_right", "center_model", st_arr, "Updated Parameters", exit_x=0.0, exit_y=0.75, entry_x=1.0, entry_y=0.55)

    # -------------------------------------------------------------
    # BOTTOM: FOCAL LOSS FORMULATION & RECOGNITION OUTPUT
    # -------------------------------------------------------------
    # Focal Loss box below model
    loss_val = (
        "<b>Focal Loss with Label Smoothing Regularization</b><br>"
        "<span style='font-size:11px; font-weight:bold; color:#B91C1C;'>L_focal = - ∑_(c=1)^82 α_c · (1 - p_c)^γ · y_c^(smooth) · log(p_c)</span><br>"
        "<span style='font-size:9.5px; color:#475569;'>γ = 1.5 (Focuses on hard sports) &nbsp;|&nbsp; α_c = 0.08 (Balances classes) &nbsp;|&nbsp; ε = 0.08 (Prevents overconfidence)</span>"
    )
    add_node("box_loss", loss_val,
             "rounded=1;arcSize=6;fillColor=#FFF1F2;strokeColor=#E11D48;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;",
             340, 1400, 640, 65)

    # Arrow from NN output to Loss
    add_edge("e_nn_loss", "box_nn_graph", "box_loss", st_arr, "Predicted Logits ŷ ∈ ℝ^82", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

    # Gradient Backprop arrow from loss back up to central model
    add_edge("e_loss_backprop", "box_loss", "center_model", st_arr, "∇ L_focal (End-to-End Backpropagation)", exit_x=0.15, exit_y=0.0, entry_x=0.15, entry_y=1.0)

    # Final Action Recognition Output card (Reference 7 bottom output style)
    action_val = (
        "<b style='font-size:13px; color:#15803D;'>Fine-Grained Sports Action Output (82 Classes)</b><br>"
        "<span style='font-size:10px; color:#334155;'>Top-1 &amp; Top-5 Predicted Categories (e.g. <i>TennisSwing 94.6%</i>, <i>HighJump 92.1%</i>)</span>"
    )
    add_node("box_final_out", action_val,
             st_card_white + "fillColor=#DCFCE7;strokeColor=#16A34A;", 340, 1495, 640, 60)
    add_edge("e_loss_out", "box_loss", "box_final_out", st_arr, "Inference Output")

    # Caption at bottom
    caption = "<b>Fig. 1.</b> System Architecture Diagram: CMAT-HOI Multimodal Sports Action Recognition Framework."
    add_node("cap", caption,
             "text;html=1;align=center;verticalAlign=middle;fontSize=12;fontColor=#475569;fontStyle=2;",
             20, 1585, 1280, 25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    print(f"Generated Reference-7 style Draw.io diagram at: {output_path}")


def render_companion_png(output_png: Path, output_svg: Path):
    """Render matching high-resolution graphic (PNG and SVG)."""
    fig, ax = plt.subplots(figsize=(13.2, 16.8), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 16.8)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    def card(x, y, w, h, bg, stroke, lw=1.5, ls="-", radius=0.03):
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}", facecolor=bg, edgecolor=stroke, linewidth=lw, linestyle=ls)
        ax.add_patch(r)
        return r

    # Outer Container Frame (Reference 7 style)
    card(0.2, 0.2, 12.8, 16.2, "#EBF5FB", "#2980B9", lw=2, radius=0.02)

    # 1. TOP DASHED BOX: DATA COLLECTION & DATASET
    card(1.6, 14.8, 10.0, 1.2, "#F8FAFC", "#475569", lw=1.5, ls="--")
    
    card(2.0, 15.0, 2.0, 0.8, "#EFF6FF", "#3B82F6")
    ax.text(3.0, 15.5, "Data Collection", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(3.0, 15.2, "Sports Video Ingestion", ha="center", fontsize=7.5, color="#475569")

    card(5.4, 15.0, 2.4, 0.8, "#E0F2FE", "#0284C7")
    ax.text(6.6, 15.5, "UCF-101 Repository", ha="center", fontsize=9.5, fontweight="bold", color="#0369A1")
    ax.text(6.6, 15.2, "Sports Action Benchmark", ha="center", fontsize=7.5, color="#475569")

    card(9.2, 15.0, 2.0, 0.8, "#DBEAFE", "#2563EB")
    ax.text(10.2, 15.5, "Data Set", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(10.2, 15.2, "9,151 Clips • 82 Classes", ha="center", fontsize=8, fontweight="bold", color="#1D4ED8")

    ax.annotate("", xy=(5.4, 15.4), xytext=(4.0, 15.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))
    ax.annotate("", xy=(9.2, 15.4), xytext=(7.8, 15.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))

    # 2. PREPROCESSING FUNNEL (Reference 7 Inverted Funnel)
    # Side badge
    card(1.0, 13.2, 1.8, 0.7, "#E8DAEF", "#8E44AD", lw=1.5)
    ax.text(1.9, 13.65, "Data Filtering &", ha="center", fontsize=8.5, fontweight="bold", color="#4A235A")
    ax.text(1.9, 13.4, "Conditioning", ha="center", fontsize=8.5, fontweight="bold", color="#4A235A")
    ax.annotate("", xy=(3.1, 13.55), xytext=(2.8, 13.55), arrowprops=dict(arrowstyle="->", lw=1.8, color="#8E44AD"))

    # 5 Funnel Bands
    fbands = [
        ("Data Cleaning & Temporal Sampling: Extract T = 16 frames (224×224 px)", 3.1, 14.15, 7.0, "#D5F5E3", "#27AE60", "#145A32"),
        ("Missing Value Handling: Zero-fill missing joints & Kalman apparatus tracking", 3.55, 13.75, 6.1, "#FCF3CF", "#F39C12", "#7D6608"),
        ("Kinematic Extraction: 1st-order temporal displacement velocities (Δx, Δy)", 4.05, 13.35, 5.1, "#D4E6F1", "#2980B9", "#1B4F72"),
        ("Outlier Handling: Keypoint confidence thresholding (c > 0.3) & aspect validation", 4.55, 12.95, 4.1, "#D1F2EB", "#1ABC9C", "#0E6251"),
        ("Standardization: Zero-center normalization & FP32 tensor packaging", 5.05, 12.55, 3.1, "#FADBD8", "#E74C3C", "#78281F")
    ]
    for ftext, fx, fy, fw, fbg, fstroke, ftextc in fbands:
        card(fx, fy, fw, 0.34, fbg, fstroke, lw=1.2, radius=0.015)
        ax.text(6.6, fy + 0.17, ftext, ha="center", va="center", fontsize=7.8, fontweight="bold", color=ftextc)

    # Big Downward Stem Arrow
    ax.annotate("", xy=(6.6, 11.95), xytext=(6.6, 12.55),
                arrowprops=dict(arrowstyle="->", lw=4, color="#34495E"))

    # Connect Dataset to Funnel
    ax.annotate("", xy=(9.2, 14.15), xytext=(10.2, 15.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B", connectionstyle="arc3,rad=0.15"))

    # 3. SPLIT: TEST DATA (LEFT) & TRAIN DATA (RIGHT)
    card(0.7, 10.85, 2.4, 0.85, "#FEF9E7", "#D4AC0D", lw=1.5)
    ax.text(1.9, 11.45, "Test Data (30%)", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(1.9, 11.2, "2,743 Video Clips", ha="center", fontsize=10, fontweight="bold", color="#15803D")
    ax.text(1.9, 10.98, "Strictly Disjoint Split", ha="center", fontsize=7.2, color="#475569")

    card(10.1, 10.85, 2.4, 0.85, "#FEF9E7", "#D4AC0D", lw=1.5)
    ax.text(11.3, 11.45, "Train Data (70%)", ha="center", fontsize=9.5, fontweight="bold", color="#1D4ED8")
    ax.text(11.3, 11.2, "6,408 Video Clips", ha="center", fontsize=10, fontweight="bold", color="#1E40AF")
    ax.text(11.3, 10.98, "Subject-Independent", ha="center", fontsize=7.2, color="#475569")

    ax.annotate("", xy=(1.9, 11.7), xytext=(6.2, 12.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B", connectionstyle="arc3,rad=0.15"))
    ax.text(3.6, 12.05, "30% Disjoint Split", fontsize=7.5, fontweight="bold", color="#1E40AF")

    ax.annotate("", xy=(11.3, 11.7), xytext=(7.0, 12.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B", connectionstyle="arc3,rad=-0.15"))
    ax.text(8.2, 12.05, "70% Training Split", fontsize=7.5, fontweight="bold", color="#1D4ED8")

    # 4. CENTER CARD: TRAINING MODEL (CMAT-HOI FRAMEWORK)
    card(3.4, 2.8, 6.4, 9.1, "#FEF9E7", "#B7950B", lw=2, radius=0.03)
    ax.text(6.6, 11.65, "Training Model: CMAT-HOI Framework", ha="center", fontsize=11, fontweight="bold", color="#78350F")

    # Multimodal Inputs
    ax.text(6.6, 11.35, "Multimodal Input Streams (T = 16 Frames)", ha="center", fontsize=8.5, fontweight="bold", color="#78350F")
    sboxes = [
        ("Visual\nResNet-50\n[16, 2048]", 3.6, "#DBEAFE", "#2563EB"),
        ("Skeletal Pose\nKeypoint R-CNN\n[16, 85]", 5.15, "#D1FAE5", "#059669"),
        ("Equipment\nFaster R-CNN\n[16, 7]", 6.7, "#FEF3C7", "#D97706"),
        ("HOI Kinematics\nKinematic Engine\n[16, 16]", 8.25, "#FCE7F3", "#DB2777")
    ]
    for stext, sx, sbg, sstroke in sboxes:
        card(sx, 10.5, 1.4, 0.7, sbg, sstroke, lw=1.2)
        ax.text(sx + 0.7, 10.85, stext, ha="center", va="center", fontsize=7, fontweight="bold", color="#1E293B")

    # Modality Stems
    card(3.6, 9.75, 6.0, 0.55, "#FEF9C3", "#D97706", lw=1.2)
    ax.text(6.6, 10.1, "Modality Alignment Stems (1D Conv + LayerNorm)", ha="center", fontsize=8.5, fontweight="bold", color="#92400E")
    ax.text(6.6, 9.9, "Unified Latent Dimension: E_vis, E_pose, E_obj, E_hoi ∈ ℝ^(16 × 256)", ha="center", fontsize=7.5, color="#451A03")

    for sx in [4.3, 5.85, 7.4, 8.95]:
        ax.annotate("", xy=(sx, 10.3), xytext=(sx, 10.5), arrowprops=dict(arrowstyle="->", lw=1, color="#D97706"))

    # Bi-Directional Cross-Attention
    card(3.6, 8.9, 6.0, 0.65, "#FDF2F8", "#BE185D", lw=1.4)
    ax.text(6.6, 9.32, "Bi-Directional Keypoint–Equipment Cross-Attention Module (h = 4 Heads)", ha="center", fontsize=8.5, fontweight="bold", color="#831843")
    ax.text(6.6, 9.08, "Pose queries Equipment (P_enh) ◄──► Equipment queries Pose (O_enh)", ha="center", fontsize=7.8, color="#4C0519")
    ax.annotate("", xy=(6.6, 9.55), xytext=(6.6, 9.75), arrowprops=dict(arrowstyle="->", lw=1.2, color="#BE185D"))

    # Gated Multimodal Fusion Unit (GMFU)
    card(3.6, 8.1, 6.0, 0.6, "#FEF3C7", "#B45309", lw=1.4)
    ax.text(6.6, 8.48, "Gated Multimodal Fusion Unit (GMFU) — Dynamic Modality Trust", ha="center", fontsize=8.5, fontweight="bold", color="#78350F")
    ax.text(6.6, 8.25, "Softmax Modality Gating Vector: g_t ∈ ℝ^4 ──► Fused Token F_t = ∑ g_t,m · E_m,t ∈ ℝ^256", ha="center", fontsize=7.8, color="#451A03")
    ax.annotate("", xy=(6.6, 8.7), xytext=(6.6, 8.9), arrowprops=dict(arrowstyle="->", lw=1.2, color="#B45309"))

    # Spatio-Temporal Transformer Encoder
    card(3.6, 7.2, 6.0, 0.7, "#ECFDF5", "#059669", lw=1.4)
    ax.text(6.6, 7.65, "Spatio-Temporal Transformer Encoder (3 Layers, 8 Attention Heads)", ha="center", fontsize=8.5, fontweight="bold", color="#064E3B")
    ax.text(6.6, 7.42, "Input: Z_0 = [ z_CLS ; F_1 ; F_2 ; ... ; F_16 ] + E_pos ∈ ℝ^(17 × 256)", ha="center", fontsize=7.5, color="#047857")
    ax.text(6.6, 7.28, "Pre-LayerNorm + Multi-Head Self-Attention + FFN (d_ff = 1024) + Residual Skips", ha="center", fontsize=7.2, color="#022C22")
    ax.annotate("", xy=(6.6, 7.9), xytext=(6.6, 8.1), arrowprops=dict(arrowstyle="->", lw=1.2, color="#059669"))

    # Multi-Scale 4-Way Temporal Pooling Head
    card(3.6, 6.35, 6.0, 0.65, "#CFFAFE", "#0891B2", lw=1.4)
    ax.text(6.6, 6.75, "Multi-Scale 4-Way Temporal Pooling Head", ha="center", fontsize=8.5, fontweight="bold", color="#0E7490")
    ax.text(6.6, 6.50, "Concatenation: Z_pool = [ z_CLS ; Attention Saliency ; Max Pool ; Avg Pool ] ∈ ℝ^1024", ha="center", fontsize=7.8, color="#164E63")
    ax.annotate("", xy=(6.6, 7.0), xytext=(6.6, 7.2), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0891B2"))

    # Neural Network Graph Schematic (Exact Reference 5 & 7 style!)
    card(3.6, 3.1, 6.0, 3.0, "#FFFFFF", "#CBD5E1", lw=1.5, radius=0.02)
    ax.text(6.6, 5.9, "3-Layer Multi-Layer Perceptron (MLP) Classifier Architecture", ha="center", fontsize=9, fontweight="bold", color="#1E293B")

    # Layer columns
    ax.text(4.2, 5.6, "Input Layer\n(1024-d)", ha="center", fontsize=7.5, fontweight="bold", color="#DC3545")
    ax.text(5.5, 5.6, "Hidden 1\n(512-d)", ha="center", fontsize=7.5, fontweight="bold", color="#E67E22")
    ax.text(6.8, 5.6, "Hidden 2\n(256-d)", ha="center", fontsize=7.5, fontweight="bold", color="#27AE60")
    ax.text(8.2, 5.6, "Output Layer\n(82 Classes)", ha="center", fontsize=7.5, fontweight="bold", color="#2980B9")

    # Node coordinates
    in_nodes = [(4.2, 5.2 - i*0.35) for i in range(5)]
    h1_nodes = [(5.5, 5.0 - j*0.38) for j in range(4)]
    h2_nodes = [(6.8, 4.8 - k*0.42) for k in range(3)]
    out_nodes = [(8.2, 4.8 - m*0.42) for m in range(3)]

    # Draw synaptic lines
    for p1 in in_nodes:
        for p2 in h1_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#CBD5E1", lw=0.6, zorder=1)
    for p1 in h1_nodes:
        for p2 in h2_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#CBD5E1", lw=0.6, zorder=1)
    for p1 in h2_nodes:
        for p2 in out_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#CBD5E1", lw=0.6, zorder=1)

    # Draw circular neuron dots
    for x, y in in_nodes:
        ax.scatter(x, y, s=180, color="#F8D7DA", edgecolors="#DC3545", linewidths=1.5, zorder=2)
    for x, y in h1_nodes:
        ax.scatter(x, y, s=180, color="#FDEBD0", edgecolors="#E67E22", linewidths=1.5, zorder=2)
    for x, y in h2_nodes:
        ax.scatter(x, y, s=180, color="#D4EFDF", edgecolors="#27AE60", linewidths=1.5, zorder=2)
    for x, y in out_nodes:
        ax.scatter(x, y, s=180, color="#D6EAF8", edgecolors="#2980B9", linewidths=1.5, zorder=2)

    ax.text(6.6, 3.25, "Linear(1024→512) + LN + GELU + Drop  ──►  Linear(512→256)  ──►  Linear(256→82)",
            ha="center", fontsize=7.2, color="#64748B")

    ax.annotate("", xy=(6.6, 6.1), xytext=(6.6, 6.35), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0891B2"))

    # Connect Train Data to Model
    ax.annotate("", xy=(9.8, 11.25), xytext=(10.1, 11.25), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))

    # 5. LEFT BOX: MODEL EVALUATION (Reference 7 style)
    card(0.7, 5.6, 2.4, 3.8, "#F0FDF4", "#16A34A", lw=1.5)
    ax.text(1.9, 9.1, "Model Evaluation", ha="center", fontsize=10.5, fontweight="bold", color="#15803D")
    ax.plot([0.9, 2.9], [8.9, 8.9], color="#BBF7D0", lw=1)
    
    card(0.9, 7.8, 2.0, 0.9, "#DCFCE7", "#16A34A", lw=1.2)
    ax.text(1.9, 8.4, "Top-1: 90.19%", ha="center", fontsize=10.5, fontweight="bold", color="#14532D")
    ax.text(1.9, 8.05, "Top-5: 99.02%", ha="center", fontsize=9.5, fontweight="bold", color="#14532D")

    eval_text = [
        "• Val Loss: 0.4397",
        "• Peak Epoch: 23",
        "• Macro F1: 90.19%",
        "• 2,743 Disjoint Test Clips",
        "• 82×82 Confusion Matrix"
    ]
    for i, line in enumerate(eval_text):
        ax.text(0.95, 7.4 - i * 0.32, line, ha="left", fontsize=7.8, color="#1E293B")

    # Connect Test Data to Model Evaluation
    ax.annotate("", xy=(1.9, 9.4), xytext=(1.9, 10.85), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E293B"))

    # Bidirectional loops between Model and Evaluation (Reference 7 style)
    ax.annotate("", xy=(3.4, 8.4), xytext=(2.9, 8.4), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#1E293B"))
    ax.text(3.15, 8.55, "Evaluation", ha="center", fontsize=7.2, color="#1E293B")

    # 6. RIGHT BOX: HYPERPARAMETER OPTIMIZATION (Reference 7 style)
    card(10.1, 5.6, 2.4, 3.8, "#FEF9E7", "#D4AC0D", lw=1.5)
    ax.text(11.3, 9.1, "Hyperparameter", ha="center", fontsize=10.5, fontweight="bold", color="#92400E")
    ax.text(11.3, 8.8, "Optimization", ha="center", fontsize=10.5, fontweight="bold", color="#92400E")
    ax.plot([10.3, 12.3], [8.65, 8.65], color="#FDE68A", lw=1)

    hyper_text = [
        "• Optimizer: AdamW",
        "• Learning Rate: 1e-4",
        "• Weight Decay: 1e-2",
        "• CosineAnnealingLR",
        "• Linear Warmup (5 ep)",
        "• Multimodal Mixup: 0.4",
        "• Joint Jitter: σ = 0.008",
        "• Frame Dropout: p = 0.1"
    ]
    for i, line in enumerate(hyper_text):
        ax.text(10.3, 8.35 - i * 0.32, line, ha="left", fontsize=7.8, color="#1E293B")

    # Bidirectional loops between Model and Hyperparameter (Reference 7 style)
    ax.annotate("", xy=(9.8, 8.4), xytext=(10.1, 8.4), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#1E293B"))
    ax.text(9.95, 8.55, "Tuning", ha="center", fontsize=7.2, color="#1E293B")

    # 7. BOTTOM: FOCAL LOSS & RECOGNITION OUTPUT
    card(3.4, 1.7, 6.4, 0.85, "#FFF1F2", "#E11D48", lw=1.5)
    ax.text(6.6, 2.3, "Focal Loss with Label Smoothing Regularization", ha="center", fontsize=9, fontweight="bold", color="#9F1239")
    ax.text(6.6, 2.05, "L_focal = - ∑ α_c · (1 - p_c)^γ · y_c^(smooth) · log(p_c)", ha="center", fontsize=8.5, fontweight="bold", color="#B91C1C")
    ax.text(6.6, 1.85, "γ = 1.5 (Hard sports)  •  α_c = 0.08 (Imbalance)  •  ε = 0.08 (Label smoothing)", ha="center", fontsize=7.5, color="#475569")

    # Arrow from NN to Loss
    ax.annotate("", xy=(6.6, 2.55), xytext=(6.6, 3.1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#E11D48"))
    ax.text(6.6, 2.8, "Predicted Logits ŷ ∈ ℝ^82", ha="center", fontsize=7.2, fontweight="bold", color="#9F1239")

    # Backpropagation arrow from Loss up to Model
    ax.annotate("", xy=(3.4, 4.5), xytext=(3.4, 2.1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#E11D48", connectionstyle="arc3,rad=-0.2"))
    ax.text(2.6, 3.2, "∇ Loss\n(Backprop)", ha="center", fontsize=7.5, fontweight="bold", color="#E11D48")

    # Final Output card
    card(3.4, 0.7, 6.4, 0.75, "#DCFCE7", "#16A34A", lw=1.5)
    ax.text(6.6, 1.18, "Fine-Grained Sports Action Output (82 Classes)", ha="center", fontsize=10, fontweight="bold", color="#15803D")
    ax.text(6.6, 0.92, "Top-1 & Top-5 Predicted Categories (e.g. TennisSwing 94.6%, HighJump 92.1%)", ha="center", fontsize=8, color="#14532D")

    ax.annotate("", xy=(6.6, 1.45), xytext=(6.6, 1.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16A34A"))

    # Caption at bottom
    ax.text(6.6, 0.35, "Fig. 1. System Architecture Diagram: CMAT-HOI Multimodal Sports Action Recognition Framework.",
            ha="center", va="center", fontsize=9.5, fontfamily="serif", fontstyle="italic", color="#334155")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()
    print(f"Generated Reference-7 style companion PNG at: {output_png}")
    print(f"Generated Reference-7 style companion SVG at: {output_svg}")


if __name__ == "__main__":
    out_drawio_paper = Path("Paper writing/CMAT_HOI_Methodology_Architecture.drawio")
    out_drawio_m7 = Path("diagrams/M7/CMAT_HOI_Methodology_Architecture.drawio")
    build_drawio_xml(out_drawio_paper)
    build_drawio_xml(out_drawio_m7)

    out_png_paper = Path("Paper writing/figures/methodology_architecture_diagram.png")
    out_svg_paper = Path("Paper writing/figures/methodology_architecture_diagram.svg")
    out_png_m7 = Path("diagrams/M7/methodology_architecture_diagram.png")
    out_svg_m7 = Path("diagrams/M7/methodology_architecture_diagram.svg")

    render_companion_png(out_png_paper, out_svg_paper)
    render_companion_png(out_png_m7, out_svg_m7)
