#!/usr/bin/env python3
"""
Generate a professional, fully-featured Draw.io architectural diagram (.drawio XML)
for the CMAT-HOI Sports Action Recognition thesis project.

The layout is inspired by academic system architecture diagrams (References 4, 6, 7),
structured into 5 clear end-to-end phases:
  - Phase 1: Dataset Ingestion, Preprocessing Funnel & Disjoint Split
  - Phase 2: Synchronized Multimodal Feature Extraction (4 Parallel Streams)
  - Phase 3: Core CMAT-HOI Network Architecture (Stems, Cross-Attention, GMFU, Transformer, 4-Way Pooling)
  - Phase 4: Model Training, Optimization & Rigorous Evaluation (Focal Loss, Peak Epoch 23 Milestone)
  - Phase 5: Real-Time Multi-Platform Deployment & Inference Engine
"""

import xml.etree.ElementTree as ET
from pathlib import Path


class DrawioGenerator:
    def __init__(self, width=1700, height=2460):
        self.root = ET.Element("mxfile", host="app.diagrams.net", version="24.0.0", type="device")
        self.diagram = ET.SubElement(self.root, "diagram", id="CMAT_HOI_Architecture", name="Methodology Architecture")
        self.graph = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            dx="1600",
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
            pageWidth=str(width),
            pageHeight=str(height),
            math="0",
            shadow="0"
        )
        self.root_cells = ET.SubElement(self.graph, "root")
        ET.SubElement(self.root_cells, "mxCell", id="0")
        ET.SubElement(self.root_cells, "mxCell", id="1", parent="0")
        self.cell_count = 1

    def add_node(self, node_id, value, style, x, y, width, height, parent="1"):
        cell = ET.SubElement(
            self.root_cells,
            "mxCell",
            id=node_id,
            value=value,
            style=style,
            vertex="1",
            parent=parent
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=str(x),
            y=str(y),
            width=str(width),
            height=str(height),
            **{"as": "geometry"}
        )
        return cell

    def add_edge(self, edge_id, source_id, target_id, style, value="", parent="1", entry_x=None, entry_y=None, exit_x=None, exit_y=None):
        edge_style = style
        if entry_x is not None and entry_y is not None:
            edge_style += f";entryX={entry_x};entryY={entry_y};entryDx=0;entryDy=0"
        if exit_x is not None and exit_y is not None:
            edge_style += f";exitX={exit_x};exitY={exit_y};exitDx=0;exitDy=0"

        cell = ET.SubElement(
            self.root_cells,
            "mxCell",
            id=edge_id,
            value=value,
            style=edge_style,
            edge="1",
            parent=parent,
            source=source_id,
            target=target_id
        )
        geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        return cell

    def build_diagram(self):
        # Default Styles
        style_header = (
            "rounded=1;arcSize=10;fillColor=#0F172A;strokeColor=#1E293B;strokeWidth=2;"
            "shadow=1;html=1;align=center;verticalAlign=middle;"
        )

        style_container_p1 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#2563EB;"
            "strokeWidth=2;dashed=1;verticalAlign=top;spacingTop=10;fontStyle=1;fontSize=14;"
            "fontColor=#1E3A8A;container=1;collapsible=0;"
        )
        style_container_p2 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#9333EA;"
            "strokeWidth=2;dashed=1;verticalAlign=top;spacingTop=10;fontStyle=1;fontSize=14;"
            "fontColor=#581C87;container=1;collapsible=0;"
        )
        style_container_p3 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FFFDF5;strokeColor=#D97706;"
            "strokeWidth=2;verticalAlign=top;spacingTop=10;fontStyle=1;fontSize=15;"
            "fontColor=#78350F;container=1;collapsible=0;"
        )
        style_container_p4 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#DC2626;"
            "strokeWidth=2;dashed=1;verticalAlign=top;spacingTop=10;fontStyle=1;fontSize=14;"
            "fontColor=#991B1B;container=1;collapsible=0;"
        )
        style_container_p5 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F0FDFA;strokeColor=#0D9488;"
            "strokeWidth=2;verticalAlign=top;spacingTop=10;fontStyle=1;fontSize=14;"
            "fontColor=#115E59;container=1;collapsible=0;"
        )

        style_card_white = (
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;"
            "strokeWidth=1.5;shadow=1;align=left;verticalAlign=top;spacing=10;fontSize=11;fontColor=#1E293B;"
        )
        style_card_center = (
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;"
            "strokeWidth=1.5;shadow=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#1E293B;"
        )

        style_edge_ortho = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            "strokeColor=#475569;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#334155;"
        )
        style_edge_blue = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            "strokeColor=#2563EB;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#1E40AF;"
        )
        style_edge_purple = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            "strokeColor=#9333EA;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#6B21A8;"
        )
        style_edge_amber = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            "strokeColor=#D97706;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#92400E;"
        )
        style_edge_red = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            "strokeColor=#DC2626;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#991B1B;"
        )
        style_edge_green = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            "strokeColor=#059669;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#065F46;"
        )
        style_edge_teal = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            "strokeColor=#0D9488;strokeWidth=2;endArrow=block;endFill=1;fontSize=10;fontColor=#115E59;"
        )

        # =========================================================================
        # 0. TOP TITLE BANNER
        # =========================================================================
        title_val = (
            "<b><font style='font-size: 21px; color: #FFFFFF; letter-spacing: 0.5px;'>"
            "METHODOLOGY ARCHITECTURAL DIAGRAM: CMAT-HOI SPORTS ACTION RECOGNITION"
            "</font></b><br>"
            "<font style='font-size: 13px; color: #94A3B8;'>"
            "End-to-End Multimodal Cross-Modal Attention Transformer with Human-Object Interaction Modeling "
            "for 82 Fine-Grained Sports Classes"
            "</font>"
        )
        self.add_node("banner_title", title_val, style_header, 60, 30, 1580, 80)

        # =========================================================================
        # PHASE 1: DATASET INGESTION, PREPROCESSING FUNNEL & DISJOINT SPLITTING
        # =========================================================================
        p1_val = "<b>PHASE 1: DATASET INGESTION, PREPROCESSING FUNNEL &amp; DISJOINT SPLITTING</b>"
        self.add_node("container_p1", p1_val, style_container_p1, 60, 130, 1580, 330)

        # Box 1.1: Data Repository & Collection
        box_data_repo = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #1E40AF;'>"
            "📁 UCF-101 Sports Action Repository"
            "</div><hr style='border: 0; border-top: 1px solid #BFDBFE; margin: 6px 0;'>"
            "• <b>Total Video Clips:</b> 9,151 Full Action Videos<br>"
            "• <b>Target Categories:</b> 82 Fine-Grained Classes<br>"
            "• <b>Complex Dynamics:</b><br>"
            "&nbsp;&nbsp;- Ball &amp; Racket (Tennis, Basketball, Cricket)<br>"
            "&nbsp;&nbsp;- Track &amp; Gymnastics (HighJump, Vault, Floor)<br>"
            "&nbsp;&nbsp;- Water &amp; Winter (Surfing, Skiing, Rowing)<br>"
            "&nbsp;&nbsp;- Athletics &amp; Combat (Boxing, Fencing, Archery)<br>"
            "• <b>Raw Video Format:</b> RGB .avi / .mp4, 25-30 FPS<br>"
            "• <b>Challenges:</b> Extreme camera motion, severe occlusions, "
            "fast apparatus flight, and dynamic background clutter."
        )
        self.add_node("box_data_repo", box_data_repo, style_card_white, 80, 175, 330, 260)

        # Box 1.2: 5-Stage Preprocessing Pipeline / Funnel (Inspired by Reference 7)
        box_funnel_head = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #1E40AF;'>"
            "⚙️ 5-Stage Sports Video Preprocessing &amp; Conditioning Funnel"
            "</div>"
        )
        self.add_node("box_funnel_head", box_funnel_head, style_card_center, 440, 175, 750, 35)

        funnel_steps = [
            ("s1", "<b>Step 1: Uniform Temporal Discretization</b>", "Extract T = 16 equidistant frames spanning full action clip: t_i = floor(i × T_orig / 16)", "#DBEAFE", "#2563EB"),
            ("s2", "<b>Step 2: Spatial Resizing &amp; Normalization</b>", "Bilinear resize to 224×224 px; ImageNet standard normalization (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225])", "#E0E7FF", "#4F46E5"),
            ("s3", "<b>Step 3: Kinematic Velocity Computation</b>", "Compute 1st-order temporal difference vectors: Δp_t = p_t - p_(t-1) for body joints &amp; apparatus centroids", "#EDE9FE", "#7C3AED"),
            ("s4", "<b>Step 4: Apparatus Imputation &amp; Noise Filtering</b>", "Zero-filling missing equipment bboxes; Kalman smoothing on joints; frame-wise aspect ratio verification", "#FAE8FF", "#C026D3"),
            ("s5", "<b>Step 5: Pre-Flight Tensor Sanitization</b>", "Cast tensors to FP32, sanitize NaN/Inf occurrences, and package synchronized multimodal batches", "#FCE7F3", "#DB2777")
        ]
        curr_y = 217
        for sid, stitle, sdesc, sfill, sstroke in funnel_steps:
            s_val = f"<div style='text-align: left;'><span style='color: {sstroke}; font-size: 11.5px;'>{stitle}</span>: <font style='font-size: 10.5px; color: #334155;'>{sdesc}</font></div>"
            s_style = f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={sfill};strokeColor={sstroke};strokeWidth=1.2;align=left;spacingLeft=12;verticalAlign=middle;"
            self.add_node(f"funnel_{sid}", s_val, s_style, 440, curr_y, 750, 40)
            curr_y += 44

        # Box 1.3: Disjoint Split Partitioning (Zero Data Leakage)
        box_split = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #1E40AF;'>"
            "⚖️ Subject-Disjoint Dataset Split"
            "</div><hr style='border: 0; border-top: 1px solid #BFDBFE; margin: 6px 0;'>"
            "<div style='background: #EFF6FF; border: 1px solid #3B82F6; border-radius: 6px; padding: 6px; margin-bottom: 8px;'>"
            "<b>🏋️ Training Split (70%)</b><br>"
            "<span style='font-size: 13px; font-weight: bold; color: #1D4ED8;'>6,408 Video Clips</span><br>"
            "<font style='font-size: 10px; color: #475569;'>Subject-independent grouping; receives Multimodal Mixup &amp; Kinematic Jitter.</font>"
            "</div>"
            "<div style='background: #F0FDF4; border: 1px solid #16A34A; border-radius: 6px; padding: 6px;'>"
            "<b>🎯 Validation &amp; Test Split (30%)</b><br>"
            "<span style='font-size: 13px; font-weight: bold; color: #15803D;'>2,743 Video Clips</span><br>"
            "<font style='font-size: 10px; color: #475569;'>Unseen athletes &amp; backgrounds; evaluates real generalization (90.19% Top-1).</font>"
            "</div>"
            "<div style='font-size: 10px; color: #64748B; margin-top: 6px; text-align: center;'>"
            "Strict zero-leakage guarantee (no common video sessions)."
            "</div>"
        )
        self.add_node("box_split", box_split, style_card_white, 1220, 175, 400, 260)

        # Edges in Phase 1
        self.add_edge("e_p1_1", "box_data_repo", "box_funnel_head", style_edge_blue, "Raw Video Feeds")
        self.add_edge("e_p1_2", "funnel_s5", "box_split", style_edge_blue, "Sanitized Tensors")

        # =========================================================================
        # PHASE 2: SYNCHRONIZED MULTIMODAL FEATURE EXTRACTION (4 Parallel Streams)
        # =========================================================================
        p2_val = "<b>PHASE 2: SYNCHRONIZED MULTIMODAL FEATURE EXTRACTION (4 PARALLEL TEMPORAL STREAMS, T = 16)</b>"
        self.add_node("container_p2", p2_val, style_container_p2, 60, 480, 1580, 270)

        # Stream 1: Visual Appearance
        st1_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 12.5px; color: #1E40AF;'>"
            "📷 Visual Appearance Stream"
            "</div><hr style='border: 0; border-top: 1px solid #BFDBFE; margin: 4px 0;'>"
            "• <b>Backbone:</b> ResNet-50 (ImageNet Pretrained)<br>"
            "• <b>Layers:</b> Conv1 to Layer4 (avgpool pooling)<br>"
            "• <b>Extracted Tensor:</b><br>"
            "<div style='text-align: center; background: #DBEAFE; border-radius: 4px; padding: 3px; font-weight: bold; color: #1D4ED8; margin: 4px 0;'>"
            "X_vis ∈ ℝ^(16 × 2048)"
            "</div>"
            "• <b>Visual Context:</b> Scene layout, court surfaces, clothing, arena boundaries, actor orientation."
        )
        self.add_node("stream_vis", st1_val, style_card_white, 80, 525, 360, 205)

        # Stream 2: Skeletal Kinematic Pose
        st2_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 12.5px; color: #065F46;'>"
            "🏃 Skeletal Kinematics Stream"
            "</div><hr style='border: 0; border-top: 1px solid #A7F3D0; margin: 4px 0;'>"
            "• <b>Backbone:</b> Keypoint R-CNN (ResNet-50-FPN)<br>"
            "• <b>Extraction:</b> 17 Keypoints (x, y, c) = 51-d<br>"
            "• <b>Velocities:</b> First-order Δ(x, y) = 34-d<br>"
            "• <b>Extracted Tensor:</b><br>"
            "<div style='text-align: center; background: #D1FAE5; border-radius: 4px; padding: 3px; font-weight: bold; color: #047857; margin: 4px 0;'>"
            "X_pose ∈ ℝ^(16 × 85)"
            "</div>"
            "• <b>Biomechanics:</b> Articulated human posture, joint velocities, limb angles, athletic trajectory."
        )
        self.add_node("stream_pose", st2_val, style_card_white, 470, 525, 360, 205)

        # Stream 3: Sports Apparatus Detection
        st3_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 12.5px; color: #92400E;'>"
            "🎾 Sports Apparatus Stream"
            "</div><hr style='border: 0; border-top: 1px solid #FDE68A; margin: 4px 0;'>"
            "• <b>Backbone:</b> Faster R-CNN (FPN Backbone)<br>"
            "• <b>Extraction:</b> Top gear candidate per frame<br>"
            "• <b>Attributes:</b> [x1, y1, x2, y2, score, cls, aspect]<br>"
            "• <b>Extracted Tensor:</b><br>"
            "<div style='text-align: center; background: #FEF3C7; border-radius: 4px; padding: 3px; font-weight: bold; color: #B45309; margin: 4px 0;'>"
            "X_obj ∈ ℝ^(16 × 7)"
            "</div>"
            "• <b>Equipment:</b> Ball, racket, baseball bat, barbell, skateboard, bow coordinates &amp; motion paths."
        )
        self.add_node("stream_obj", st3_val, style_card_white, 860, 525, 360, 205)

        # Stream 4: Geometric HOI Kinematic Stream
        st4_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 12.5px; color: #9D174D;'>"
            "🤝 Geometric HOI Kinematics Stream"
            "</div><hr style='border: 0; border-top: 1px solid #FBCFE8; margin: 4px 0;'>"
            "• <b>Engine:</b> Deterministic Kinematic Engine<br>"
            "• <b>Interaction Geometry:</b> Wrists/Ankles ↔ Gear<br>"
            "• <b>Features:</b> Euclidean dist, unit angles, d(dist)/dt<br>"
            "• <b>Extracted Tensor:</b><br>"
            "<div style='text-align: center; background: #FCE7F3; border-radius: 4px; padding: 3px; font-weight: bold; color: #BE185D; margin: 4px 0;'>"
            "X_hoi ∈ ℝ^(16 × 16)"
            "</div>"
            "• <b>HOI Dynamics:</b> Ball strike instant, bat grip angle, release moment, landing foot-ground contact."
        )
        self.add_node("stream_hoi", st4_val, style_card_white, 1250, 525, 370, 205)

        # Edges from Train Split to 4 Streams
        self.add_edge("e_p1_s1", "box_split", "stream_vis", style_edge_purple, "Train Batches (6,408 clips)", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        self.add_edge("e_p1_s2", "box_split", "stream_pose", style_edge_purple, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        self.add_edge("e_p1_s3", "box_split", "stream_obj", style_edge_purple, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        self.add_edge("e_p1_s4", "box_split", "stream_hoi", style_edge_purple, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

        # =========================================================================
        # PHASE 3: CMAT-HOI CORE NETWORK ARCHITECTURE
        # =========================================================================
        p3_val = "<b>PHASE 3: CMAT-HOI CORE ARCHITECTURE (CROSS-MODAL ATTENTION TRANSFORMER &amp; GATED FUSION)</b>"
        self.add_node("container_p3", p3_val, style_container_p3, 60, 770, 1580, 750)

        # Substage 3A: Dimension-Aligned Modality Stems
        stem1_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 11px; color: #1E40AF;'>"
            "Visual Modality Stem"
            "</div>"
            "<font style='font-size: 9.5px; color: #334155;'>"
            "1D Conv (k=3, s=1, p=1) + LN + GELU<br>"
            "ℝ^(16 × 2048) ──► <b>E_vis ∈ ℝ^(16 × 256)</b>"
            "</font>"
        )
        self.add_node("stem_vis", stem1_val, "rounded=1;arcSize=6;fillColor=#DBEAFE;strokeColor=#1D4ED8;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 80, 815, 360, 60)

        stem2_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 11px; color: #065F46;'>"
            "Pose Modality Stem"
            "</div>"
            "<font style='font-size: 9.5px; color: #334155;'>"
            "1D Conv (k=3, s=1, p=1) + LN + GELU<br>"
            "ℝ^(16 × 85) ──► <b>E_pose ∈ ℝ^(16 × 256)</b>"
            "</font>"
        )
        self.add_node("stem_pose", stem2_val, "rounded=1;arcSize=6;fillColor=#D1FAE5;strokeColor=#047857;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 470, 815, 360, 60)

        stem3_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 11px; color: #92400E;'>"
            "Object Modality Stem"
            "</div>"
            "<font style='font-size: 9.5px; color: #334155;'>"
            "1D Conv (k=3, s=1, p=1) + LN + GELU<br>"
            "ℝ^(16 × 7) ──► <b>E_obj ∈ ℝ^(16 × 256)</b>"
            "</font>"
        )
        self.add_node("stem_obj", stem3_val, "rounded=1;arcSize=6;fillColor=#FEF3C7;strokeColor=#B45309;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 860, 815, 360, 60)

        stem4_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 11px; color: #9D174D;'>"
            "HOI Modality Stem"
            "</div>"
            "<font style='font-size: 9.5px; color: #334155;'>"
            "1D Conv (k=3, s=1, p=1) + LN + GELU<br>"
            "ℝ^(16 × 16) ──► <b>E_hoi ∈ ℝ^(16 × 256)</b>"
            "</font>"
        )
        self.add_node("stem_hoi", stem4_val, "rounded=1;arcSize=6;fillColor=#FCE7F3;strokeColor=#BE185D;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 1250, 815, 370, 60)

        # Edges Streams to Stems
        self.add_edge("e_s1_stem", "stream_vis", "stem_vis", style_edge_blue)
        self.add_edge("e_s2_stem", "stream_pose", "stem_pose", style_edge_green)
        self.add_edge("e_s3_stem", "stream_obj", "stem_obj", style_edge_amber)
        self.add_edge("e_s4_stem", "stream_hoi", "stem_hoi", style_edge_red)

        # Substage 3B: Bi-Directional Cross-Attention Mechanism
        cross_attn_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #831843;'>"
            "⚡ Bi-Directional Keypoint–Equipment Cross-Attention Mechanism (h = 4 Heads, d_k = 64)"
            "</div><hr style='border: 0; border-top: 1px solid #F472B6; margin: 5px 0;'>"
            "<table width='100%' style='font-size: 11px; color: #1E293B;'>"
            "<tr>"
            "<td width='50%' style='padding-right: 15px; border-right: 1px dashed #F472B6; vertical-align: top;'>"
            "<b>1. Pose Queries Equipment Attention (Pose-Guided Enhancement):</b><br>"
            "• Query: Q = E_pose &nbsp;|&nbsp; Key: K = E_obj &nbsp;|&nbsp; Value: V = E_obj<br>"
            "• Attention Weights: A_P→O = Softmax( (Q · K^T) / sqrt(d_k) ) ∈ ℝ^(16 × 16)<br>"
            "• <b>Enhanced Pose: P_enh = LayerNorm( E_pose + MHA(Q, K, V) ) ∈ ℝ^(16 × 256)</b><br>"
            "<font style='font-size: 10px; color: #64748B;'>Conditions human joint movements on where sports equipment is currently located.</font>"
            "</td>"
            "<td width='50%' style='padding-left: 15px; vertical-align: top;'>"
            "<b>2. Equipment Queries Pose Attention (Equipment-Guided Enhancement):</b><br>"
            "• Query: Q = E_obj &nbsp;|&nbsp; Key: K = E_pose &nbsp;|&nbsp; Value: V = E_pose<br>"
            "• Attention Weights: A_O→P = Softmax( (Q · K^T) / sqrt(d_k) ) ∈ ℝ^(16 × 16)<br>"
            "• <b>Enhanced Object: O_enh = LayerNorm( E_obj + MHA(Q, K, V) ) ∈ ℝ^(16 × 256)</b><br>"
            "<font style='font-size: 10px; color: #64748B;'>Resolves equipment trajectory ambiguity by attending to player hand/wrist kinematics.</font>"
            "</td>"
            "</tr>"
            "</table>"
        )
        self.add_node("box_cross_attn", cross_attn_val, "rounded=1;arcSize=8;fillColor=#FDF2F8;strokeColor=#BE185D;strokeWidth=2;html=1;align=left;spacing=10;shadow=1;", 350, 905, 990, 115)

        # Edges from Pose Stem and Object Stem to Cross Attention
        self.add_edge("e_stem_pose_cross", "stem_pose", "box_cross_attn", style_edge_red, "E_pose (256-d)", exit_x=0.5, exit_y=1.0, entry_x=0.25, entry_y=0.0)
        self.add_edge("e_stem_obj_cross", "stem_obj", "box_cross_attn", style_edge_red, "E_obj (256-d)", exit_x=0.5, exit_y=1.0, entry_x=0.75, entry_y=0.0)

        # Substage 3C: Gated Multimodal Fusion Unit (GMFU)
        gmfu_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #78350F;'>"
            "🎛️ Gated Multimodal Fusion Unit (GMFU) — Dynamic Frame-Level Modality Weighting"
            "</div><hr style='border: 0; border-top: 1px solid #FCD34D; margin: 5px 0;'>"
            "<table width='100%' style='font-size: 11px; color: #1E293B;'>"
            "<tr>"
            "<td width='55%' style='vertical-align: top;'>"
            "• <b>Modality Gating Vector:</b> "
            "<span style='color: #B45309; font-weight: bold;'>g_t = Softmax( W_g · [ E_vis,t ; P_enh,t ; O_enh,t ; E_hoi,t ] + b_g ) ∈ ℝ^4</span><br>"
            "• <b>Dynamic Fused Token:</b> "
            "<span style='color: #B45309; font-weight: bold;'>F_t = g_t,1 · E_vis,t + g_t,2 · P_enh,t + g_t,3 · O_enh,t + g_t,4 · E_hoi,t ∈ ℝ^256</span><br>"
            "• <b>Full Sequence:</b> Fused Feature Matrix <b>F = [F_1, F_2, ..., F_16]^T ∈ ℝ^(16 × 256)</b>"
            "</td>"
            "<td width='45%' style='vertical-align: top; background: #FEF9C3; border-radius: 4px; padding: 6px; border: 1px solid #F59E0B;'>"
            "<b>Adaptive Modality Trust Allocation:</b><br>"
            "<font style='font-size: 10px; color: #451A03;'>"
            "• Suppresses occluded keypoints during sudden body turns.<br>"
            "• Amplifies visual context during outdoor sports (e.g., Surfing, Skiing).<br>"
            "• Prioritizes HOI kinematic vectors during decisive ball-impact frames."
            "</font>"
            "</td>"
            "</tr>"
            "</table>"
        )
        self.add_node("box_gmfu", gmfu_val, "rounded=1;arcSize=8;fillColor=#FEF9C3;strokeColor=#D97706;strokeWidth=2;html=1;align=left;spacing=10;shadow=1;", 140, 1050, 1420, 100)

        # Edges into GMFU:
        # Visual Stem goes down the left
        self.add_edge("e_stem_vis_gmfu", "stem_vis", "box_gmfu", style_edge_amber, "E_vis (256-d)", exit_x=0.5, exit_y=1.0, entry_x=0.1, entry_y=0.0)
        # Cross-Attention outputs P_enh and O_enh go down
        self.add_edge("e_cross_gmfu", "box_cross_attn", "box_gmfu", style_edge_amber, "[P_enh; O_enh] (512-d)", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        # HOI Stem goes down the right
        self.add_edge("e_stem_hoi_gmfu", "stem_hoi", "box_gmfu", style_edge_amber, "E_hoi (256-d)", exit_x=0.5, exit_y=1.0, entry_x=0.9, entry_y=0.0)

        # Substage 3D: Spatio-Temporal Positional Encoding & Transformer Encoder
        trans_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 13.5px; color: #064E3B;'>"
            "🧠 Spatio-Temporal Transformer Encoder (Pre-LayerNorm, 3 Stacked Layers)"
            "</div><hr style='border: 0; border-top: 1px solid #6EE7B7; margin: 5px 0;'>"
            "<table width='100%' style='font-size: 11px; color: #1E293B;'>"
            "<tr>"
            "<td width='40%' style='vertical-align: top; border-right: 1px dashed #6EE7B7; padding-right: 10px;'>"
            "<b>1. Token Sequence &amp; Positional Encoding:</b><br>"
            "• Prepend Learnable Class Token: <b>z_CLS ∈ ℝ^256</b><br>"
            "• Add Fixed Sinusoidal Positional Encoding: <b>E_pos ∈ ℝ^(17 × 256)</b><br>"
            "• <b>Input Matrix: Z_0 = [ z_CLS ; F_1 ; F_2 ; ... ; F_16 ] + E_pos ∈ ℝ^(17 × 256)</b><br>"
            "<font style='font-size: 10px; color: #047857;'>Preserves exact temporal sequence ordering without temporal collapse.</font>"
            "</td>"
            "<td width='35%' style='vertical-align: top; border-right: 1px dashed #6EE7B7; padding: 0 10px;'>"
            "<b>2. Stacked Transformer Block Architecture:</b><br>"
            "• <b>3 Stacked Encoder Layers:</b> l = 1, 2, 3<br>"
            "• <b>Multi-Head Self-Attention:</b> h = 8 heads, d_k = 32<br>"
            "• <b>Feed-Forward Network:</b> d_ff = 1024 with GELU activation<br>"
            "• <b>LayerNorm &amp; Residuals:</b> Z'_l = MSA(LN(Z_(l-1))) + Z_(l-1)<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Z_l = FFN(LN(Z'_l)) + Z'_l"
            "</td>"
            "<td width='25%' style='vertical-align: top; padding-left: 10px;'>"
            "<b>3. Output Latents:</b><br>"
            "• Sequence Tokens: <b>Z_final ∈ ℝ^(17 × 256)</b><br>"
            "• Captures long-range athletic dependencies across start, apex, and follow-through.<br>"
            "• Dropout: p = 0.1 for attention &amp; FFN."
            "</td>"
            "</tr>"
            "</table>"
        )
        self.add_node("box_trans", trans_val, "rounded=1;arcSize=8;fillColor=#ECFDF5;strokeColor=#059669;strokeWidth=2;html=1;align=left;spacing=10;shadow=1;", 140, 1175, 1420, 125)

        self.add_edge("e_gmfu_trans", "box_gmfu", "box_trans", style_edge_green, "Fused Tokens F ∈ ℝ^(16 × 256)", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

        # Substage 3E: Multi-Scale 4-Way Temporal Pooling Head
        pool_head_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #155E75;'>"
            "📊 Multi-Scale 4-Way Temporal Pooling Mechanism (Comprehensive Action Dynamics)"
            "</div>"
        )
        self.add_node("box_pool_head", pool_head_val, style_card_center, 140, 1325, 1420, 32)

        pool_components = [
            ("p_cls", "<b>1. [z_CLS] Token</b>", "z_cls = Z_final[0] ∈ ℝ^256<br><font style='font-size: 10px; color: #475569;'>Learned global contextual action summary across entire clip.</font>", "#E0F2FE", "#0284C7"),
            ("p_attn", "<b>2. Temporal Attention Pooling</b>", "z_Attn = ∑_(t=1)^16 α_t · Z_final[t] ∈ ℝ^256<br><font style='font-size: 10px; color: #475569;'>Learned saliency weights α_t focus on peak impact moments.</font>", "#CCFBF1", "#0D9488"),
            ("p_max", "<b>3. Global Max Pooling</b>", "z_Max = max_(t=1..16) (Z_final[t]) ∈ ℝ^256<br><font style='font-size: 10px; color: #475569;'>Captures extreme instantaneous athletic spikes (e.g. smash, jump).</font>", "#E0E7FF", "#4338CA"),
            ("p_avg", "<b>4. Global Average Pooling</b>", "z_Avg = (1/16) ∑_(t=1)^16 Z_final[t] ∈ ℝ^256<br><font style='font-size: 10px; color: #475569;'>Captures sustained background rhythm and full-motion baseline.</font>", "#EDE9FE", "#6D28D9")
        ]
        pool_x = 140
        for pid, ptitle, pdesc, pfill, pstroke in pool_components:
            pval = f"<div style='text-align: center; color: {pstroke}; font-size: 11.5px;'>{ptitle}</div><div style='text-align: center; font-size: 10px; margin-top: 3px;'>{pdesc}</div>"
            pstyle = f"rounded=1;arcSize=6;fillColor={pfill};strokeColor={pstroke};strokeWidth=1.5;html=1;align=center;verticalAlign=middle;padding=6;"
            self.add_node(pid, pval, pstyle, pool_x, 1365, 340, 75)
            pool_x += 360

        concat_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 12px; color: #0E7490;'>"
            "🔗 Concatenation: Z_pool = [ z_CLS ; z_Attn ; z_Max ; z_Avg ] ∈ ℝ^1024  (Unified Rich Multi-Scale Sports Action Descriptor)"
            "</div>"
        )
        self.add_node("box_concat", concat_val, "rounded=1;arcSize=6;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=2;html=1;align=center;verticalAlign=middle;", 140, 1450, 1420, 38)

        # Edges from Transformer to Pooling and into Concat
        self.add_edge("e_trans_pool", "box_trans", "box_pool_head", style_edge_teal, "Transformer Latents Z_final ∈ ℝ^(17 × 256)", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        for pid in ["p_cls", "p_attn", "p_max", "p_avg"]:
            self.add_edge(f"e_{pid}_concat", pid, "box_concat", style_edge_teal, "", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

        # =========================================================================
        # PHASE 4: OPTIMIZATION, LOSS FORMULATION & RIGOROUS EVALUATION
        # =========================================================================
        p4_val = "<b>PHASE 4: OPTIMIZATION, LOSS FORMULATION &amp; RIGOROUS MODEL EVALUATION</b>"
        self.add_node("container_p4", p4_val, style_container_p4, 60, 1540, 1580, 420)

        # Box 4.1: MLP Classifier & Training Regimen
        mlp_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #991B1B;'>"
            "🎯 3-Layer MLP Classifier &amp; Optimization"
            "</div><hr style='border: 0; border-top: 1px solid #FCA5A5; margin: 5px 0;'>"
            "• <b>Classifier Architecture:</b><br>"
            "&nbsp;&nbsp;- Layer 1: Linear(1024 ──► 512) + LayerNorm + GELU + Dropout(0.3)<br>"
            "&nbsp;&nbsp;- Layer 2: Linear(512 ──► 256) + LayerNorm + GELU + Dropout(0.2)<br>"
            "&nbsp;&nbsp;- Layer 3: Linear(256 ──► 82) ──► Logits ŷ ∈ ℝ^82<br>"
            "• <b>Optimizer:</b> AdamW (Learning Rate = 1e-4, Weight Decay = 1e-2)<br>"
            "• <b>Learning Rate Scheduler:</b><br>"
            "&nbsp;&nbsp;- CosineAnnealingLR (T_max = 30, η_min = 1e-6)<br>"
            "&nbsp;&nbsp;- Linear Warmup for first 5 epochs (stability)<br>"
            "• <b>Data Augmentation Strategy:</b><br>"
            "&nbsp;&nbsp;- Multimodal Mixup: x̃ = λx_i + (1-λ)x_j, λ ~ Beta(0.4, 0.4)<br>"
            "&nbsp;&nbsp;- Kinematic Jitter: N(0, 0.008^2) Gaussian perturbation on joints<br>"
            "&nbsp;&nbsp;- Temporal Frame Dropout: Random frame zeroing (p = 0.1)"
        )
        self.add_node("box_mlp", mlp_val, style_card_white, 80, 1585, 470, 350)

        # Box 4.2: Loss Function Formulation
        loss_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #991B1B;'>"
            "⚖️ Focal Loss with Label Smoothing Regularization"
            "</div><hr style='border: 0; border-top: 1px solid #FCA5A5; margin: 5px 0;'>"
            "<div style='text-align: center; background: #FEF2F2; border: 1.5px solid #EF4444; border-radius: 6px; padding: 6px; margin: 6px 0;'>"
            "<b style='font-size: 12px; color: #B91C1C;'>L_total = - ∑_(c=1)^82 α_c · (1 - p_c)^γ · y_c^(smooth) · log(p_c)</b>"
            "</div>"
            "• <b>Focusing Hyperparameter (γ = 1.5):</b><br>"
            "&nbsp;&nbsp;Down-weights easy, well-classified actions (e.g. Walking, Running), "
            "forcing gradient updates to focus on hard, fine-grained sports (e.g. TennisSwing vs. BadmintonSmash).<br>"
            "• <b>Class Weighting (α_c = 0.08):</b><br>"
            "&nbsp;&nbsp;Compensates for class frequency variances across 82 categories.<br>"
            "• <b>Label Smoothing (ε = 0.08):</b><br>"
            "&nbsp;&nbsp;<span style='color: #7F1D1D;'>y_c^(smooth) = (1 - ε) · y_c + (ε / 82)</span><br>"
            "&nbsp;&nbsp;Prevents softmax overconfidence on visually ambiguous actions.<br>"
            "• <b>Gradient Backpropagation:</b><br>"
            "&nbsp;&nbsp;Drives end-to-end parameter updates through Transformer &amp; Stems."
        )
        self.add_node("box_loss", loss_val, style_card_white, 580, 1585, 480, 350)

        # Box 4.3: Model Evaluation & Peak Performance Milestone (Reference 4 & 7 style)
        eval_val = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #15803D;'>"
            "🏆 Generalization Evaluation &amp; Peak Milestone"
            "</div><hr style='border: 0; border-top: 1px solid #86EFAC; margin: 5px 0;'>"
            "• <b>Evaluation Set:</b> 2,743 Disjoint Test Clips (Zero Leakage)<br>"
            "<div style='background: #DCFCE7; border: 1.5px solid #16A34A; border-radius: 6px; padding: 8px; margin: 8px 0; text-align: center;'>"
            "<span style='font-size: 13px; font-weight: bold; color: #14532D;'>PEAK VALIDATION MILESTONE (EPOCH 23)</span><br>"
            "<table width='100%' style='margin-top: 4px; font-size: 11.5px; font-weight: bold;'>"
            "<tr>"
            "<td style='color: #15803D;'>Top-1 Accuracy:</td><td style='color: #15803D; font-size: 13px;'>90.19%</td>"
            "<td style='color: #15803D;'>Top-5 Accuracy:</td><td style='color: #15803D; font-size: 13px;'>99.02%</td>"
            "</tr>"
            "<tr>"
            "<td style='color: #15803D;'>Validation Loss:</td><td style='color: #15803D;'>0.4397</td>"
            "<td style='color: #15803D;'>Macro F1-Score:</td><td style='color: #15803D;'>90.19%</td>"
            "</tr>"
            "</table>"
            "</div>"
            "• <b>Diagnostic Tools &amp; Verification:</b><br>"
            "&nbsp;&nbsp;- 82×82 Normalized Confusion Matrix (low inter-class confusion)<br>"
            "&nbsp;&nbsp;- Saliency Attention Weight Heatmaps (sharp temporal localization)<br>"
            "&nbsp;&nbsp;- Early Stopping monitor with patience = 7 epochs<br>"
            "• <b>Artifact Checkpoint:</b> Serialized <span style='font-family: monospace; font-weight: bold; color: #15803D;'>best_model.pt</span> "
            "with full state dict and optimizer states for real-time deployment."
        )
        self.add_node("box_eval", eval_val, style_card_white, 1090, 1585, 520, 350)

        # Connecting Edges in Phase 4
        # Concat down to MLP
        self.add_edge("e_concat_mlp", "box_concat", "box_mlp", style_edge_red, "Z_pool (1024-d)", exit_x=0.2, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        # MLP to Loss
        self.add_edge("e_mlp_loss", "box_mlp", "box_loss", style_edge_red, "Logits ŷ ∈ ℝ^82", exit_x=1.0, exit_y=0.4, entry_x=0.0, entry_y=0.4)
        # Loss backprop arrow back to Transformer (Backprop loop)
        self.add_edge("e_loss_backprop", "box_loss", "box_trans", style_edge_red, "∇ Loss (Backpropagation)", exit_x=0.5, exit_y=0.0, entry_x=0.7, entry_y=1.0)
        # Test Split to Model Evaluation
        self.add_edge("e_test_eval", "box_split", "box_eval", style_edge_green, "30% Test Set (2,743 clips)", exit_x=0.8, exit_y=1.0, entry_x=0.8, entry_y=0.0)
        # MLP to Model Evaluation
        self.add_edge("e_mlp_eval", "box_mlp", "box_eval", style_edge_green, "Epoch Checkpoints", exit_x=1.0, exit_y=0.75, entry_x=0.0, entry_y=0.75)

        # =========================================================================
        # PHASE 5: REAL-TIME INFERENCE PIPELINE & MULTI-PLATFORM DEPLOYMENT
        # =========================================================================
        p5_val = "<b>PHASE 5: REAL-TIME INFERENCE PIPELINE &amp; MULTI-PLATFORM DEPLOYMENT</b>"
        self.add_node("container_p5", p5_val, style_container_p5, 60, 1980, 1580, 370)

        # Box 5.1: Live Video Ingestion & Circular Buffer
        box_live = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #115E59;'>"
            "📹 Live Stream &amp; Circular Queue"
            "</div><hr style='border: 0; border-top: 1px solid #99F6E4; margin: 5px 0;'>"
            "• <b>Real-Time Video Sources:</b><br>"
            "&nbsp;&nbsp;- 30+ FPS Live USB/CSI Webcam feed<br>"
            "&nbsp;&nbsp;- Unseen match broadcast (.mp4 / .avi)<br>"
            "&nbsp;&nbsp;- Synthetic edge simulation test stream<br>"
            "• <b>Circular Sliding Window Buffer:</b><br>"
            "&nbsp;&nbsp;- Queue Length: T = 16 frames<br>"
            "&nbsp;&nbsp;- Rolling stride = 1 frame (continuous prediction)<br>"
            "&nbsp;&nbsp;- Thread-safe asynchronous video capture<br>"
            "&nbsp;&nbsp;- Zero frame dropping with hardware decoders"
        )
        self.add_node("box_live", box_live, style_card_white, 80, 2025, 430, 305)

        # Box 5.2: ActionInferenceEngine (Runtime)
        box_engine = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #115E59;'>"
            "⚡ ActionInferenceEngine (Production Runtime)"
            "</div><hr style='border: 0; border-top: 1px solid #99F6E4; margin: 5px 0;'>"
            "• <b>Core Script:</b> <span style='font-family: monospace; color: #0F766E; font-weight: bold;'>src/inference/engine.py</span><br>"
            "• <b>Pre-Trained Weights:</b> Loaded from <span style='font-family: monospace; color: #0F766E; font-weight: bold;'>best_model.pt</span> (Epoch 23)<br>"
            "• <b>Synchronized Runtime Feature Extraction:</b><br>"
            "&nbsp;&nbsp;- Fast ResNet-50 appearance forward pass<br>"
            "&nbsp;&nbsp;- Keypoint R-CNN skeleton tracker + velocity cache<br>"
            "&nbsp;&nbsp;- Apparatus candidate scoring &amp; kinematic engine<br>"
            "• <b>Inference Latency Profile:</b><br>"
            "&nbsp;&nbsp;- Forward Pass Execution: <b>~18.4 ms / window</b><br>"
            "&nbsp;&nbsp;- Full End-to-End Latency: <b>&lt; 33.3 ms (True 30+ FPS Real-Time)</b><br>"
            "• <b>Temporal Attention Saliency Extraction:</b><br>"
            "&nbsp;&nbsp;Extracts real-time frame importance weights α_t for live HUD display."
        )
        self.add_node("box_engine", box_engine, style_card_white, 540, 2025, 500, 305)

        # Box 5.3: Deployment Platforms & Prediction HUD
        box_ui = (
            "<div style='text-align: center; font-weight: bold; font-size: 13px; color: #115E59;'>"
            "🖥️ Multi-Platform Deployment &amp; Prediction HUD"
            "</div><hr style='border: 0; border-top: 1px solid #99F6E4; margin: 5px 0;'>"
            "• <b>Classification Outputs (82 Sports Classes):</b><br>"
            "&nbsp;&nbsp;- <span style='color: #0F766E; font-weight: bold;'>Top-1 Predicted Class</span> (e.g., <i>TennisSwing 94.6%</i>)<br>"
            "&nbsp;&nbsp;- <span style='color: #0F766E; font-weight: bold;'>Top-5 Confidence Distribution</span> bar chart<br>"
            "&nbsp;&nbsp;- Dynamic Temporal Attention Saliency timeline<br>"
            "• <b>Deployed Platforms &amp; Interfaces:</b><br>"
            "&nbsp;&nbsp;1. <b>Streamlit Web Dashboard:</b> (<span style='font-family: monospace;'>app.py</span>)<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Interactive video upload, metric analytics, and ablation explorer.<br>"
            "&nbsp;&nbsp;2. <b>Desktop Native GUI:</b> (<span style='font-family: monospace;'>src/ui/desktop_app.py</span>)<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Low-latency Tkinter multi-threaded application.<br>"
            "&nbsp;&nbsp;3. <b>Live Video Stream HUD:</b> (<span style='font-family: monospace;'>src/ui/video_stream.py</span>)<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Real-time OpenCV webcam overlay with joint skeletons."
        )
        self.add_node("box_ui", box_ui, style_card_white, 1070, 2025, 540, 305)

        # Edges in Phase 5
        # Best model checkpoint from Phase 4 to ActionInferenceEngine
        self.add_edge("e_best_model_deploy", "box_eval", "box_engine", style_edge_teal, "Serialized best_model.pt", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        # Live video to Engine
        self.add_edge("e_live_engine", "box_live", "box_engine", style_edge_teal, "16-Frame FIFO Buffer", exit_x=1.0, exit_y=0.5, entry_x=0.0, entry_y=0.5)
        # Engine to UI & Predictions
        self.add_edge("e_engine_ui", "box_engine", "box_ui", style_edge_teal, "Top-5 Probabilities & Saliency", exit_x=1.0, exit_y=0.5, entry_x=0.0, entry_y=0.5)

        # =========================================================================
        # CAPTION AT BOTTOM
        # =========================================================================
        caption_val = (
            "<b>Figure 1:</b> Comprehensive Methodology Architecture of the Proposed CMAT-HOI Framework for Fine-Grained Sports Action Recognition. "
            "The pipeline systematically spans dataset ingestion &amp; conditioning (Phase 1), synchronized 4-stream multimodal extraction (Phase 2), "
            "the core Cross-Modal Attention Transformer network (Phase 3), optimization with Focal Loss &amp; disjoint evaluation (Phase 4), "
            "and real-time 30+ FPS multi-platform deployment (Phase 5)."
        )
        self.add_node("caption_fig1", caption_val, "text;html=1;align=center;verticalAlign=middle;fontSize=12;fontColor=#475569;fontStyle=2;", 60, 2370, 1580, 50)

    def save(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="  ", level=0)
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
        print(f"Successfully generated: {path.resolve()}")


if __name__ == "__main__":
    generator = DrawioGenerator()
    generator.build_diagram()

    # Save to both target locations
    target_1 = Path("Paper writing/CMAT_HOI_Methodology_Architecture.drawio")
    target_2 = Path("diagrams/M7/CMAT_HOI_Methodology_Architecture.drawio")

    generator.save(target_1)
    generator.save(target_2)
