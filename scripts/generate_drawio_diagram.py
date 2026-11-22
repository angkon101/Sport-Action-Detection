#!/usr/bin/env python3
"""
Generate a clean, simplified, easy-to-understand Draw.io architectural diagram (.drawio XML)
for the CMAT-HOI Sports Action Recognition thesis project.

Directly inspired by Reference 7 & Reference 4:
  - Clear, un-cluttered stages with readable font sizes.
  - Zero intimidating mathematical formulas or tiny dense tables.
  - Plain, intuitive explanations that anyone can understand in 5 seconds:
      1. Dataset & Preprocessing Funnel (16 frames, resize, normalize)
      2. 4-Stream Feature Extraction (Visual, Pose, Equipment, Interaction)
      3. CMAT-HOI Core Model (Stems -> Cross-Attention -> Gated Fusion -> Transformer)
      4. Training & Evaluation (Focal Loss <-> 90.19% Accuracy)
      5. Deployment / Real-Time Prediction (Live Video -> Trained Model -> Action Output)
"""

import xml.etree.ElementTree as ET
from pathlib import Path


class SimpleDrawioGenerator:
    def __init__(self, width=1400, height=1950):
        self.root = ET.Element("mxfile", host="app.diagrams.net", version="24.0.0", type="device")
        self.diagram = ET.SubElement(self.root, "diagram", id="CMAT_HOI_Simplified", name="Methodology Architecture")
        self.graph = ET.SubElement(
            self.diagram,
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
            pageWidth=str(width),
            pageHeight=str(height),
            math="0",
            shadow="0"
        )
        self.root_cells = ET.SubElement(self.graph, "root")
        ET.SubElement(self.root_cells, "mxCell", id="0")
        ET.SubElement(self.root_cells, "mxCell", id="1", parent="0")

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

    def add_edge(self, edge_id, source_id, target_id, style, value="", parent="1", exit_x=None, exit_y=None, entry_x=None, entry_y=None):
        edge_style = style
        if exit_x is not None and exit_y is not None:
            edge_style += f";exitX={exit_x};exitY={exit_y};exitDx=0;exitDy=0"
        if entry_x is not None and entry_y is not None:
            edge_style += f";entryX={entry_x};entryY={entry_y};entryDx=0;entryDy=0"

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
        ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        return cell

    def build_diagram(self):
        # Clean Styles
        style_banner = (
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#1E293B;strokeColor=#0F172A;"
            "strokeWidth=2;shadow=1;align=center;verticalAlign=middle;"
        )
        style_container_p1 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F0F9FF;strokeColor=#0284C7;"
            "strokeWidth=2;dashed=1;verticalAlign=top;spacingTop=8;fontStyle=1;fontSize=13;"
            "fontColor=#0369A1;container=1;collapsible=0;"
        )
        style_container_p2 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#9333EA;"
            "strokeWidth=2;dashed=1;verticalAlign=top;spacingTop=8;fontStyle=1;fontSize=13;"
            "fontColor=#6B21A8;container=1;collapsible=0;"
        )
        style_container_p3 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FFFDF5;strokeColor=#D97706;"
            "strokeWidth=2;verticalAlign=top;spacingTop=8;fontStyle=1;fontSize=14;"
            "fontColor=#92400E;container=1;collapsible=0;"
        )
        style_container_p4 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#EF4444;"
            "strokeWidth=2;dashed=1;verticalAlign=top;spacingTop=8;fontStyle=1;fontSize=13;"
            "fontColor=#B91C1C;container=1;collapsible=0;"
        )
        style_container_p5 = (
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ECFDF5;strokeColor=#059669;"
            "strokeWidth=2;verticalAlign=top;spacingTop=8;fontStyle=1;fontSize=13;"
            "fontColor=#047857;container=1;collapsible=0;"
        )

        style_card = (
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;"
            "strokeWidth=1.5;shadow=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#1E293B;"
        )
        style_card_left = (
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;"
            "strokeWidth=1.5;shadow=1;align=left;verticalAlign=middle;spacingLeft=10;fontSize=11;fontColor=#1E293B;"
        )

        style_arrow_blue = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0284C7;strokeWidth=2;endArrow=block;endFill=1;fontSize=11;fontColor=#0369A1;"
        style_arrow_purple = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#9333EA;strokeWidth=2;endArrow=block;endFill=1;fontSize=11;fontColor=#6B21A8;"
        style_arrow_amber = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#D97706;strokeWidth=2;endArrow=block;endFill=1;fontSize=11;fontColor=#92400E;"
        style_arrow_red = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#EF4444;strokeWidth=2;endArrow=block;endFill=1;fontSize=11;fontColor=#B91C1C;"
        style_arrow_green = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#059669;strokeWidth=2;endArrow=block;endFill=1;fontSize=11;fontColor=#047857;"

        # =========================================================================
        # 0. TITLE BANNER
        # =========================================================================
        banner_val = (
            "<b style='font-size: 19px; color: #FFFFFF;'>CMAT-HOI: System Methodology &amp; Architecture Diagram</b><br>"
            "<span style='font-size: 12px; color: #94A3B8;'>End-to-End Multimodal Framework for Sports Action Recognition (82 Classes)</span>"
        )
        self.add_node("banner", banner_val, style_banner, 40, 20, 1320, 65)

        # =========================================================================
        # 1. STEP 1: DATA INGESTION & PREPROCESSING (Funnel Style like Ref 7)
        # =========================================================================
        self.add_node("cont_p1", "<b>STEP 1: DATA COLLECTION, PREPROCESSING FUNNEL &amp; SPLITTING</b>", style_container_p1, 40, 100, 1320, 240)

        # 1A: Dataset
        ds_val = (
            "<b style='font-size: 13px; color: #0284C7;'>📁 Sports Action Dataset</b><br>"
            "<hr style='border:0; border-top:1px solid #BAE6FD; margin:4px 0;'>"
            "• <b>UCF-101 Subset:</b> 9,151 Videos<br>"
            "• <b>82 Action Categories:</b><br>"
            "&nbsp;&nbsp;Basketball, Tennis, Cricket,<br>"
            "&nbsp;&nbsp;Gymnastics, Diving, Boxing...<br>"
            "• <b>Input:</b> Raw RGB Video Clips"
        )
        self.add_node("box_dataset", ds_val, style_card_left, 60, 135, 250, 185)

        # 1B: Funnel (Simple, high-level, clear)
        funnel_val = (
            "<b style='font-size: 13px; color: #0284C7;'>⚙️ 5-Step Video Preprocessing Funnel</b><br>"
            "<hr style='border:0; border-top:1px solid #BAE6FD; margin:4px 0;'>"
            "<table width='100%' style='font-size: 11px; text-align: left;'>"
            "<tr><td style='padding: 2px 0;'><b>1. Frame Sampling:</b> Select 16 equidistant frames across each video</td></tr>"
            "<tr><td style='padding: 2px 0;'><b>2. Image Resizing:</b> Resize frames to standard 224 × 224 pixels</td></tr>"
            "<tr><td style='padding: 2px 0;'><b>3. Motion Velocities:</b> Compute speed and direction between consecutive frames</td></tr>"
            "<tr><td style='padding: 2px 0;'><b>4. Data Cleaning:</b> Handle missing player joints and equipment bounding boxes</td></tr>"
            "<tr><td style='padding: 2px 0;'><b>5. Normalization:</b> Zero-center normalize pixels &amp; package for model</td></tr>"
            "</table>"
        )
        self.add_node("box_funnel", funnel_val, style_card_left, 340, 135, 590, 185)

        # 1C: Train / Test Split
        split_val = (
            "<b style='font-size: 13px; color: #0284C7;'>⚖️ Dataset Split</b><br>"
            "<hr style='border:0; border-top:1px solid #BAE6FD; margin:4px 0;'>"
            "<div style='background:#EFF6FF; border:1px solid #3B82F6; border-radius:5px; padding:4px; margin-bottom:6px;'>"
            "<b>🏋️ Training Set (70%)</b><br>"
            "<span style='color:#1D4ED8; font-size:12px; font-weight:bold;'>6,408 Video Clips</span><br>"
            "<font style='font-size:9.5px; color:#64748B;'>Used to train the neural network</font>"
            "</div>"
            "<div style='background:#F0FDF4; border:1px solid #16A34A; border-radius:5px; padding:4px;'>"
            "<b>🎯 Testing Set (30%)</b><br>"
            "<span style='color:#15803D; font-size:12px; font-weight:bold;'>2,743 Video Clips</span><br>"
            "<font style='font-size:9.5px; color:#64748B;'>Unseen data to verify accuracy</font>"
            "</div>"
        )
        self.add_node("box_split", split_val, style_card, 960, 135, 380, 185)

        # Arrows for Step 1
        self.add_edge("e1_1", "box_dataset", "box_funnel", style_arrow_blue, "Raw Videos")
        self.add_edge("e1_2", "box_funnel", "box_split", style_arrow_blue, "Preprocessed")

        # =========================================================================
        # 2. STEP 2: 4-STREAM FEATURE EXTRACTION (Simple Parallel Cards)
        # =========================================================================
        self.add_node("cont_p2", "<b>STEP 2: 4-STREAM MULTIMODAL FEATURE EXTRACTION (WHAT THE MODEL SEES)</b>", style_container_p2, 40, 360, 1320, 180)

        streams = [
            ("s_vis", "1. Visual Appearance", "ResNet-50", "Background scene, sports arena,<br>court lines, player clothing", "#EFF6FF", "#2563EB", 60),
            ("s_pose", "2. Player Body Pose", "Keypoint R-CNN", "17 human body joints (arms, legs)<br>+ Movement speed &amp; angles", "#ECFDF5", "#059669", 390),
            ("s_obj", "3. Sports Equipment", "Faster R-CNN", "Ball, tennis racket, bat,<br>skateboard, barbell detection", "#FFFBEB", "#D97706", 720),
            ("s_hoi", "4. Human-Object Action", "Interaction Engine", "Hands/feet distance to equipment<br>+ Ball hit/release timing", "#FDF2F8", "#DB2777", 1050)
        ]
        for sid, stitle, smodel, sdesc, sfill, sborder, sx in streams:
            sval = (
                f"<b style='font-size: 12.5px; color: {sborder};'>{stitle}</b><br>"
                f"<span style='font-size: 11px; font-weight: bold; color: #1E293B;'>Model: {smodel}</span><hr style='border:0; border-top:1px solid #E2E8F0; margin:4px 0;'>"
                f"<span style='font-size: 10px; color: #475569;'>{sdesc}</span>"
            )
            self.add_node(sid, sval, f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={sfill};strokeColor={sborder};strokeWidth=1.5;shadow=1;align=center;verticalAlign=middle;padding=6;", sx, 395, 300, 125)

        # Arrow down from Train Split to Step 2
        self.add_edge("e_train_extract", "box_split", "s_obj", style_arrow_purple, "Extract Features from Training Videos", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)

        # =========================================================================
        # 3. STEP 3: CMAT-HOI CORE MODEL (Clean intuitive pipeline)
        # =========================================================================
        self.add_node("cont_p3", "<b>STEP 3: CMAT-HOI CORE NETWORK ARCHITECTURE</b>", style_container_p3, 40, 560, 1320, 500)

        # 3A: Dimension Alignment
        stem_val = (
            "<b style='font-size: 12.5px; color: #92400E;'>1. Modality Alignment (Stems)</b><br>"
            "<span style='font-size: 10.5px; color: #451A03;'>Transforms all 4 streams (images, body joints, equipment, interactions) into a unified matching size (256 dimensions)</span>"
        )
        self.add_node("box_stems", stem_val, "rounded=1;arcSize=6;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 60, 600, 1280, 55)

        # 3B: Cross-Attention
        cross_val = (
            "<b style='font-size: 12.5px; color: #831843;'>2. Cross-Modal Attention (Body Pose ↔ Sports Equipment)</b><br>"
            "<span style='font-size: 10.5px; color: #4C0519;'>Connects human movements directly with sports equipment: "
            "Helps the model know exactly <i>where</i> the ball/racket is when the player swings or jumps.</span>"
        )
        self.add_node("box_cross", cross_val, "rounded=1;arcSize=6;fillColor=#FDF2F8;strokeColor=#BE185D;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 60, 680, 1280, 55)

        # 3C: Gated Multimodal Fusion
        gmfu_val = (
            "<b style='font-size: 12.5px; color: #78350F;'>3. Smart Gated Fusion (Adaptive Modality Trust)</b><br>"
            "<span style='font-size: 10.5px; color: #451A03;'>Automatically learns which stream is most trustworthy in each video frame:<br>"
            "• If body is occluded, rely more on equipment and scene &nbsp;|&nbsp; "
            "• At the moment of ball contact, give top priority to hand-object interaction!</span>"
        )
        self.add_node("box_gmfu", gmfu_val, "rounded=1;arcSize=6;fillColor=#FEF9C3;strokeColor=#CA8A04;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 60, 760, 1280, 65)

        # 3D: Transformer Encoder
        trans_val = (
            "<b style='font-size: 13px; color: #064E3B;'>4. Spatio-Temporal Transformer (Learns Video Sequence Patterns)</b><br>"
            "<span style='font-size: 10.5px; color: #022C22;'>Reads all 16 frames in chronological sequence using Self-Attention:<br>"
            "Understands the full athletic storyline from preparation $\\to$ peak strike/jump $\\to$ follow-through.</span>"
        )
        self.add_node("box_trans", trans_val, "rounded=1;arcSize=6;fillColor=#ECFDF5;strokeColor=#059669;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 60, 850, 1280, 65)

        # 3E: Multi-Scale Pooling
        pool_val = (
            "<b style='font-size: 12.5px; color: #0E7490;'>5. Multi-Scale Temporal Pooling</b><br>"
            "<span style='font-size: 10.5px; color: #164E63;'>Combines 4 perspectives into one complete action signature: "
            "<b>Global Context</b> + <b>Peak Fast Spike</b> + <b>Average Flow</b> + <b>Key Moments</b></span>"
        )
        self.add_node("box_pool", pool_val, "rounded=1;arcSize=6;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;html=1;align=center;verticalAlign=middle;", 60, 940, 1280, 55)

        # Arrows inside Step 3
        self.add_edge("e_s_stems", "s_pose", "box_stems", style_arrow_amber, "", exit_x=0.5, exit_y=1.0, entry_x=0.25, entry_y=0.0)
        self.add_edge("e_s_stems2", "s_obj", "box_stems", style_arrow_amber, "", exit_x=0.5, exit_y=1.0, entry_x=0.75, entry_y=0.0)
        self.add_edge("e_stem_cross", "box_stems", "box_cross", style_arrow_amber, "")
        self.add_edge("e_cross_gmfu", "box_cross", "box_gmfu", style_arrow_amber, "")
        self.add_edge("e_gmfu_trans", "box_gmfu", "box_trans", style_arrow_amber, "")
        self.add_edge("e_trans_pool", "box_trans", "box_pool", style_arrow_amber, "")

        # =========================================================================
        # 4. STEP 4: MODEL TRAINING & EVALUATION LOOP (Like Ref 4 & 7)
        # =========================================================================
        self.add_node("cont_p4", "<b>STEP 4: TRAINING OPTIMIZATION &amp; INDEPENDENT EVALUATION</b>", style_container_p4, 40, 1080, 1320, 240)

        # 4A: Training
        train_val = (
            "<b style='font-size: 13px; color: #B91C1C;'>🎯 Model Training &amp; Loss</b><br>"
            "<hr style='border:0; border-top:1px solid #FECACA; margin:4px 0;'>"
            "• <b>Classifier:</b> 3-Layer Neural Network<br>"
            "• <b>Loss:</b> Focal Loss with Label Smoothing<br>"
            "&nbsp;&nbsp;(Prevents confusion between similar sports)<br>"
            "• <b>Optimizer:</b> AdamW with Cosine Learning Rate<br>"
            "• <b>Augmentations:</b> Mixup &amp; Joint Jitter"
        )
        self.add_node("box_train_opt", train_val, style_card_left, 60, 1115, 420, 185)

        # 4B: Center connection
        center_train_val = (
            "<b style='font-size: 14px; color: #1E293B;'>CMAT-HOI Training Loop</b><br>"
            "<hr style='border:0; border-top:1px solid #CBD5E1; margin:6px 0;'>"
            "<span style='font-size: 11px; color: #475569;'>The model adjusts millions of parameters epoch-by-epoch until it reaches peak recognition accuracy.</span><br><br>"
            "<span style='background:#DCFCE7; border:1px solid #16A34A; color:#14532D; font-weight:bold; padding:4px 8px; border-radius:4px; font-size:11px;'>"
            "Best Model Checkpoint: Epoch 23 (Saved to best_model.pt)"
            "</span>"
        )
        self.add_node("box_center_loop", center_train_val, style_card, 510, 1115, 380, 185)

        # 4C: Evaluation / Results
        eval_val = (
            "<b style='font-size: 13px; color: #15803D;'>🏆 Test Evaluation (Unseen Data)</b><br>"
            "<hr style='border:0; border-top:1px solid #BBF7D0; margin:4px 0;'>"
            "• <b>Evaluated on:</b> 2,743 Independent Test Clips<br>"
            "<div style='background:#ECFDF5; border:1.5px solid #059669; border-radius:5px; padding:6px; margin:6px 0; text-align:center;'>"
            "<b style='color:#065F46; font-size:14px;'>Top-1 Accuracy: 90.19%</b><br>"
            "<b style='color:#065F46; font-size:12px;'>Top-5 Accuracy: 99.02%</b>"
            "</div>"
            "• <b>Validation Loss:</b> 0.4397 (Lowest error)<br>"
            "• <b>Zero Data Leakage:</b> Strictly separated actors"
        )
        self.add_node("box_eval_res", eval_val, style_card_left, 920, 1115, 420, 185)

        # Arrows in Step 4
        self.add_edge("e_pool_train", "box_pool", "box_center_loop", style_arrow_red, "Unified Feature Vector", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        self.add_edge("e_train_loss", "box_train_opt", "box_center_loop", style_arrow_red, "Gradient Updates", exit_x=1.0, exit_y=0.5, entry_x=0.0, entry_y=0.5)
        self.add_edge("e_eval_test", "box_center_loop", "box_eval_res", style_arrow_green, "Validate on Test Data", exit_x=1.0, exit_y=0.5, entry_x=0.0, entry_y=0.5)

        # =========================================================================
        # 5. STEP 5: REAL-TIME DEPLOYMENT & ACTION PREDICTION (Bottom Box like Ref 7)
        # =========================================================================
        self.add_node("cont_p5", "<b>STEP 5: REAL-TIME DEPLOYMENT &amp; PREDICTION INTERFACES</b>", style_container_p5, 40, 1340, 1320, 220)

        # 5A: Input Source
        live_val = (
            "<b style='font-size: 13px; color: #047857;'>📹 Live Video / Camera Input</b><br>"
            "<hr style='border:0; border-top:1px solid #A7F3D0; margin:4px 0;'>"
            "• Live 30+ FPS USB / Web Camera Feed<br>"
            "• Or Unseen Match Video (.mp4 / .avi)<br>"
            "• Rolling 16-frame sliding window buffer"
        )
        self.add_node("box_live", live_val, style_card_left, 60, 1375, 360, 165)

        # 5B: Trained Engine
        engine_val = (
            "<b style='font-size: 13px; color: #047857;'>⚡ Trained CMAT-HOI Engine</b><br>"
            "<hr style='border:0; border-top:1px solid #A7F3D0; margin:4px 0;'>"
            "• Loads saved weights (<b>best_model.pt</b>)<br>"
            "• <b>Ultra-Fast Speed:</b> ~18.4 milliseconds per clip<br>"
            "• Delivers smooth, real-time sports recognition"
        )
        self.add_node("box_engine", engine_val, style_card_left, 470, 1375, 420, 165)

        # 5C: Final Output & Applications
        out_val = (
            "<b style='font-size: 13px; color: #047857;'>🏆 Action Prediction &amp; UI</b><br>"
            "<hr style='border:0; border-top:1px solid #A7F3D0; margin:4px 0;'>"
            "• <b>Output:</b> Top Predicted Action (e.g. <i>TennisSwing 94%</i>)<br>"
            "• <b>Streamlit Web App:</b> Interactive dashboard (<span style='font-family:monospace;'>app.py</span>)<br>"
            "• <b>Desktop GUI:</b> Native live camera app (<span style='font-family:monospace;'>desktop_app.py</span>)"
        )
        self.add_node("box_out", out_val, style_card_left, 940, 1375, 400, 165)

        # Arrows in Step 5
        self.add_edge("e_best_to_engine", "box_center_loop", "box_engine", style_arrow_green, "Deploy Saved Model", exit_x=0.5, exit_y=1.0, entry_x=0.5, entry_y=0.0)
        self.add_edge("e_live_engine", "box_live", "box_engine", style_arrow_green, "Stream Frames")
        self.add_edge("e_engine_out", "box_engine", "box_out", style_arrow_green, "Predictions")

        # =========================================================================
        # CAPTION
        # =========================================================================
        caption_val = (
            "<b>Fig. 1.</b> Simplified System Architecture Diagram: CMAT-HOI End-to-End Sports Action Recognition Framework."
        )
        self.add_node("caption", caption_val, "text;html=1;align=center;verticalAlign=middle;fontSize=12;fontColor=#475569;fontStyle=2;", 40, 1575, 1320, 30)

    def save(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="  ", level=0)
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
        print(f"Successfully generated simplified diagram: {path.resolve()}")


if __name__ == "__main__":
    gen = SimpleDrawioGenerator()
    gen.build_diagram()

    target_1 = Path("Paper writing/CMAT_HOI_Methodology_Architecture.drawio")
    target_2 = Path("diagrams/M7/CMAT_HOI_Methodology_Architecture.drawio")

    gen.save(target_1)
    gen.save(target_2)
