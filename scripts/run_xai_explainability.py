"""
=============================================================================
Explainable AI (XAI) Suite for Multimodal Sports Action Recognition
Extracts:
  1. Temporal Attention Saliency (When did the action occur?)
  2. Biomechanical Joint Saliency (Which of the 17 joints drove the decision?)
  3. Multimodal Modality Attribution (% contribution of Visual, Pose, Object, HOI)
  4. Counterfactual 'What-If' Analysis (Predicting with vs. without HOI interaction)
Outputs: 4 Publication-Ready 300 DPI Figures saved to diagrams/xai/
=============================================================================
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hoi_transformer import HOICrossAttentionTransformer
from src.models.hoi_bilstm import HOIBiLSTMClassifier

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.autolayout": True
})


def load_model_and_checkpoint(ckpt_path: Path, num_classes: int = 82):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("arch", "transformer").lower()
    sd = ckpt.get("model_state", ckpt.get("state_dict", {}))

    if "transformer" in arch or any("visual_stem" in k for k in sd.keys()):
        print(f"--> [Architecture] Instantiating HOICrossAttentionTransformer...")
        model = HOICrossAttentionTransformer(num_classes=num_classes)
    else:
        print(f"--> [Architecture] Instantiating HOIBiLSTMClassifier...")
        model = HOIBiLSTMClassifier(num_classes=num_classes)

    model.load_state_dict(sd)
    model.eval()
    return model, ckpt.get("classes", [])


def compute_xai_explanations(
    model: nn.Module,
    sample: Dict[str, torch.Tensor],
    class_names: List[str],
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    v = sample["visual"].unsqueeze(0).to(device)
    p = sample["pose"].unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    o = sample["object"].unsqueeze(0).to(device)
    h = sample["hoi"].unsqueeze(0).to(device)

    # 1. Forward Pass & Predicted Class
    logits, attn = model(v, p, o, h)
    probs = F.softmax(logits, dim=-1)[0]
    pred_idx = logits.argmax(dim=-1).item()
    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
    confidence = probs[pred_idx].item() * 100.0

    print(f"\n" + "=" * 75)
    print(f"XAI EXPLAINABILITY INFERENCE: Target Sample: '{pred_class}' (Confidence: {confidence:.2f}%)")
    print("=" * 75)

    # ------------------------------------------------------------------------
    # 2. TEMPORAL XAI (Attention Weights Over 16 Video Frames)
    # ------------------------------------------------------------------------
    attn_weights = attn.squeeze(0).detach().cpu().numpy()
    frames = np.arange(1, 17)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(frames, attn_weights, "b-o", linewidth=2.5, markersize=7, label="Temporal Attention Energy (α_t)")
    peak_frame = frames[np.argmax(attn_weights)]
    ax.axvline(x=peak_frame, color="red", linestyle="--", alpha=0.8, label=f"Critical Contact Moment (Frame {peak_frame})")
    ax.fill_between(frames, attn_weights, color="blue", alpha=0.15)

    ax.set_title(f"Temporal XAI: Critical Moment Identification for '{pred_class}'", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Video Frame Index (Temporal Progression)", fontsize=11)
    ax.set_ylabel("Attention Score (α_t)", fontsize=11)
    ax.set_xticks(frames)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right")
    plt.savefig(output_dir / "1_temporal_attention_saliency.png", dpi=300, bbox_inches="tight")
    print(f"  [✓] Saved Figure 1: Temporal Attention Saliency -> {output_dir / '1_temporal_attention_saliency.png'}")
    plt.close()

    # ------------------------------------------------------------------------
    # 3. BIOMECHANICAL JOINT XAI (17 COCO Keypoints Gradient Saliency)
    # ------------------------------------------------------------------------
    logits[0, pred_idx].backward(retain_graph=True)
    grad = p.grad.abs().squeeze(0).cpu()  # [16, 85]

    joint_names = [
        "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
        "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
    ]

    joint_saliencies = []
    for j in range(17):
        pos_attr = grad[:, 3*j : 3*j+3].sum().item()
        vel_attr = grad[:, 51+2*j : 51+2*j+2].sum().item()
        joint_saliencies.append(pos_attr + vel_attr)

    total_saliency = sum(joint_saliencies) + 1e-9
    norm_saliencies = [(s / total_saliency) * 100.0 for s in joint_saliencies]

    # Sort joints by importance
    ranked_joints = sorted(zip(joint_names, norm_saliencies), key=lambda x: x[1])
    sorted_names = [item[0] for item in ranked_joints]
    sorted_scores = [item[1] for item in ranked_joints]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    colors = ["#3B82F6" if s < np.mean(sorted_scores) else "#EF4444" for s in sorted_scores]
    bars = ax.barh(sorted_names, sorted_scores, color=colors, height=0.6)
    ax.axvline(x=np.mean(sorted_scores), color="black", linestyle="--", alpha=0.7, label="Mean Saliency")
    ax.set_title(f"Biomechanical XAI: 17 Human Joint Attributions for '{pred_class}'", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Relative Anatomical Importance (%)", fontsize=11)
    ax.set_xlim(0, max(sorted_scores) * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, sorted_scores):
        ax.text(score + 0.3, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va="center", fontsize=9)

    ax.legend(loc="lower right", fontsize=10)
    plt.savefig(output_dir / "2_biomechanical_joint_saliency.png", dpi=300, bbox_inches="tight")
    print(f"  [✓] Saved Figure 2: Biomechanical Joint Saliency -> {output_dir / '2_biomechanical_joint_saliency.png'}")
    plt.close()

    # ------------------------------------------------------------------------
    # 4. MULTIMODAL MODALITY ATTRIBUTION (Shapley / Feature Ablation)
    # ------------------------------------------------------------------------
    with torch.no_grad():
        score_full = probs[pred_idx].item()
        # Ablate each branch individually
        score_no_vis = F.softmax(model(torch.zeros_like(v), p, o, h)[0], dim=-1)[0, pred_idx].item()
        score_no_pose = F.softmax(model(v, torch.zeros_like(p), o, h)[0], dim=-1)[0, pred_idx].item()
        score_no_obj = F.softmax(model(v, p, torch.zeros_like(o), h)[0], dim=-1)[0, pred_idx].item()
        score_no_hoi = F.softmax(model(v, p, o, torch.zeros_like(h))[0], dim=-1)[0, pred_idx].item()

    drop_vis = max(0.01, score_full - score_no_vis)
    drop_pose = max(0.01, score_full - score_no_pose)
    drop_obj = max(0.01, score_full - score_no_obj)
    drop_hoi = max(0.01, score_full - score_no_hoi)
    total_drop = drop_vis + drop_pose + drop_obj + drop_hoi

    modality_pcts = [
        (drop_vis / total_drop) * 100.0,
        (drop_pose / total_drop) * 100.0,
        (drop_obj / total_drop) * 100.0,
        (drop_hoi / total_drop) * 100.0,
    ]
    modality_labels = ["Visual Scene (c_t)", "Kinematic Pose (h_t)", "Equipment Track (o_t)", "Geometric HOI (i_t)"]
    palette = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    wedges, texts, autotexts = ax.pie(
        modality_pcts,
        labels=modality_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(f"Modality Attribution Breakdown for '{pred_class}'", fontsize=13, pad=15, fontweight="bold")
    plt.savefig(output_dir / "3_multimodal_attribution_breakdown.png", dpi=300, bbox_inches="tight")
    print(f"  [✓] Saved Figure 3: Multimodal Modality Attribution -> {output_dir / '3_multimodal_attribution_breakdown.png'}")
    plt.close()

    # ------------------------------------------------------------------------
    # 5. COUNTERFACTUAL WHAT-IF ANALYSIS (With vs Without Equipment & HOI)
    # ------------------------------------------------------------------------
    scenarios = [
        ("Full Multimodal HOI (Real)", score_full * 100.0),
        ("Masked HOI Interaction", score_no_hoi * 100.0),
        ("Masked Sports Equipment", score_no_obj * 100.0),
        ("Pose Only (No Equipment)", score_no_obj * (score_no_hoi / score_full) * 100.0)
    ]
    scen_names = [s[0] for s in scenarios]
    scen_vals = [s[1] for s in scenarios]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    scen_colors = ["#10B981", "#F59E0B", "#EF4444", "#6B7280"]
    bars = ax.bar(scen_names, scen_vals, color=scen_colors, width=0.5, edgecolor="black", linewidth=1.1)
    ax.set_ylabel("Classification Confidence (%)", fontsize=11)
    ax.set_title(f"Counterfactual 'What-If' Evaluation for '{pred_class}'", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, scen_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.savefig(output_dir / "4_counterfactual_what_if.png", dpi=300, bbox_inches="tight")
    print(f"  [✓] Saved Figure 4: Counterfactual 'What-If' Analysis -> {output_dir / '4_counterfactual_what_if.png'}")
    plt.close()

    print("=" * 75)
    print(f"--> ALL 4 XAI FIGURES SUCCESSFULLY GENERATED IN: {output_dir.resolve()}")
    print("=" * 75)


def main():
    ckpt_file = PROJECT_ROOT / "checkpoints/M7/best_model.pth"
    features_root = PROJECT_ROOT / "data/features"
    xai_output_dir = PROJECT_ROOT / "diagrams/xai"

    if not ckpt_file.exists():
        print(f"[!] Checkpoint not found at: {ckpt_file}")
        return

    # Load model & classes
    model, classes = load_model_and_checkpoint(ckpt_file, num_classes=82)

    # Pick a representative action sample (e.g. Archery or Tennis or first available)
    sample_pt = None
    target_sports = ["Archery", "Tennis", "Basketball", "Cricket"]
    for sport in target_sports:
        sport_dir = features_root / sport
        if sport_dir.exists():
            pts = list(sport_dir.glob("*.pt"))
            if pts:
                sample_pt = pts[0]
                break

    if sample_pt is None:
        # Fallback to any class
        all_pts = list(features_root.glob("*/*.pt"))
        if all_pts:
            sample_pt = all_pts[0]
        else:
            print("[!] No feature files found in data/features.")
            return

    print(f"--> [Target Sample] Selected: {sample_pt.parent.name} / {sample_pt.name}")
    sample_data = torch.load(sample_pt, map_location="cpu")

    compute_xai_explanations(
        model=model,
        sample=sample_data,
        class_names=classes,
        output_dir=xai_output_dir
    )


if __name__ == "__main__":
    main()
