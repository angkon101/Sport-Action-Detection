"""Real-Time Inference and Action Validation Engine for Sports Action Recognition.

Processes input video files or live camera frames:
1. Samples frames and extracts 4-stream multimodal features (ResNet-50, Keypoint R-CNN, Faster R-CNN, HOI)
2. Runs the HOIBiLSTMClassifier with Temporal Attention
3. Overlays skeleton bones, sports equipment bounding boxes, HOI interaction vectors, and a top prediction HUD banner
4. Exports annotated video or renders live GUI stream.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.hoi_feature_extractor import SportsFeatureExtractor, SPORTS_OBJECT_COCO_IDS
from src.models.hoi_bilstm import HOIBiLSTMClassifier


# Standard COCO 17-Joint Skeleton bone connections
COCO_SKELETON_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head: nose to eyes, eyes to ears
    (5, 6),                                    # Shoulders: L-shoulder to R-shoulder
    (5, 7), (7, 9),                            # Left arm: shoulder to elbow, elbow to wrist
    (6, 8), (8, 10),                           # Right arm: shoulder to elbow, elbow to wrist
    (5, 11), (6, 12),                          # Torso: shoulders to hips
    (11, 12),                                  # Hips: L-hip to R-hip
    (11, 13), (13, 15),                        # Left leg: hip to knee, knee to ankle
    (12, 14), (14, 16),                        # Right leg: hip to knee, knee to ankle
]

# Color palettes (BGR format for OpenCV)
NEON_GREEN = (50, 255, 50)
NEON_CYAN = (255, 255, 0)
NEON_YELLOW = (0, 255, 255)
NEON_ORANGE = (0, 165, 255)
NEON_PURPLE = (255, 0, 255)
DARK_BG = (20, 20, 20)


def load_model_and_classes(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[HOIBiLSTMClassifier], List[str]]:
    """Loads trained HOIBiLSTMClassifier and class names list."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_classes_path = PROJECT_ROOT / "data" / "splits" / "class_indices_82.json"

    classes = []
    if default_classes_path.exists():
        import json
        with open(default_classes_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)
            classes = sorted(list(class_map.keys()))

    if not classes:
        # Fallback to scanning Model Data categories if available
        model_data_dir = PROJECT_ROOT / "Model Data" / "Train"
        if model_data_dir.exists():
            classes = sorted([d.name for d in model_data_dir.iterdir() if d.is_dir()])
        else:
            classes = [f"Class_{i}" for i in range(82)]

    num_classes = len(classes)
    model = HOIBiLSTMClassifier(
        num_classes=num_classes,
        use_visual=True,
        use_pose=True,
        use_object=True,
        use_hoi=True,
        use_attention=True,
    ).to(device)

    def _load_weights(ckpt_dict):
        state = ckpt_dict.get("model_state", ckpt_dict.get("state_dict", ckpt_dict))
        # Handle attribute name variations (e.g. fusion vs fusion_layer, attention vs temporal_pooling)
        remapped_state = {}
        for k, v in state.items():
            new_k = k
            if new_k.startswith("fusion."):
                new_k = new_k.replace("fusion.", "fusion_layer.")
            elif new_k.startswith("attention."):
                new_k = new_k.replace("attention.", "temporal_pooling.")
            remapped_state[new_k] = v
        model.load_state_dict(remapped_state)
        cls_list = ckpt_dict.get("classes", None)
        return cls_list

    # Load weights if checkpoint exists
    if checkpoint_path is not None:
        ckpt_p = Path(checkpoint_path)
        if ckpt_p.exists():
            print(f"--> Loading trained model weights from: {ckpt_p}")
            try:
                ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_p, map_location=device)
            loaded_classes = _load_weights(ckpt)
            if loaded_classes:
                classes = loaded_classes
            model.eval()
            return model, classes

    # Default checkpoint location check
    default_ckpt = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pth"
    if not default_ckpt.exists():
        default_ckpt = PROJECT_ROOT / "checkpoints" / "M7" / "best_model.pt"
    if default_ckpt.exists():
        print(f"--> Found default checkpoint at: {default_ckpt}")
        try:
            ckpt = torch.load(default_ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(default_ckpt, map_location=device)
        loaded_classes = _load_weights(ckpt)
        if loaded_classes:
            classes = loaded_classes
        model.eval()
        return model, classes

    print("[!] No pre-trained checkpoint found; using model in initialization mode for visual validation.")
    model.eval()
    return model, classes


def draw_skeleton(
    img: np.ndarray,
    keypoints: np.ndarray,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """Draws 17 skeletal joints and bones onto an image."""
    # Draw bones
    for p1_idx, p2_idx in COCO_SKELETON_BONES:
        kpt1, kpt2 = keypoints[p1_idx], keypoints[p2_idx]
        if kpt1[2] > conf_thresh and kpt2[2] > conf_thresh:
            pt1 = (int(kpt1[0]), int(kpt1[1]))
            pt2 = (int(kpt2[0]), int(kpt2[1]))
            cv2.line(img, pt1, pt2, NEON_GREEN, 2, cv2.LINE_AA)

    # Draw joint dots
    for i in range(17):
        kpt = keypoints[i]
        if kpt[2] > conf_thresh:
            pt = (int(kpt[0]), int(kpt[1]))
            # Wrists and ankles in orange, others in cyan
            color = NEON_ORANGE if i in [9, 10, 15, 16] else NEON_CYAN
            radius = 5 if i in [9, 10, 15, 16] else 3
            cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def draw_hoi_interactions(
    img: np.ndarray,
    keypoints: np.ndarray,
    box: np.ndarray,
    has_object: bool,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """Draws sports equipment bounding box and HOI vector lines connecting wrists/ankles to gear."""
    if not has_object:
        return img

    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Draw object bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), NEON_YELLOW, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, NEON_YELLOW, -1, cv2.LINE_AA)
    cv2.putText(img, "SPORTS GEAR", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_YELLOW, 1, cv2.LINE_AA)

    # Draw dashed/connecting HOI interaction lines from wrists (9, 10) & ankles (15, 16)
    interaction_joints = [9, 10, 15, 16]
    for j_idx in interaction_joints:
        kpt = keypoints[j_idx]
        if kpt[2] > conf_thresh:
            pt = (int(kpt[0]), int(kpt[1]))
            # Draw glowing interaction vector line
            cv2.line(img, pt, (cx, cy), NEON_PURPLE, 1, cv2.LINE_AA)

    return img


def draw_hud_banner(
    img: np.ndarray,
    predictions: List[Tuple[str, float]],
    attention_weight: Optional[float] = None,
) -> np.ndarray:
    """Draws a semi-transparent top HUD banner with Top-1 action and confidence bars."""
    h, w = img.shape[:2]
    banner_h = 75
    overlay = img.copy()

    # Top banner background
    cv2.rectangle(overlay, (0, 0), (w, banner_h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    top_class, top_conf = predictions[0]

    # Action label
    label_text = f"ACTION: {top_class}"
    conf_text = f"{top_conf:.1f}% Confidence"
    cv2.putText(img, label_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, conf_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_GREEN, 1, cv2.LINE_AA)

    # Attention indicator if available
    if attention_weight is not None:
        attn_text = f"Temporal Saliency: {attention_weight:.2f}"
        cv2.putText(img, attn_text, (w - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_CYAN, 1, cv2.LINE_AA)

    # Top-2 & Top-3 mini labels on the right
    if len(predictions) > 1:
        c2, p2 = predictions[1]
        cv2.putText(img, f"2. {c2[:14]}: {p2:.0f}%", (w - 240, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    if len(predictions) > 2:
        c3, p3 = predictions[2]
        cv2.putText(img, f"3. {c3[:14]}: {p3:.0f}%", (w - 240, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return img


def annotate_video_frames(
    frames: List[np.ndarray],
    raw_kpts: np.ndarray,
    raw_boxes: np.ndarray,
    obj_feats: torch.Tensor,
    predictions: List[Tuple[str, float]],
    attention_weights: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Annotates a sequence of RGB frames with pose, object, HOI, and predictions."""
    annotated = []
    num_frames = len(frames)

    for t in range(num_frames):
        # Convert RGB to BGR for OpenCV rendering
        bgr = cv2.cvtColor(frames[t].copy(), cv2.COLOR_RGB2BGR)

        kpts_t = raw_kpts[t]
        box_t = raw_boxes[t]
        has_obj = (obj_feats[t, 6].item() == 1.0)
        attn_t = float(attention_weights[t]) if attention_weights is not None else None

        # 1. Draw 17-joint skeleton
        bgr = draw_skeleton(bgr, kpts_t)

        # 2. Draw sports equipment and HOI interaction lines
        bgr = draw_hoi_interactions(bgr, kpts_t, box_t, has_obj)

        # 3. Draw HUD prediction banner
        bgr = draw_hud_banner(bgr, predictions, attn_t)

        # Convert back to RGB
        annotated.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return annotated


class ActionInferenceEngine:
    """End-to-End inference and validation engine for any video clip or camera stream."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        num_frames: int = 16,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.extractor = SportsFeatureExtractor(device=self.device, num_frames=self.num_frames)
        self.model, self.classes = load_model_and_classes(checkpoint_path, self.device)

    def predict_video(
        self,
        video_path_or_frames: Union[str, Path, List[np.ndarray]],
        topk: int = 5,
    ) -> Dict[str, object]:
        """Runs full 4-stream multimodal prediction on a video or list of RGB frames.

        Returns:
            dict containing:
                'predictions': List of (class_name, confidence_percent)
                'annotated_frames': List of RGB annotated frames
                'attention_weights': 16 temporal attention values
                'top_class': string name of predicted sport
                'top_confidence': float percentage
        """
        # 1. Extract multimodal features & detections
        if isinstance(video_path_or_frames, (str, Path)):
            feat_dict = self.extractor.process_video_with_detections(video_path_or_frames)
        else:
            feat_dict = self.extractor.process_frames(video_path_or_frames)

        v = feat_dict["visual"].unsqueeze(0).to(self.device)  # [1, 16, 2048]
        p = feat_dict["pose"].unsqueeze(0).to(self.device)    # [1, 16, 85]
        o = feat_dict["object"].unsqueeze(0).to(self.device)  # [1, 16, 7]
        h = feat_dict["hoi"].unsqueeze(0).to(self.device)     # [1, 16, 16]

        # 2. Run model
        with torch.no_grad():
            logits, alpha = self.model(visual=v, pose=p, obj=o, hoi=h)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        # Top-K predictions
        top_probs, top_indices = probs.topk(min(topk, len(self.classes)))
        predictions = [
            (self.classes[idx.item()], prob.item() * 100.0)
            for idx, prob in zip(top_indices, top_probs)
        ]

        attn_np = alpha.squeeze(0).cpu().numpy() if alpha is not None else None

        # 3. Annotate frames with skeleton, bounding box, and HOI vectors
        annotated_frames = annotate_video_frames(
            frames=feat_dict["frames"],
            raw_kpts=feat_dict["raw_kpts"],
            raw_boxes=feat_dict["raw_boxes"],
            obj_feats=feat_dict["object"],
            predictions=predictions,
            attention_weights=attn_np,
        )

        return {
            "predictions": predictions,
            "top_class": predictions[0][0],
            "top_confidence": predictions[0][1],
            "annotated_frames": annotated_frames,
            "attention_weights": attn_np,
            "raw_frames": feat_dict["frames"],
        }

    def save_annotated_video(
        self,
        annotated_frames: List[np.ndarray],
        output_path: Union[str, Path],
        fps: float = 8.0,
    ) -> Path:
        """Saves annotated RGB frames to an MP4 video file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not annotated_frames:
            raise ValueError("No frames to save.")

        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))

        for frame in annotated_frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()
        return out_p
