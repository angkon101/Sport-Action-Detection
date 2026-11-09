"""Multimodal Feature Extractor for Sports Action Recognition.

Extracts 4 complementary streams from video frames:
1. Visual Appearance: Pretrained ResNet-50 2048-d embedding
2. Human Pose: 17 keypoints + temporal joint displacement (85-d)
3. Sports Objects: Centroid, scale, velocity, presence mask (7-d)
4. Human-Object Interaction (HOI): Distances, unit vectors, and delta velocities (16-d)

Uses standard PyTorch torchvision detection backbones (Keypoint R-CNN & Faster R-CNN)
providing identical model weights to Detectron2 with zero C++ compilation dependencies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.models.detection as detection
import torchvision.transforms as transforms


# COCO Sports-related class IDs in standard Faster R-CNN / Detectron2
SPORTS_OBJECT_COCO_IDS = {
    2: "bicycle",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    1: "person",
}


class SportsFeatureExtractor:
    """Extracts multimodal visual, pose, object, and HOI feature sequences from video clips."""

    def __init__(self, device: Optional[torch.device] = None, num_frames: int = 16):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames

        print(f"--> Initializing Feature Extractors on {self.device}...")

        # 1. Visual CNN Feature Extractor (ResNet-50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Strip classification fc layer to extract 2048-d pooled embedding
        self.visual_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(self.device)
        self.visual_backbone.eval()

        self.visual_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Keypoint R-CNN (17 Human Pose Keypoints)
        self.keypoint_model = detection.keypointrcnn_resnet50_fpn(
            weights=detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device)
        self.keypoint_model.eval()

        # 3. Faster R-CNN (Object / Sports Gear Detection)
        self.object_model = detection.fasterrcnn_resnet50_fpn(
            weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device)
        self.object_model.eval()

    def sample_video_frames(self, video_path: str | Path) -> List[np.ndarray]:
        """Uniformly samples T frames from a video file using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames <= self.num_frames or total_frames <= 0:
            # Read all available frames
            raw = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                raw.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(raw) == 0:
                raise ValueError(f"Video {video_path} has 0 readable frames.")

            indices = np.linspace(0, len(raw) - 1, self.num_frames, dtype=int)
            return [raw[i] for i in indices]

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()

        return frames

    @torch.no_grad()
    def extract_visual_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Extracts 2048-d visual feature vector per frame. Returns [T, 2048]."""
        tensors = torch.stack([self.visual_transform(f) for f in frames]).to(self.device)
        feats = self.visual_backbone(tensors)  # [T, 2048, 1, 1]
        return feats.flatten(1).cpu()          # [T, 2048]

    @torch.no_grad()
    def extract_pose_features(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """Extracts 17 keypoints with confidence and velocity deltas.

        Returns:
            pose_tensor: [T, 85]
            raw_keypoints: [T, 17, 3] in pixel coordinates
        """
        t_frames = len(frames)
        raw_kpts = np.zeros((t_frames, 17, 3), dtype=np.float32)

        detection_inputs = [transforms.functional.to_tensor(f).to(self.device) for f in frames]
        predictions = self.keypoint_model(detection_inputs)

        for t, pred in enumerate(predictions):
            h, w = frames[t].shape[:2]
            keypoints = pred["keypoints"].cpu().numpy()  # [N, 17, 3] (x, y, visibility)
            scores = pred["scores"].cpu().numpy()

            if len(scores) > 0:
                # Pick the person with the highest confidence score
                best_idx = np.argmax(scores)
                kpt = keypoints[best_idx]  # [17, 3]
                raw_kpts[t] = kpt

        # Normalize (x, y) to [0, 1]
        norm_kpts = np.zeros_like(raw_kpts)
        for t in range(t_frames):
            h, w = frames[t].shape[:2]
            norm_kpts[t, :, 0] = raw_kpts[t, :, 0] / max(w, 1)
            norm_kpts[t, :, 1] = raw_kpts[t, :, 1] / max(h, 1)
            norm_kpts[t, :, 2] = raw_kpts[t, :, 2]  # visibility/confidence

        # Temporal velocity: ΔP_t = P_t - P_{t-1}
        velocities = np.zeros((t_frames, 17, 2), dtype=np.float32)
        for t in range(1, t_frames):
            velocities[t] = norm_kpts[t, :, :2] - norm_kpts[t - 1, :, :2]

        # Combine: 17 * 3 (pos + conf) + 17 * 2 (vel) = 85
        pose_feat = np.concatenate([norm_kpts.reshape(t_frames, -1), velocities.reshape(t_frames, -1)], axis=1)
        return torch.from_numpy(pose_feat).float(), raw_kpts

    @torch.no_grad()
    def extract_object_features(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """Extracts sports object detection features [cx, cy, w, h, vx, vy, mask].

        Returns:
            obj_tensor: [T, 7]
            raw_boxes: [T, 4] in pixel coordinates
        """
        t_frames = len(frames)
        obj_feats = np.zeros((t_frames, 7), dtype=np.float32)
        raw_boxes = np.zeros((t_frames, 4), dtype=np.float32)

        detection_inputs = [transforms.functional.to_tensor(f).to(self.device) for f in frames]
        predictions = self.object_model(detection_inputs)

        for t, pred in enumerate(predictions):
            h, w = frames[t].shape[:2]
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            # Find sports equipment or prominent non-person object
            best_obj_idx = None
            best_score = 0.25

            for idx, (label, score) in enumerate(zip(labels, scores)):
                if label in SPORTS_OBJECT_COCO_IDS and label != 1:  # Not 'person'
                    if score > best_score:
                        best_score = score
                        best_obj_idx = idx

            # If no specific sports gear detected, pick the highest-confidence object interacting in scene
            if best_obj_idx is None and len(scores) > 1:
                # Top non-person object
                for idx, (label, score) in enumerate(zip(labels, scores)):
                    if label != 1 and score > 0.35:
                        best_obj_idx = idx
                        break

            if best_obj_idx is not None:
                box = boxes[best_obj_idx]  # [x1, y1, x2, y2]
                raw_boxes[t] = box
                cx = ((box[0] + box[2]) / 2.0) / max(w, 1)
                cy = ((box[1] + box[3]) / 2.0) / max(h, 1)
                bw = (box[2] - box[0]) / max(w, 1)
                bh = (box[3] - box[1]) / max(h, 1)
                mask = 1.0
                obj_feats[t] = [cx, cy, bw, bh, 0.0, 0.0, mask]

        # Calculate object velocities: vx, vy
        for t in range(1, t_frames):
            if obj_feats[t, 6] == 1.0 and obj_feats[t - 1, 6] == 1.0:
                obj_feats[t, 4] = obj_feats[t, 0] - obj_feats[t - 1, 0]
                obj_feats[t, 5] = obj_feats[t, 1] - obj_feats[t - 1, 1]

        return torch.from_numpy(obj_feats).float(), raw_boxes

    def compute_hoi_features(
        self,
        raw_kpts: np.ndarray,      # [T, 17, 3]
        raw_boxes: np.ndarray,     # [T, 4]
        obj_feats: torch.Tensor,   # [T, 7]
        frames: List[np.ndarray],
    ) -> torch.Tensor:
        """Computes geometric HOI vectors between wrists/ankles and detected sports object.

        Returns:
            hoi_tensor: [T, 16]
        """
        t_frames = len(frames)
        hoi_feats = np.zeros((t_frames, 16), dtype=np.float32)

        # Joint indices: L-wrist=9, R-wrist=10, L-ankle=15, R-ankle=16
        key_joints = [9, 10, 15, 16]

        prev_distances = np.zeros(len(key_joints), dtype=np.float32)

        for t in range(t_frames):
            h, w = frames[t].shape[:2]
            obj_present = (obj_feats[t, 6].item() == 1.0)

            if not obj_present:
                prev_distances.fill(0.0)
                continue

            # Object normalized centroid
            obj_cx = obj_feats[t, 0].item()
            obj_cy = obj_feats[t, 1].item()
            p_obj = np.array([obj_cx, obj_cy], dtype=np.float32)

            current_distances = []
            slot = 0

            for j_idx, joint_id in enumerate(key_joints):
                joint_x = (raw_kpts[t, joint_id, 0] / max(w, 1))
                joint_y = (raw_kpts[t, joint_id, 1] / max(h, 1))
                p_joint = np.array([joint_x, joint_y], dtype=np.float32)

                # 1. Euclidean distance
                dist = float(np.linalg.norm(p_joint - p_obj))
                current_distances.append(dist)

                # 2. Directional unit vector
                diff = p_obj - p_joint
                norm = np.linalg.norm(diff) + 1e-6
                u_vec = diff / norm

                # 3. Rate of distance variation (delta distance)
                delta_d = dist - prev_distances[j_idx] if t > 0 else 0.0

                # Pack into slot (4 values per joint: dist, ux, uy, delta_d)
                hoi_feats[t, slot] = dist
                hoi_feats[t, slot + 1] = float(u_vec[0])
                hoi_feats[t, slot + 2] = float(u_vec[1])
                hoi_feats[t, slot + 3] = delta_d
                slot += 4

            prev_distances = np.array(current_distances, dtype=np.float32)

        return torch.from_numpy(hoi_feats).float()

    def process_video(self, video_path: str | Path) -> Dict[str, torch.Tensor]:
        """Processes a single video clip and returns the 4 multimodal feature tensors.

        Returns:
            dict containing:
                'visual': [16, 2048]
                'pose':   [16, 85]
                'object': [16, 7]
                'hoi':    [16, 16]
        """
        frames = self.sample_video_frames(video_path)

        visual_feat = self.extract_visual_features(frames)
        pose_feat, raw_kpts = self.extract_pose_features(frames)
        obj_feat, raw_boxes = self.extract_object_features(frames)
        hoi_feat = self.compute_hoi_features(raw_kpts, raw_boxes, obj_feat, frames)

        return {
            "visual": visual_feat,
            "pose": pose_feat,
            "object": obj_feat,
            "hoi": hoi_feat,
        }
