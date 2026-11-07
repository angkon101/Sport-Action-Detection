"""UCF101 Sports Action Classes and Split Filtering Utilities.

According to the official UCF101 dataset taxonomy (Soomro et al., CRCV),
actions are divided into 5 types:
1. Human-Object Interaction
2. Body-Motion Only
3. Human-Human Interaction
4. Playing Musical Instruments
5. Sports (gymnastics, athletics, water sports, team & racket sports, etc.)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Curated list of sports-related actions from UCF101
UCF101_SPORTS_CLASSES: List[str] = [
    "Archery",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "CleanAndJerk",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "Fencing",
    "FrisbeeCatch",
    "GolfSwing",
    "HorseRace",
    "HorseRiding",
    "IceDancing",
    "JavelinThrow",
    "JumpRope",
    "Kayaking",
    "PoleVault",
    "PullUps",
    "PushUps",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "Shotput",
    "Skateboarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "VolleyballSpiking",
]

# UCF Sports Action Dataset (13 action categories / viewpoints)
UCF_SPORTS_ACTION_CLASSES: List[str] = [
    "Diving-Side",
    "Golf-Swing-Back",
    "Golf-Swing-Front",
    "Golf-Swing-Side",
    "Kicking-Front",
    "Kicking-Side",
    "Lifting",
    "Riding-Horse",
    "Run-Side",
    "SkateBoarding-Front",
    "Swing-Bench",
    "Swing-SideAngle",
    "Walk-Front",
]


def get_sports_class_to_idx(classes: Optional[List[str]] = None) -> Dict[str, int]:
    """Returns a dictionary mapping class name to 0-indexed integer ID."""
    classes = classes or UCF101_SPORTS_CLASSES
    return {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}


def get_idx_to_sports_class(classes: Optional[List[str]] = None) -> Dict[int, str]:
    """Returns a dictionary mapping 0-indexed integer ID to class name."""
    classes = classes or UCF101_SPORTS_CLASSES
    return {idx: cls_name for idx, cls_name in enumerate(sorted(classes))}


def filter_splits_for_sports(
    input_split_file: Path,
    output_split_file: Path,
    target_classes: Optional[List[str]] = None,
) -> int:
    """Filters an official UCF101 split file (e.g.

    trainlist01.txt or testlist01.txt) to include only the selected sports
    classes, re-mapping class labels to 0-indexed contiguous IDs.

    Returns:
        Number of filtered video entries written.
    """
    class_to_idx = get_sports_class_to_idx(target_classes)
    output_split_file.parent.mkdir(parents=True, exist_ok=True)

    matched_lines = []
    with open(input_split_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            video_rel_path = parts[0]  # e.g., 'Basketball/v_Basketball_g01_c01.avi'
            cls_name = video_rel_path.split("/")[0] if "/" in video_rel_path else video_rel_path.split("\\")[0]

            if cls_name in class_to_idx:
                new_idx = class_to_idx[cls_name]
                matched_lines.append(f"{video_rel_path} {new_idx}\n")

    with open(output_split_file, "w", encoding="utf-8") as out:
        out.writelines(matched_lines)

    return len(matched_lines)

