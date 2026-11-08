"""Detects and categorizes sports-related action folders from a UCF-101 directory."""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Strict Non-Sports classes from UCF101 (Instruments, daily chores/grooming, etc.)
NON_SPORTS_CLASSES = {
    # Musical Instruments
    "Drumming", "PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute",
    "PlayingGuitar", "PlayingPiano", "PlayingSitar", "PlayingTabla", "PlayingViolin",
    # Daily Grooming & Domestic Activities
    "ApplyEyeMakeup", "ApplyLipstick", "BlowDryHair", "BlowingCandles", "BrushingTeeth",
    "CuttingInKitchen", "Haircut", "Hammering", "HeadMassage", "Knitting", "Mixing",
    "MoppingFloor", "PizzaTossing", "ShavingBeard", "Typing", "WritingOnBoard", "YoYo",
    # Miscellaneous / Non-Sports
    "BabyCrawling", "BandMarching", "MilitaryParade", "WalkingWithDog", "SalsaSpin"
}

# Categorized taxonomy of sports in UCF101
SPORTS_TAXONOMY = {
    "Ball & Team Sports": [
        "BaseballPitch", "Basketball", "BasketballDunk", "Billiards", "Bowling",
        "CricketBowling", "CricketShot", "FieldHockeyPenalty", "FrisbeeCatch",
        "GolfSwing", "SoccerJuggling", "SoccerPenalty", "TableTennisShot",
        "TennisSwing", "VolleyballSpiking"
    ],
    "Track & Field (Athletics)": [
        "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault",
        "Shotput", "ThrowDiscus"
    ],
    "Gymnastics & Acrobatics": [
        "BalanceBeam", "FloorGymnastics", "ParallelBars", "PommelHorse",
        "StillRings", "TrampolineJumping", "UnevenBars", "IceDancing"
    ],
    "Water & Extreme Sports": [
        "BreastStroke", "CliffDiving", "Diving", "FrontCrawl", "Kayaking",
        "Rafting", "Rowing", "SkateBoarding", "Skiing", "Skijet",
        "SkyDiving", "Surfing"
    ],
    "Combat & Martial Arts": [
        "Archery", "BoxingPunchingBag", "BoxingSpeedBag", "Fencing",
        "Nunchucks", "Punch", "SumoWrestling"
    ],
    "Fitness, Strength & Athletics": [
        "BenchPress", "BodyWeightSquats", "CleanAndJerk", "HandstandPushups",
        "HandstandWalking", "JumpingJack", "JumpRope", "Lunges", "PullUps",
        "PushUps", "RopeClimbing", "WallPushups", "RockClimbingIndoor", "TaiChi"
    ],
    "Equestrian & Leisure Sports": [
        "Biking", "HorseRace", "HorseRiding", "HulaHoop", "JugglingBalls", "Swing"
    ]
}


def scan_sports_folders(ucf_dir: Path) -> dict:
    if not ucf_dir.exists():
        raise FileNotFoundError(f"Folder not found: {ucf_dir}")

    all_dirs = {d.name: d for d in ucf_dir.iterdir() if d.is_dir()}
    detected = {}
    total_videos = 0

    for cat_name, class_list in SPORTS_TAXONOMY.items():
        cat_classes = []
        for cls_name in class_list:
            if cls_name in all_dirs:
                cls_dir = all_dirs[cls_name]
                video_count = len(list(cls_dir.glob("*.avi"))) + len(list(cls_dir.glob("*.mp4")))
                cat_classes.append({
                    "name": cls_name,
                    "videos": video_count,
                    "path": str(cls_dir)
                })
                total_videos += video_count
        if cat_classes:
            detected[cat_name] = cat_classes

    return {
        "source_directory": str(ucf_dir),
        "total_sports_classes": sum(len(c) for c in detected.values()),
        "total_sports_videos": total_videos,
        "taxonomy": detected,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect sports action folders in UCF-101")
    parser.add_argument("--ucf-dir", type=str, default=r"E:\Akhi Thesis\UCF101\UCF-101", help="Path to UCF-101 directory")
    parser.add_argument("--output", type=str, default="data/splits/detected_sports_classes.json", help="Output JSON path")
    args = parser.parse_args()

    ucf_path = Path(args.ucf_dir)
    results = scan_sports_folders(ucf_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"UCF-101 SPORTS ACTION DETECTION REPORT")
    print(f"Directory: {ucf_path}")
    print("=" * 60)
    print(f"Total Sports Classes Found : {results['total_sports_classes']} / 101")
    print(f"Total Sports Videos Found  : {results['total_sports_videos']}")
    print("-" * 60)

    for cat_name, classes in results["taxonomy"].items():
        cat_video_sum = sum(c["videos"] for c in classes)
        print(f"\n[*] {cat_name} ({len(classes)} classes, {cat_video_sum} videos):")
        names_str = ", ".join(f"{c['name']} ({c['videos']})" for c in classes)
        print(f"   {names_str}")

    print("\n" + "=" * 60)
    print(f"Complete report saved to: {out_path}")


if __name__ == "__main__":
    main()
