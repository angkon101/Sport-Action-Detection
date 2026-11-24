import os
import subprocess
import sys
import time

def run_cmd(cmd, check=True):
    print(f"--> Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")
    return res

def push_with_retry(max_retries=3, delay=5):
    for attempt in range(1, max_retries + 1):
        print(f"--> Pushing to origin main (attempt {attempt}/{max_retries})...")
        res = subprocess.run("git push origin main", shell=True, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        if res.returncode == 0:
            print("--> Push successful!")
            return True
        print(f"Push attempt {attempt} failed. Waiting {delay}s before retry...")
        time.sleep(delay)
    raise RuntimeError("Failed to push after multiple attempts.")

def main():
    features_base = r"data\features"
    if not os.path.exists(features_base):
        print("Features base path not found.")
        return

    # Gather top-level items
    all_top = sorted(os.listdir(features_base))
    root_classes = [c for c in all_top if c not in ["Train", "Test"] and os.path.isdir(os.path.join(features_base, c))]
    
    test_dir = os.path.join(features_base, "Test")
    test_classes = sorted(os.listdir(test_dir)) if os.path.exists(test_dir) else []
    
    train_dir = os.path.join(features_base, "Train")
    train_classes = sorted(os.listdir(train_dir)) if os.path.exists(train_dir) else []

    batches = []

    # Batch 1: Root classes
    if root_classes:
        paths = [os.path.join(features_base, c) for c in root_classes]
        batches.append({
            "name": f"Root sample features ({len(root_classes)} classes)",
            "paths": paths
        })

    # Test batches (2 batches)
    test_mid = len(test_classes) // 2
    if test_classes[:test_mid]:
        batches.append({
            "name": f"Test split part 1/2 ({len(test_classes[:test_mid])} classes: {test_classes[0]} to {test_classes[test_mid-1]})",
            "paths": [os.path.join(test_dir, c) for c in test_classes[:test_mid]]
        })
    if test_classes[test_mid:]:
        batches.append({
            "name": f"Test split part 2/2 ({len(test_classes[test_mid:])} classes: {test_classes[test_mid]} to {test_classes[-1]})",
            "paths": [os.path.join(test_dir, c) for c in test_classes[test_mid:]]
        })

    # Train batches (4 batches of ~21 classes each)
    q = len(train_classes) // 4
    t_splits = [
        train_classes[0:q],
        train_classes[q:2*q],
        train_classes[2*q:3*q],
        train_classes[3*q:]
    ]
    for idx, split in enumerate(t_splits):
        if split:
            batches.append({
                "name": f"Train split part {idx+1}/4 ({len(split)} classes: {split[0]} to {split[-1]})",
                "paths": [os.path.join(train_dir, c) for c in split]
            })

    total_batches = len(batches)
    print(f"=== Total Batches to Commit and Push: {total_batches} ===\n")

    for i, b in enumerate(batches, 1):
        print(f"\n=======================================================")
        print(f"Processing Batch {i}/{total_batches}: {b['name']}")
        print(f"=======================================================")

        # Stage files
        for p in b["paths"]:
            run_cmd(f'git add "{p}"')

        # Check if there are staged changes
        diff_res = subprocess.run("git diff --cached --quiet", shell=True)
        if diff_res.returncode == 0:
            print("--> No new changes staged for this batch. Skipping commit & push.")
            continue

        # Commit
        commit_msg = f"feat(data): add multimodal features batch {i}/{total_batches} ({b['name']})"
        run_cmd(f'git commit -m "{commit_msg}"')

        # Push with retry
        push_with_retry()

    print("\n=======================================================")
    print("ALL FEATURE BATCHES COMMITTED AND PUSHED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    main()
