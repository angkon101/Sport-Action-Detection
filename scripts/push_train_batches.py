import os
import subprocess
import sys
import time

def run_cmd(cmd, check=True):
    print(f"--> {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {res.returncode}): {cmd}")
    return res

def push_with_retry(max_retries=3, delay=5):
    for attempt in range(1, max_retries + 1):
        print(f"--> Pushing to origin main (attempt {attempt}/{max_retries})...")
        res = subprocess.run("git push origin main -v", shell=True, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        if res.returncode == 0:
            print("--> Push successful!")
            return True
        print(f"Push attempt {attempt} failed. Retrying in {delay}s...")
        time.sleep(delay)
    raise RuntimeError("Failed to push after retries.")

def main():
    train_dir = r"data\features\Train"
    classes = sorted(os.listdir(train_dir))
    
    slices = [
        (classes[0:20], "batch 1/4 (Archery to Fencing)"),
        (classes[20:40], "batch 2/4 (FieldHockeyPenalty to JumpingJack)"),
        (classes[40:60], "batch 3/4 (Kayaking to Shotput)"),
        (classes[60:82], "batch 4/4 (SkateBoarding to WallPushups)")
    ]

    for idx, (batch_classes, desc) in enumerate(slices, 1):
        print(f"\n=======================================================")
        print(f"Processing Train Batch {idx}/4: {desc}")
        print(f"=======================================================")
        
        for c in batch_classes:
            c_path = os.path.join(train_dir, c)
            run_cmd(f'git add "{c_path}"')

        # Check if changes staged
        diff_res = subprocess.run("git diff --cached --quiet", shell=True)
        if diff_res.returncode == 0:
            print("--> No new changes staged. Skipping.")
            continue

        commit_msg = f"feat(data): add multimodal train features {desc}"
        run_cmd(f'git commit -m "{commit_msg}"')

        push_with_retry()
        print("--> Waiting 3 seconds before next batch...")
        time.sleep(3)

    print("\n=======================================================")
    print("ALL 4 TRAIN FEATURE BATCHES PUSHED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    main()
