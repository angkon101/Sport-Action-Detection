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
    run_cmd("git config http.version HTTP/1.1")
    train_dir = r"data\features\Train"
    all_classes = sorted(os.listdir(train_dir))

    # Determine tracked classes
    out = subprocess.check_output("git ls-files data/features/Train", shell=True, text=True)
    tracked = set()
    for line in out.splitlines():
        parts = line.replace("\\", "/").split("/")
        if len(parts) >= 4:
            tracked.add(parts[3])

    untracked = [c for c in all_classes if c not in tracked]
    print(f"Total Train classes: {len(all_classes)}, Tracked: {len(tracked)}, Untracked: {len(untracked)}")

    chunk_size = 7
    chunks = [untracked[i:i+chunk_size] for i in range(0, len(untracked), chunk_size)]
    total_chunks = len(chunks)

    print(f"Total Micro-Chunks to process: {total_chunks}")

    for idx, chunk in enumerate(chunks, 1):
        desc = f"chunk {idx}/{total_chunks} ({len(chunk)} classes: {chunk[0]} to {chunk[-1]})"
        print(f"\n=======================================================")
        print(f"Processing Train {desc}")
        print(f"=======================================================")

        for c in chunk:
            c_path = os.path.join(train_dir, c)
            run_cmd(f'git add "{c_path}"')

        diff_res = subprocess.run("git diff --cached --quiet", shell=True)
        if diff_res.returncode == 0:
            print("--> No new changes staged. Skipping.")
            continue

        commit_msg = f"feat(data): add train features {desc}"
        run_cmd(f'git commit -m "{commit_msg}"')

        push_with_retry()
        print("--> Waiting 3 seconds before next chunk...")
        time.sleep(3)

    # Finally, stage and push any scripts or other files
    run_cmd('git add scripts/')
    diff_res = subprocess.run("git diff --cached --quiet", shell=True)
    if diff_res.returncode != 0:
        run_cmd('git commit -m "feat(scripts): update batch pusher scripts"')
        push_with_retry()

    print("\n=======================================================")
    print("ALL REMAINING TRAIN FEATURE CHUNKS PUSHED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    main()
