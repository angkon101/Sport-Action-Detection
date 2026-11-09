"""Packages the codebase for Google Colab upload."""

import argparse
from pathlib import Path
import time
import zipfile


EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", "runs", "wandb", "checkpoints", "reports", "diagrams"
}

EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".git", ".pt", ".pth", ".zip"
}


def package_codebase(
    project_root: Path,
    output_zip: Path,
) -> None:
    print("=" * 65)
    print("PACKAGING CODEBASE FOR GOOGLE COLAB")
    print("=" * 65)
    print(f"Project root   : {project_root}")
    print(f"Output archive : {output_zip}")
    print("-" * 65)

    start_time = time.time()
    file_count = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for file_path in project_root.rglob("*"):
            if file_path.is_dir():
                continue

            rel_path = file_path.relative_to(project_root)
            parts = rel_path.parts

            if any(part in EXCLUDE_DIRS for part in parts):
                continue

            if file_path.suffix in EXCLUDE_EXTS:
                continue

            # Skip the large raw dataset folders (since ucf action.zip already exists separately!)
            if parts[0] in ["ucf action", "UCF101", "data/raw", "data/features"]:
                continue

            zip_out.write(file_path, arcname=str(rel_path))
            file_count += 1

    elapsed = time.time() - start_time
    size_kb = output_zip.stat().st_size / 1024

    print("\n" + "=" * 65)
    print("CODEBASE PACKAGED SUCCESSFULLY!")
    print(f"Archive file : {output_zip.resolve()}")
    print(f"Total files  : {file_count}")
    print(f"Archive size : {size_kb:.1f} KB")
    print(f"Time taken   : {elapsed:.2f} seconds")
    print("=" * 65)
    print("\nNow you have two ready files for Google Colab:")
    print("1. 'Sports_Action_Codebase.zip' (Lightweight code, uploads in seconds)")
    print("2. 'ucf action.zip' (Your dataset)")


def main():
    parser = argparse.ArgumentParser(description="Package codebase for Colab")
    parser.add_argument("--output", type=str, default="Sports_Action_Codebase.zip")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    package_codebase(project_root, project_root / args.output)


if __name__ == "__main__":
    main()
