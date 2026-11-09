"""Packages the entire project into a zip file for direct upload to Google Colab."""

import argparse
from pathlib import Path
import time
import zipfile


EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", "runs", "wandb", "checkpoints", "reports"
}

EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".git", ".pt", ".pth"
}


def package_project(
    project_root: Path,
    output_zip: Path,
    include_dataset: bool = True,
) -> None:
    print("=" * 65)
    print("PACKAGING PROJECT FOR GOOGLE COLAB DIRECT UPLOAD")
    print("=" * 65)
    print(f"Project root    : {project_root}")
    print(f"Output archive  : {output_zip}")
    print(f"Include dataset : {include_dataset} (ucf action folder)")
    print("-" * 65)

    start_time = time.time()
    file_count = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for file_path in project_root.rglob("*"):
            if file_path.is_dir():
                continue

            # Check directory exclusions
            rel_path = file_path.relative_to(project_root)
            parts = rel_path.parts

            if any(part in EXCLUDE_DIRS for part in parts):
                continue

            if file_path.suffix in EXCLUDE_EXTS:
                continue

            # Skip the output zip itself
            if file_path.name == output_zip.name or file_path.suffix == ".zip":
                continue

            # Check dataset inclusion
            if not include_dataset and parts[0] == "ucf action":
                continue

            # Also skip raw UCF101 unneeded folder if present
            if parts[0] == "UCF101":
                continue

            zip_out.write(file_path, arcname=str(rel_path))
            file_count += 1

            if file_count % 500 == 0:
                print(f"  --> Packed {file_count} files...")

    elapsed = time.time() - start_time
    size_mb = output_zip.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("PACKAGING COMPLETED SUCCESSFULLY!")
    print(f"Archive file  : {output_zip.resolve()}")
    print(f"Total files   : {file_count}")
    print(f"Archive size  : {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")
    print(f"Time taken    : {elapsed:.1f} seconds")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Package project for Google Colab upload")
    parser.add_argument("--output", type=str, default="Sports_Action_Recognition_Thesis.zip", help="Output zip name")
    parser.add_argument("--code-only", action="store_true", help="Package only codebase without the dataset")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_path = project_root / args.output

    package_project(
        project_root=project_root,
        output_zip=output_path,
        include_dataset=not args.code_only,
    )


if __name__ == "__main__":
    main()
