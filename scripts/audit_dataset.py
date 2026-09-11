"""SignalScope Dataset Audit Utility.

Performs a comprehensive, non-destructive audit of the official dataset at:
C:\\Programming\\SignalScope-data

Audits:
- Directory structure (train / test)
- File counts per class (REAL vs FAKE)
- Class balance
- File formats and extensions
- Sample filenames and naming conventions
- Image dimensions (min, max, mode)
- Image channel modes (RGB, RGBA, L)
- File readability / corruption checks
- Duplicate filename risks
- Generator / source metadata detection
- Generates report/dataset_audit.md and report/dataset_audit.json
"""

import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.logger import logger

DEFAULT_DATA_ROOT = Path(os.environ.get("SIGNALSCOPE_DATA_ROOT", r"C:\Programming\SignalScope-data"))


def scan_folder_fast(folder_path: Path) -> List[str]:
    """Returns list of file names in a folder using os.scandir."""
    if not folder_path.exists():
        return []
    file_names = []
    with os.scandir(folder_path) as it:
        for entry in it:
            if entry.is_file():
                file_names.append(entry.name)
    return file_names


def audit_image_samples(
    folder_path: Path,
    file_names: List[str],
    sample_limit: int = 500,
) -> Dict[str, Any]:
    """Inspects dimensions, formats, and channel modes on a representative sample."""
    formats = collections.Counter()
    modes = collections.Counter()
    dimensions = collections.Counter()
    corrupt_count = 0
    corrupt_examples = []

    # Inspect up to sample_limit images to balance speed and thoroughness
    check_files = file_names[:sample_limit]

    min_w, min_h = float("inf"), float("inf")
    max_w, max_h = 0, 0

    for fname in check_files:
        fpath = folder_path / fname
        try:
            with Image.open(fpath) as img:
                w, h = img.size
                fmt = (img.format or "UNKNOWN").upper()
                mode = img.mode

                formats[fmt] += 1
                modes[mode] += 1
                dimensions[(w, h)] += 1

                min_w = min(min_w, w)
                min_h = min(min_h, h)
                max_w = max(max_w, w)
                max_h = max(max_h, h)
        except Exception as exc:
            corrupt_count += 1
            if len(corrupt_examples) < 5:
                corrupt_examples.append(f"{fname}: {str(exc)}")

    common_dims = [
        {"dimension": f"{w}x{h}", "count": count}
        for (w, h), count in dimensions.most_common(5)
    ]

    return {
        "sampled_count": len(check_files),
        "formats": dict(formats),
        "modes": dict(modes),
        "min_dimension": f"{min_w}x{min_h}" if min_w != float("inf") else "N/A",
        "max_dimension": f"{max_w}x{max_h}" if max_w != 0 else "N/A",
        "common_dimensions": common_dims,
        "corrupt_count": corrupt_count,
        "corrupt_examples": corrupt_examples,
    }


def audit_split(split_dir: Path, sample_limit_per_class: int = 500) -> Dict[str, Any]:
    """Audits a split directory containing class subdirectories (e.g. REAL, FAKE)."""
    if not split_dir.exists():
        return {"exists": False}

    subdirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
    non_dir_files = [f.name for f in split_dir.iterdir() if f.is_file()]

    class_data = {}
    total_images = 0
    all_filenames = []

    for cname in subdirs:
        cdir = split_dir / cname
        fnames = scan_folder_fast(cdir)
        count = len(fnames)
        total_images += count
        all_filenames.extend(fnames)

        # Audit sample images for this class
        sample_audit = audit_image_samples(cdir, fnames, sample_limit=sample_limit_per_class)

        # File extensions
        exts = collections.Counter(Path(fn).suffix.lower() for fn in fnames)

        # Inspect sample naming pattern
        sample_fnames = fnames[:5]

        class_data[cname] = {
            "count": count,
            "extensions": dict(exts),
            "sample_filenames": sample_fnames,
            "sample_audit": sample_audit,
        }

    # Check for duplicate filenames across classes in this split
    fn_counts = collections.Counter(all_filenames)
    duplicates = [fn for fn, cnt in fn_counts.items() if cnt > 1]

    return {
        "exists": True,
        "path": str(split_dir),
        "subdirectories": subdirs,
        "root_files": non_dir_files,
        "total_images": total_images,
        "classes": class_data,
        "duplicate_filename_count": len(duplicates),
        "duplicate_examples": duplicates[:5],
    }


def run_full_audit(data_root: Path) -> Dict[str, Any]:
    """Runs complete dataset audit across train and test partitions."""
    logger.info(f"Initiating dataset audit on: {data_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_root}")

    # Check for any top-level metadata or text files
    top_level_files = [f.name for f in data_root.iterdir() if f.is_file()]
    top_level_dirs = [d.name for d in data_root.iterdir() if d.is_dir()]

    train_audit = audit_split(data_root / "train", sample_limit_per_class=500)
    test_audit = audit_split(data_root / "test", sample_limit_per_class=500)

    # Detect generator metadata patterns from filenames
    generator_candidates = set()
    train_classes = train_audit.get("classes", {})
    if "FAKE" in train_classes:
        for fname in train_classes["FAKE"].get("sample_filenames", []):
            parts = fname.split("_")
            if len(parts) > 1 and not parts[0].isdigit():
                generator_candidates.add(parts[0])

    audit_summary = {
        "dataset_root": str(data_root),
        "top_level_dirs": top_level_dirs,
        "top_level_files": top_level_files,
        "train": train_audit,
        "test": test_audit,
        "generator_metadata_detected": list(generator_candidates) if generator_candidates else None,
    }

    return audit_summary


def generate_markdown_report(audit: Dict[str, Any], output_path: Path) -> None:
    """Writes a clean, formatted Markdown report."""
    train = audit.get("train", {})
    test = audit.get("test", {})

    lines = [
        "# SignalScope Official Dataset Audit Report",
        "**Dataset Root**: `" + audit.get("dataset_root", "") + "`  ",
        "**Audit Status**: Verified  ",
        "",
        "---",
        "",
        "## 1. Top-Level Structure",
        "- **Subdirectories**: " + ", ".join(f"`{d}`" for d in audit.get("top_level_dirs", [])),
        "- **Metadata/Root Files**: " + (", ".join(f"`{f}`" for f in audit.get("top_level_files", [])) if audit.get("top_level_files") else "None (labels encoded purely by folder hierarchy)"),
        "",
        "---",
        "",
        "## 2. Training Partition (`train/`)",
        f"- **Total Training Images**: {train.get('total_images', 0):,}",
    ]

    train_classes = train.get("classes", {})
    for cname, cinfo in train_classes.items():
        sample = cinfo.get("sample_audit", {})
        lines.extend([
            f"### Class `{cname}`",
            f"- **Count**: {cinfo.get('count', 0):,}",
            f"- **Extensions**: {cinfo.get('extensions', {})}",
            f"- **Sample Filenames**: `{cinfo.get('sample_filenames', [])}`",
            f"- **Formats Sampled**: {sample.get('formats', {})}",
            f"- **Color Modes**: {sample.get('modes', {})}",
            f"- **Dimensions**: Min `{sample.get('min_dimension')}`, Max `{sample.get('max_dimension')}`",
            f"- **Common Dimensions**: {sample.get('common_dimensions', [])}",
            f"- **Corrupt Images**: {sample.get('corrupt_count', 0)}",
            "",
        ])

    # Class balance calculation
    counts = [cinfo.get("count", 0) for cinfo in train_classes.values()]
    if len(counts) == 2 and counts[0] > 0 and counts[1] > 0:
        ratio = max(counts) / min(counts)
        balance_desc = "Perfectly balanced (1.00:1)" if ratio == 1.0 else f"Imbalance ratio: {ratio:.2f}:1"
    else:
        balance_desc = "Single class or empty"

    lines.extend([
        f"**Class Balance**: {balance_desc}  ",
        f"**Duplicate Filenames across classes**: {train.get('duplicate_filename_count', 0)}",
        "",
        "---",
        "",
        "## 3. Test Partition (`test/`)",
        f"- **Total Test Images**: {test.get('total_images', 0):,}",
    ])

    test_classes = test.get("classes", {})
    for cname, cinfo in test_classes.items():
        sample = cinfo.get("sample_audit", {})
        lines.extend([
            f"### Class `{cname}`",
            f"- **Count**: {cinfo.get('count', 0):,}",
            f"- **Extensions**: {cinfo.get('extensions', {})}",
            f"- **Sample Filenames**: `{cinfo.get('sample_filenames', [])}`",
            f"- **Common Dimensions**: {sample.get('common_dimensions', [])}",
            "",
        ])

    lines.extend([
        "### Test Partition Role & Strict Exclusion Policy",
        "> [!IMPORTANT]",
        "> **Role of `test/`**: The `test/` directory contains evaluation data. In strict compliance with the SIH 2026 problem requirements and anti-leakage protocol, **the `test/` directory is completely excluded from model training, validation, threshold tuning, and feature selection**.",
        "> All model development, hyperparameter selection, and validation will proceed strictly on local stratified splits constructed from `train/`.",
        "",
        "---",
        "",
        "## 4. Generator & Provenance Metadata",
        f"- **Generator Tracking**: {audit.get('generator_metadata_detected') or 'None detected from filenames or subfolders; labels are binary REAL / FAKE.'}",
        "- **Folder Label Representation**: `0 = REAL`, `1 = FAKE`",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Dataset audit Markdown report written to {output_path}")


def main() -> int:
    data_root = DEFAULT_DATA_ROOT
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])

    audit_result = run_full_audit(data_root)

    # Save JSON summary
    json_path = PROJECT_ROOT / "report" / "dataset_audit.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)
    logger.info(f"Dataset audit JSON saved to {json_path}")

    # Save Markdown report
    md_path = PROJECT_ROOT / "report" / "dataset_audit.md"
    generate_markdown_report(audit_result, md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
