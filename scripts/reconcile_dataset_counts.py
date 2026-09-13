"""Dataset count reconciliation script for SignalScope version-2 dataset."""

import os
from pathlib import Path
from collections import Counter

v2_path = Path(r"C:\Programming\SignalScope-data\version-2")
train_path = Path(r"C:\Programming\SignalScope-data\train")

print("=== RECONCILING DATASET COUNTS ===")

# 1. Official train scan
train_img_count = 0
train_exts = Counter()
for root, _, files in os.walk(train_path):
    for f in files:
        ext = Path(f).suffix.lower()
        train_exts[ext] += 1
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            train_img_count += 1

print(f"\n1. Official train/ directory:")
print(f"   Total files: {sum(train_exts.values()):,}")
print(f"   Total valid images: {train_img_count:,}")
for ext, count in train_exts.items():
    print(f"     {ext}: {count:,}")

# 2. version-2 scan
v2_files = 0
v2_imgs = 0
v2_csvs = 0
v2_exts = Counter()
folder_breakdown = Counter()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

for root, dirs, files in os.walk(v2_path):
    rel = Path(root).relative_to(v2_path)
    top_folder = rel.parts[0] if rel.parts else "root"
    for f in files:
        v2_files += 1
        ext = Path(f).suffix.lower()
        v2_exts[ext] += 1
        if ext in IMAGE_EXTENSIONS:
            v2_imgs += 1
            folder_breakdown[top_folder] += 1
        elif ext == ".csv":
            v2_csvs += 1

print(f"\n2. version-2/ directory:")
print(f"   Total files: {v2_files:,}")
print(f"   Total image files: {v2_imgs:,}")
print(f"   Total CSV files: {v2_csvs:,}")
print(f"   Extensions:")
for ext, count in v2_exts.items():
    print(f"     {ext}: {count:,}")

print(f"\n   Image count breakdown by top-level folder:")
for folder, cnt in sorted(folder_breakdown.items()):
    print(f"     {folder}: {cnt:,} images")

# 3. Hypotheses for the ~220,000 estimate
print(f"\n3. Hypotheses for User's ~220,000 estimate:")
print(f"   A. Official Train (100k) + Version-2 (143.4k) = {train_img_count + v2_imgs:,} total across repo data.")
print(f"   B. If 'REAL' was estimated as ~200k (e.g. 20 categories x ~10k, but only 11 categories exist with 12,048 each = 132,528).")
print(f"   C. If only the Real pool was considered alongside the official training pool: 100k + 132.5k = 232.5k (close to ~220k-240k rounded down).")
