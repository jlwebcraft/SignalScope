"""Independent verification of SignalScope v2.0.0 GitHub Release asset.

1. Resolves stable public download URL for v2.0.0/best_model.pt
2. Downloads asset independently
3. Verifies file size matches 111,359,988 bytes
4. Verifies SHA-256 checksum matches 47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CANONICAL_SHA256 = "47f6b2a19d6113d25028b1434d5c830a4521830621a44f76af43acd6be55178d"
EXPECTED_SIZE = 111359988
RELEASE_URL = "https://github.com/jlwebcraft/SignalScope/releases/download/v2.0.0/best_model.pt"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_release_asset():
    print("==================================================")
    print(" SIGNALSCOPE v2.0.0 RELEASE ARTIFACT VERIFICATION")
    print("==================================================")
    print(f"Target URL: {RELEASE_URL}")

    cache_dir = Path("cache/verification")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "v2_downloaded_best_model.pt"
    if dest.exists():
        dest.unlink()

    print("\n[1/3] Downloading release asset via curl...")
    cmd = ["curl.exe", "-L", "-sS", "-o", str(dest), RELEASE_URL]
    res = subprocess.run(cmd)
    assert res.returncode == 0, f"Download failed with returncode {res.returncode}"
    assert dest.exists(), "Downloaded file not found on disk!"

    size = dest.stat().st_size
    print(f"[2/3] Checking file size...")
    print(f"      Downloaded size: {size} bytes ({size / (1024*1024):.2f} MB)")
    print(f"      Expected size:   {EXPECTED_SIZE} bytes ({EXPECTED_SIZE / (1024*1024):.2f} MB)")
    assert size == EXPECTED_SIZE, f"Size mismatch! Got {size}, expected {EXPECTED_SIZE}"

    print(f"[3/3] Computing SHA-256 checksum...")
    sha = compute_sha256(dest)
    print(f"      Computed SHA-256: {sha}")
    print(f"      Expected SHA-256: {CANONICAL_SHA256}")
    assert sha == CANONICAL_SHA256, f"SHA-256 mismatch! Got {sha}, expected {CANONICAL_SHA256}"

    print("\n==================================================")
    print(" RELEASE ASSET INTEGRITY VERIFIED 100%!")
    print(" SHA-256: MATCHES CANONICAL HASH")
    print(" SIZE:    EXACT MATCH (111,359,988 bytes)")
    print("==================================================")

    # Clean up verification file
    dest.unlink()
    return True


if __name__ == "__main__":
    verify_release_asset()
