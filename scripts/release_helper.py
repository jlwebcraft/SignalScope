"""GitHub Release and Asset Helper for SignalScope using curl.exe.
"""

import json
import subprocess
import sys


def get_token() -> str:
    p = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        text=True,
        capture_output=True,
        check=True,
    )
    for line in p.stdout.strip().split("\n"):
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("Could not retrieve GitHub password/token from git credential-manager")


def check_releases(repo: str = "jlwebcraft/SignalScope"):
    token = get_token()
    cmd = [
        "curl.exe",
        "-sS",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Accept: application/vnd.github+json",
        f"https://api.github.com/repos/{repo}/releases",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    releases = json.loads(res.stdout)
    print(f"Total releases found: {len(releases)}")
    for r in releases:
        print(f"\nTag: {r.get('tag_name')}, Name: {r.get('name')}, Author: {r.get('author', {}).get('login')}")
        assets = r.get("assets", [])
        print(f"Assets ({len(assets)}):")
        for a in assets:
            print(f"  - Name: {a.get('name')}, Size: {a.get('size')} bytes, State: {a.get('state')}")
            print(f"    Uploader: {a.get('uploader', {}).get('login')}, Created: {a.get('created_at')}")
            print(f"    Download URL: {a.get('browser_download_url')}")


if __name__ == "__main__":
    check_releases()
