"""Send audio files to the triage server and print the result.

Usage:
    python examples/test_client.py file1.wav [file2.m4a ...] [--url http://localhost:8080]
"""
import argparse
import json

import requests

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--url", default="http://localhost:8080")
ap.add_argument("--full", action="store_true", help="print the full JSON incl. per-segment results")
a = ap.parse_args()

ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
for path in a.files:
    with open(path, "rb") as f:
        r = requests.post(f"{a.url}/analyze", files={"file": (path.split("/")[-1], f)}, timeout=600)
    if r.status_code != 200:
        print(f"{path}: HTTP {r.status_code} {r.text}")
        continue
    res = r.json()
    print(f"{ICON[res['status']]} {res['status']:6s} {path}  {res['probabilities']}  "
          f"({res['duration_s']} s audio, {len(res['segments'])} clips, {res['processing_s']} s)")
    if a.full:
        print(json.dumps(res, indent=2, ensure_ascii=False))
