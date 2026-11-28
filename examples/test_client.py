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
    b, d, bt = res["breathing"], res["disease"], res["breathing_type"]
    print(f"{ICON[res['status']]} {res['status']:6s} {path.split('/')[-1]}\n"
          f"     breathing: {b['status']} (P={b['p_breathing']}, {b['breathing_clips']}/{b['total_clips']} clips)  |  "
          f"disease [experimental]: {d['status']} (P(abnormal)={d['p_abnormal']})  |  "
          f"{res['duration_s']} s audio, {res['processing_s']} s\n"
          f"     breathing type [experimental]: {bt['type']}  clip fractions {bt['clip_fractions']}\n"
          f"     respiration [experimental]: {res['respiration']['rate_bpm']} breaths/min "
          f"(regularity {res['respiration']['regularity']}, reliable={res['respiration']['reliable']}), amplitude {res['respiration']['amplitude_db']} dB  |  "
          f"breathing {b['breathing_fraction']:.0%} / noise {b['noise_fraction']:.0%}")
    if a.full:
        print(json.dumps(res, indent=2, ensure_ascii=False))
