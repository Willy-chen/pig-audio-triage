"""Extract band-limited AST embeddings for training the heads.

Usage:
    python scripts/extract_features.py segments.csv features_bandlimited.pkl [--ast willychenwii/pig-respiratory-ast]

segments.csv needs the columns  Audio_Path, Start, label, pig_id  (one row per 10 s clip;
label 0 = no breathing, 1 = normal, 2 = abnormal). The raw farm audio is not public (data-use agreement),
so this step can only be re-run by the lab. Output: pickle of (X[n,768], y, pig_ids).
"""
import argparse, csv, os, pickle, sys

import librosa
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from triage import BAND_SR, SR, Triage  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("segments")
ap.add_argument("out")
ap.add_argument("--ast", default=os.environ.get("AST_MODEL", "willychenwii/pig-respiratory-ast"))
a = ap.parse_args()

t = Triage.__new__(Triage)                      # only the AST part is needed
from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402
import torch  # noqa: E402
t.device = "cuda" if torch.cuda.is_available() else "cpu"
t.fe = ASTFeatureExtractor.from_pretrained(a.ast)
t.ast = ASTModel.from_pretrained(a.ast).to(t.device).eval()

rows = list(csv.DictReader(open(a.segments)))
X = []
for i, r in enumerate(rows):
    seg, _ = librosa.load(r["Audio_Path"], sr=BAND_SR, offset=float(r["Start"]), duration=10.0)
    seg = librosa.resample(seg, orig_sr=BAND_SR, target_sr=SR)
    X.append(t.embed(np.pad(seg, (0, max(0, 10 * SR - len(seg))))[:10 * SR]))
    if i % 100 == 0:
        print(f"{i}/{len(rows)}", flush=True)
pickle.dump((np.array(X), np.array([int(r["label"]) for r in rows]), np.array([r["pig_id"] for r in rows])),
            open(a.out, "wb"))
print("saved", a.out)
