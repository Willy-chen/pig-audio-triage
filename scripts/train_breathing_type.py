"""4-class breathing type head: breathing / light breathing / heavy breathing / heavy breathing with noises.

Labels: Audacity strong labels (/home/willy/pig/strong_labels/<pig>.txt), 12 pigs, all 4 kHz stethoscope recordings.
Windows: 10 s (same as the server), hop 2 s, kept if >= 70 % of the window is covered by ONE breathing-type label.
Embeddings: band-limited exactly like the server (native -> 4 kHz -> 16 kHz), AST paper checkpoint, last-3-layer mean.
Evaluation: leave-one-pig-out, XGBoost (server settings), class-balanced weights.
Then trains the final head on everything -> breathing_type_head.json.
Lab script: the paths are local to the lab machine. See README §5.
"""
import glob, json, os, pickle, sys
import numpy as np, pandas as pd, librosa, torch, xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import ASTFeatureExtractor, ASTModel

OUT = os.path.dirname(os.path.abspath(__file__))
CK = "/home/willy/pig/pigcais/20260209_n/best_ast_model"
CLASSES = ["breathing", "light_breathing", "heavy_breathing", "heavy_breathing_with_noises"]
TAG = {"breathing": 0, "light breathing": 1, "heavy breathing": 2, "heavy breathing with noises": 3}
SR, WIN, HOP, COVER = 16000, 10.0, 2.0, 0.7

# ---------------------------------------------------------------- windows
rows = []
for txt in sorted(glob.glob("/home/willy/pig/strong_labels/2024*.txt") + glob.glob("/home/willy/pig/strong_labels/2025*.txt")):
    pig = os.path.basename(txt)[:-4]
    wav = glob.glob(f"/home/willy/pig/audio/*/{pig}.wav")[0]
    ev = []
    for line in open(txt, encoding="utf-8", errors="ignore"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and " ".join(p[2].lower().split()) in TAG:
            ev.append((float(p[0]), float(p[1]), TAG[" ".join(p[2].lower().split())]))
    dur = librosa.get_duration(path=wav)
    for t in np.arange(0, dur - WIN + 1e-6, HOP):
        cov = np.zeros(4)
        for a, b, c in ev:
            cov[c] += max(0.0, min(b, t + WIN) - max(a, t))
        if cov.max() >= COVER * WIN:
            rows.append(dict(pig=pig, wav=wav, start=float(t), label=int(cov.argmax())))
W = pd.DataFrame(rows)
print(pd.crosstab(W.pig, W.label.map(dict(enumerate(CLASSES))), margins=True), flush=True)

# ---------------------------------------------------------------- embeddings
cache = f"{OUT}/features.pkl"
if os.path.exists(cache):
    X = pickle.load(open(cache, "rb"))
else:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = ASTFeatureExtractor.from_pretrained(CK); ast = ASTModel.from_pretrained(CK).to(dev).eval()
    X = np.zeros((len(W), 768), np.float32)
    for wav, idx in W.groupby("wav").indices.items():
        a, _ = librosa.load(wav, sr=4000)
        a = librosa.resample(a, orig_sr=4000, target_sr=SR)
        segs = [a[int(s * SR):int(s * SR) + int(WIN * SR)] for s in W.start.values[idx]]
        with torch.no_grad():
            for k in range(0, len(segs), 32):
                x = fe([np.pad(s, (0, int(WIN * SR) - len(s))) for s in segs[k:k + 32]],
                       sampling_rate=SR, return_tensors="pt").input_values.to(dev)
                hs = ast(x, output_hidden_states=True).hidden_states
                X[idx[k:k + 32]] = torch.stack(hs[-3:]).mean(0).mean(1).cpu().numpy()
        print("  embedded", os.path.basename(wav), len(idx), flush=True)
    pickle.dump(X, open(cache, "wb"))
y, g = W.label.values, W.pig.values

# ---------------------------------------------------------------- LOPO
XP = dict(objective="multi:softprob", num_class=4, max_depth=4, eta=0.1, tree_method="hist", seed=42, verbosity=0)
def fit(Xt, yt):
    w = np.ones(len(yt))
    for c in np.unique(yt):
        w[yt == c] = len(yt) / (len(np.unique(yt)) * (yt == c).sum())
    return xgb.train(XP, xgb.DMatrix(Xt, label=yt, weight=w), 100)

P = np.zeros((len(y), 4))
for pig in np.unique(g):
    te = g == pig
    P[te] = fit(X[~te], y[~te]).predict(xgb.DMatrix(X[te]))
pred = P.argmax(1)
rep = classification_report(y, pred, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0)
print(classification_report(y, pred, labels=range(4), target_names=CLASSES, zero_division=0))
cm = confusion_matrix(y, pred, labels=range(4))
print("confusion (rows = label, cols = predicted)\n", pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
# per-pig accuracy: does it generalise across pigs, or does it learn the pig?
perpig = {p: float((pred[g == p] == y[g == p]).mean()) for p in np.unique(g)}
print("per-pig accuracy:", {k: round(v, 2) for k, v in perpig.items()})
# merged 3-class view: light / normal / heavy (heavy + heavy-with-noises)
m3 = np.array([1, 0, 2, 2])
print("3-class (light / breathing / heavy+heavy-noisy) macro F1:", round(f1_score(m3[y], m3[pred], average="macro"), 3))

json.dump({"n_windows": int(len(y)), "per_class_windows": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
           "pigs_per_class": {c: int(W[W.label == i].pig.nunique()) for i, c in enumerate(CLASSES)},
           "lopo_report": rep, "confusion": cm.tolist(), "per_pig_accuracy": perpig,
           "macro_f1_3class_merged": float(f1_score(m3[y], m3[pred], average="macro"))},
          open(f"{OUT}/results.json", "w"), indent=1)
fit(X, y).save_model(f"{OUT}/breathing_type_head.json")
print("saved", f"{OUT}/breathing_type_head.json")
