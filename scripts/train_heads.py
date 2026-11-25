"""Train the two XGBoost heads from band-limited embeddings (see extract_features.py).

Usage:
    python scripts/train_heads.py features_bandlimited.pkl [model_dir]

- breathing_head.json : binary, breathing (normal or abnormal) vs no breathing. Class-balanced, breathing x3.
- disease_head.json   : 3-class [no-breathing, normal, abnormal]. Class-balanced, normal x3, abnormal x5.
Both: 100 trees, depth 4, learning rate 0.1 (the paper's settings).
"""
import os, pickle, sys

import numpy as np
import xgboost as xgb

pkl = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "model")
X, y, _ = pickle.load(open(pkl, "rb"))
X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
base = dict(max_depth=4, eta=0.1, tree_method="hist", seed=42, verbosity=0)


def balanced(labels, mults):
    w = np.empty(len(labels))
    k = len(mults)
    for c, m in enumerate(mults):
        w[labels == c] = len(labels) / (k * (labels == c).sum()) * m
    return w


yb = (y > 0).astype(int)
b = xgb.train({**base, "objective": "binary:logistic"}, xgb.DMatrix(X, label=yb, weight=balanced(yb, [1, 3])), 100)
b.save_model(os.path.join(out, "breathing_head.json"))
d = xgb.train({**base, "objective": "multi:softprob", "num_class": 3},
              xgb.DMatrix(X, label=y, weight=balanced(y, [1, 3, 5])), 100)
d.save_model(os.path.join(out, "disease_head.json"))
print(f"saved breathing_head.json and disease_head.json to {out} (n={len(y)})")
