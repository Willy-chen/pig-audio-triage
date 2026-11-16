"""Re-train the XGBoost head from the published AST embeddings.

Usage:
    python scripts/train_head.py path/to/features_3layer_mean.pkl [out.json]

The pickle is the public file from the paper's Zenodo record
(https://doi.org/10.5281/zenodo.20084289): a tuple (X[1553,768], y, pig_ids)
with y in {0: no-breathing, 1: normal, 2: abnormal}.

Settings match the paper's selected configuration: class weights balanced,
then normal x3 and abnormal x5; 100 trees, depth 4, learning rate 0.1.
"""
import pickle
import sys

import numpy as np
import xgboost as xgb

N_MULT, A_MULT = 3.0, 5.0


def main(pkl, out="model/xgb_head.json"):
    X, y, _ = pickle.load(open(pkl, "rb"))
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
    n = len(y)
    w = np.empty(n)
    for c, mult in ((0, 1.0), (1, N_MULT), (2, A_MULT)):
        w[y == c] = n / (3 * (y == c).sum()) * mult
    d = xgb.DMatrix(X, label=y, weight=w)
    params = dict(objective="multi:softprob", num_class=3, max_depth=4,
                  eta=0.1, tree_method="hist", seed=42, verbosity=0)
    bst = xgb.train(params, d, num_boost_round=100)
    bst.save_model(out)
    acc = (bst.predict(d).argmax(1) == y).mean()
    print(f"saved {out}  (n={n}, train accuracy {acc:.3f} - not a test score)")


if __name__ == "__main__":
    main(*sys.argv[1:])
