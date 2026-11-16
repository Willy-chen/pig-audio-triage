"""Core inference: audio file -> AST embedding per 10 s clip -> XGBoost -> traffic light."""
import os

import librosa
import numpy as np
import torch
import xgboost as xgb
from transformers import ASTFeatureExtractor, ASTModel

SR = 16000
CLIP_S = 10.0
MIN_TAIL_S = 2.0          # a trailing piece shorter than this is dropped (unless it is the only one)
T_ABNORMAL = 0.25         # red   if P(abnormal) >= 0.25
T_NORMAL = 0.70           # green if P(normal)   >= 0.70, otherwise yellow
CLASSES = ("no_breathing", "normal", "abnormal")

AST_MODEL = os.environ.get("AST_MODEL", "willychenwii/pig-respiratory-ast")
XGB_HEAD = os.environ.get("XGB_HEAD", os.path.join(os.path.dirname(__file__), "..", "model", "xgb_head.json"))


def light(p):
    """Map [p_no_breathing, p_normal, p_abnormal] to a status (abnormal checked first)."""
    if p[2] >= T_ABNORMAL:
        return "red"
    if p[1] >= T_NORMAL:
        return "green"
    return "yellow"


class Triage:
    def __init__(self, ast_model=AST_MODEL, xgb_head=XGB_HEAD):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = ASTFeatureExtractor.from_pretrained(ast_model)
        self.ast = ASTModel.from_pretrained(ast_model).to(self.device).eval()
        self.head = xgb.Booster()
        self.head.load_model(xgb_head)
        self.ast_model = ast_model

    def clips(self, y):
        n = int(CLIP_S * SR)
        out = []
        for start in range(0, max(len(y), 1), n):
            seg = y[start:start + n]
            if len(seg) < MIN_TAIL_S * SR and out:
                break
            out.append((start / SR, np.pad(seg, (0, n - len(seg)))))
        return out

    @torch.no_grad()
    def embed(self, seg):
        x = self.fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        hs = self.ast(x, output_hidden_states=True).hidden_states
        return torch.stack(hs[-3:]).mean(0).mean(1).cpu().numpy().squeeze()   # last-3-layer mean, 768-d

    def analyze(self, path):
        y, _ = librosa.load(path, sr=SR, mono=True)
        duration = len(y) / SR
        clips = self.clips(y)
        X = np.stack([self.embed(seg) for _, seg in clips])
        P = self.head.predict(xgb.DMatrix(X))
        mean = P.mean(0)
        return {
            "status": light(mean),
            "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, mean)},
            "duration_s": round(duration, 2),
            "segments": [
                {"start_s": round(t, 2), "end_s": round(min(t + CLIP_S, duration), 2),
                 "status": light(p),
                 "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p)}}
                for (t, _), p in zip(clips, P)
            ],
        }
