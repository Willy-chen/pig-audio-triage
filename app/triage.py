"""Core inference: audio file -> band-limited AST embedding per 10 s clip -> breathing + disease heads."""
import os

import librosa
import numpy as np
import torch
import xgboost as xgb
from transformers import ASTFeatureExtractor, ASTModel

SR = 16000
BAND_SR = 4000            # every input is reduced to 4 kHz sampling (= 2 kHz bandwidth) before the model
CLIP_S = 10.0
MIN_TAIL_S = 2.0          # a trailing piece shorter than this is dropped (unless it is the only one)

T_BREATHING = 0.70        # clip counts as "breathing" if P(breathing) >= 0.70
MIN_BREATHING_FRAC = 0.5  # recording counts as "breathing detected" if >= half of its clips are breathing
CLASSES = ("no_breathing", "normal", "abnormal")
T_ABNORMAL = 0.25         # disease (experimental): "abnormal" if mean P(abnormal) over breathing clips >= 0.25

HERE = os.path.dirname(os.path.abspath(__file__))
AST_MODEL = os.environ.get("AST_MODEL", "willychenwii/pig-respiratory-ast")
BREATHING_HEAD = os.environ.get("BREATHING_HEAD", os.path.join(HERE, "..", "model", "breathing_head.json"))
DISEASE_HEAD = os.environ.get("DISEASE_HEAD", os.path.join(HERE, "..", "model", "disease_head.json"))


def load_bandlimited(path):
    """Decode, mix to mono, resample to 4 kHz (anti-aliased), then up to 16 kHz for the AST.

    Recordings arrive at 4 kHz (stethoscope) up to 48 kHz (phones). Reducing all of them to the same 2 kHz bandwidth
    stops the model from reacting to the recording format instead of the pig."""
    y, _ = librosa.load(path, sr=BAND_SR, mono=True)
    return librosa.resample(y, orig_sr=BAND_SR, target_sr=SR)


class Triage:
    def __init__(self, ast_model=AST_MODEL, breathing_head=BREATHING_HEAD, disease_head=DISEASE_HEAD):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = ASTFeatureExtractor.from_pretrained(ast_model)
        self.ast = ASTModel.from_pretrained(ast_model).to(self.device).eval()
        self.breathing_head = xgb.Booster(model_file=breathing_head)   # binary: P(breathing)
        self.disease_head = xgb.Booster(model_file=disease_head)       # 3-class: [no-breathing, normal, abnormal]
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
        y = load_bandlimited(path)
        duration = len(y) / SR
        clips = self.clips(y)
        D = xgb.DMatrix(np.stack([self.embed(seg) for _, seg in clips]))
        p_breath = self.breathing_head.predict(D)
        P3 = self.disease_head.predict(D)               # 3-class probabilities per clip
        p_abn = P3[:, 2]
        is_breath = p_breath >= T_BREATHING

        detected = is_breath.mean() >= MIN_BREATHING_FRAC
        breathing = {
            "status": "detected" if detected else "not_detected",
            "p_breathing": round(float(p_breath.mean()), 4),
            "breathing_clips": int(is_breath.sum()),
            "total_clips": len(clips),
        }
        if is_breath.any():
            mean_abn = float(p_abn[is_breath].mean())
            disease = {"status": "abnormal" if mean_abn >= T_ABNORMAL else "normal",
                       "p_abnormal": round(mean_abn, 4)}
        else:
            disease = {"status": "unknown", "p_abnormal": None}
        disease["experimental"] = True

        if not detected:
            status = "yellow"                     # no clear breathing: re-record
        elif disease["status"] == "abnormal":
            status = "red"
        else:
            status = "green"

        return {
            "status": status,
            "breathing": breathing,
            "disease": disease,
            "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, P3.mean(0))},
            "duration_s": round(duration, 2),
            "segments": [
                {"start_s": round(t, 2), "end_s": round(min(t + CLIP_S, duration), 2),
                 "breathing": bool(b), "p_breathing": round(float(pb), 4), "p_abnormal": round(float(pa), 4),
                 "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p3)}}
                for (t, _), b, pb, pa, p3 in zip(clips, is_breath, p_breath, p_abn, P3)
            ],
        }
