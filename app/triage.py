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
BREATHING_TYPES = ("breathing", "light_breathing", "heavy_breathing", "heavy_breathing_with_noises")  # experimental

# breathing rate (experimental): autocorrelation of the loudness envelope, as in the lab's arm_a/breath_rate.py
ENV_HOP = 1024            # envelope frame = 64 ms at 16 kHz
RATE_MIN_BPM, RATE_MAX_BPM = 10, 100
RATE_MIN_S, RATE_MIN_REGULARITY = 20.0, 0.4   # below these the rate is returned but flagged reliable=false

HERE = os.path.dirname(os.path.abspath(__file__))
AST_MODEL = os.environ.get("AST_MODEL", "willychenwii/pig-respiratory-ast")
BREATHING_HEAD = os.environ.get("BREATHING_HEAD", os.path.join(HERE, "..", "model", "breathing_head.json"))
DISEASE_HEAD = os.environ.get("DISEASE_HEAD", os.path.join(HERE, "..", "model", "disease_head.json"))
BREATHING_TYPE_HEAD = os.environ.get("BREATHING_TYPE_HEAD", os.path.join(HERE, "..", "model", "breathing_type_head.json"))


def load_bandlimited(path):
    """Decode, mix to mono, resample to 4 kHz (anti-aliased), then up to 16 kHz for the AST.

    Recordings arrive at 4 kHz (stethoscope) up to 48 kHz (phones). Reducing all of them to the same 2 kHz bandwidth
    stops the model from reacting to the recording format instead of the pig."""
    y, _ = librosa.load(path, sr=BAND_SR, mono=True)
    return librosa.resample(y, orig_sr=BAND_SR, target_sr=SR)


def breathing_rate(y):
    """Breaths per minute from the loudness envelope (autocorrelation peak). Returns (bpm, regularity 0..1) or
    (None, None) if the recording is too short or has no clear rhythm. Not validated against counted breaths."""
    fps = SR / ENV_HOP
    e = librosa.feature.rms(y=y, frame_length=4 * ENV_HOP, hop_length=ENV_HOP)[0]
    e = e - e.mean()
    lo, hi = int(fps * 60 / RATE_MAX_BPM), int(fps * 60 / RATE_MIN_BPM)
    hi = min(hi, len(e) // 2)              # need at least two cycles
    if hi <= lo or e.std() < 1e-8:
        return None, None
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac /= ac[0] + 1e-12
    # strongest true local peak in the lag range (the edge of the range is not a peak: autocorrelation falls with lag)
    peaks = [k for k in range(max(lo, 1), hi) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return None, None
    k = max(peaks, key=lambda k: ac[k])
    return 60 * fps / k, float(ac[k])


class Triage:
    def __init__(self, ast_model=AST_MODEL, breathing_head=BREATHING_HEAD, disease_head=DISEASE_HEAD,
                 breathing_type_head=BREATHING_TYPE_HEAD):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = ASTFeatureExtractor.from_pretrained(ast_model)
        self.ast = ASTModel.from_pretrained(ast_model).to(self.device).eval()
        self.breathing_head = xgb.Booster(model_file=breathing_head)   # binary: P(breathing)
        self.disease_head = xgb.Booster(model_file=disease_head)       # 3-class: [no-breathing, normal, abnormal]
        self.breathing_type_head = xgb.Booster(model_file=breathing_type_head)  # 4-class effort level (experimental)
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
        P4 = self.breathing_type_head.predict(D)        # 4-class breathing type per clip
        is_breath = p_breath >= T_BREATHING

        detected = is_breath.mean() >= MIN_BREATHING_FRAC
        breathing = {                             # stage 1: how much of the recording is usable breathing
            "status": "detected" if detected else "not_detected",
            "p_breathing": round(float(p_breath.mean()), 4),
            "breathing_clips": int(is_breath.sum()),
            "total_clips": len(clips),
            "breathing_fraction": round(float(is_breath.mean()), 4),
            "noise_fraction": round(1 - float(is_breath.mean()), 4),
        }
        if is_breath.any():
            mean_abn = float(p_abn[is_breath].mean())
            disease = {"status": "abnormal" if mean_abn >= T_ABNORMAL else "normal",
                       "p_abnormal": round(mean_abn, 4)}
        else:
            disease = {"status": "unknown", "p_abnormal": None}
        disease["experimental"] = True

        if is_breath.any():                       # breathing type: averaged over breathing clips only
            m4 = P4[is_breath].mean(0)
            top = P4[is_breath].argmax(1)          # stage 2: share of breathing clips of each type
            breathing_type = {"type": BREATHING_TYPES[int(m4.argmax())],
                              "probabilities": {c: round(float(v), 4) for c, v in zip(BREATHING_TYPES, m4)},
                              "clip_fractions": {c: round(float((top == i).mean()), 4)
                                                 for i, c in enumerate(BREATHING_TYPES)}}
        else:
            breathing_type = {"type": None, "probabilities": None, "clip_fractions": None}
        breathing_type["experimental"] = True

        bpm, regularity = breathing_rate(y) if detected else (None, None)
        amp = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        respiration = {
            "rate_bpm": None if bpm is None else round(bpm, 1),
            "regularity": None if regularity is None else round(regularity, 3),
            "reliable": bool(bpm is not None and duration >= RATE_MIN_S and regularity >= RATE_MIN_REGULARITY),
            "amplitude_db": round(float(20 * np.log10(np.percentile(amp, 90) + 1e-9)), 1),
            "experimental": True,
        }

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
            "breathing_type": breathing_type,
            "respiration": respiration,
            "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, P3.mean(0))},
            "duration_s": round(duration, 2),
            "segments": [
                {"start_s": round(t, 2), "end_s": round(min(t + CLIP_S, duration), 2),
                 "breathing": bool(b), "p_breathing": round(float(pb), 4), "p_abnormal": round(float(pa), 4),
                 "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p3)},
                 "breathing_type": {c: round(float(v), 4) for c, v in zip(BREATHING_TYPES, p4)}}
                for (t, _), b, pb, pa, p3, p4 in zip(clips, is_breath, p_breath, p_abn, P3, P4)
            ],
        }
