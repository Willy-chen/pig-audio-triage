# Pig respiratory audio triage (demo server)

A small Python server for the 晶創 platform: **upload a pig breathing recording → get back 🟢 green / 🟡 yellow / 🔴 red**.
The web app (Firebase) calls this server over HTTP. The server holds the ML model, so the web app doesn't need Python.

> ⚠️ **Demo, not a diagnosis.** Red means "abnormal-sounding breathing, please check / consult a vet". The abnormal
> class was trained on a small dataset (16 pigs) where normal and abnormal recordings mostly came from different
> recording setups, so the model may partly be recognising the setup. Please word the farmer view accordingly.

---

## 1. What it does

```
audio file (wav / mp3 / m4a / flac, any length, any sample rate)
  → converted to 16 kHz mono, cut into 10-second clips
  → fine-tuned Audio Spectrogram Transformer (AST)   → 768-number "embedding" per clip
  → XGBoost classifier                              → P(no breathing), P(normal), P(abnormal) per clip
  → averaged over the whole recording               → traffic light
```

| Status | Rule (on the averaged probabilities) | Meaning for the user |
|---|---|---|
| 🔴 `red` | P(abnormal) ≥ 0.25 | Abnormal breathing sounds detected, please check this pig |
| 🟢 `green` | otherwise, if P(normal) ≥ 0.70 | Normal breathing |
| 🟡 `yellow` | anything else | No clear breathing / uncertain. Please re-record (closer, quieter) |

Red is checked first on purpose: missing a sick pig costs more than a false alarm. The thresholds are the ones chosen
in the paper's grid search and live at the top of `app/triage.py` (`T_ABNORMAL`, `T_NORMAL`).

Every clip also gets its own status, so the UI can show *where* in the recording the problem is.

## 2. API

Base URL: wherever the container runs (local default `http://localhost:8080`).

### `GET /health`
```json
{"ok": true, "device": "cpu", "ast_model": "willychenwii/pig-respiratory-ast"}
```

### `POST /analyze`: upload a file
`multipart/form-data` with one field, `file`.

```bash
curl -F "file=@recording.m4a" http://localhost:8080/analyze
```
```js
// browser / web app
const fd = new FormData();
fd.append("file", fileInput.files[0]);
const res = await fetch(`${API}/analyze`, { method: "POST", body: fd }).then(r => r.json());
// res.status === "green" | "yellow" | "red"
```

### `POST /analyze_url`: analyse a file that is already in storage
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"url": "https://firebasestorage.googleapis.com/v0/b/<bucket>/o/audio%2Fpig123.m4a?alt=media&token=..."}' \
     http://localhost:8080/analyze_url
```
Use this for **automatic results after upload**: a Cloud Function triggered when a new file lands in Firebase Storage
(`onObjectFinalized`) calls `/analyze_url` with a download URL and writes the JSON back to the pig's Firestore record.

### Response
Full example: [`examples/sample_response.json`](examples/sample_response.json).
```json
{
  "status": "green",
  "probabilities": {"no_breathing": 0.0011, "normal": 0.9988, "abnormal": 0.0001},
  "duration_s": 60.0,
  "segments": [
    {"start_s": 0.0, "end_s": 10.0, "status": "green",
     "probabilities": {"no_breathing": 0.0003, "normal": 0.9996, "abnormal": 0.0001}}
  ],
  "processing_s": 3.8
}
```
Errors: `400` (unreadable audio / download failed), `413` (file > 50 MB, change with `MAX_UPLOAD_MB`).

## 3. Run it

### Option A: Docker (recommended)
```bash
docker build -t pig-audio-triage .
docker run -p 8080:8080 pig-audio-triage
python examples/test_client.py my_recording.wav          # in another terminal
```
The build downloads the AST weights from Hugging Face once, so the container starts without internet.
Image size is about 3 GB (CPU-only PyTorch). To build from a local copy of the weights instead (e.g. behind a firewall):
```bash
python scripts/download_model.py                 # -> model/ast/ (checksum-verified)
docker build -t pig-audio-triage --build-arg AST_MODEL=/srv/model/ast .
```

### Option B: plain Python (3.10+)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt
python scripts/download_model.py                 # downloads the AST weights (~345 MB) to model/ast/ and verifies them
cd app && AST_MODEL=../model/ast uvicorn main:app --port 8080
```

### Cloud deployment
Any container host works. Suggested: **Google Cloud Run** (same Google account as Firebase, scales to zero when idle):
```bash
gcloud run deploy pig-audio-triage --source . --region asia-east1 --memory 4Gi --cpu 2 --timeout 600
```
A Firebase Cloud Function is **not** a good fit: the model is ~345 MB and needs PyTorch.

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `AST_MODEL` | `willychenwii/pig-respiratory-ast` | Hugging Face repo id or local folder with the AST weights |
| `XGB_HEAD` | `model/xgb_head.json` | XGBoost classifier file (included in this repo) |
| `MAX_UPLOAD_MB` | `50` | Upload size limit |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set to the web app's domain in production |
| `PORT` | `8080` | (Docker only) listening port |

## 4. Speed and size

| | Size | Time per 10 s clip |
|---|---|---|
| AST (fine-tuned, 86 M parameters) | 345 MB | ~25 ms on GPU (RTX 4080), ~0.3–2 s on CPU |
| XGBoost head | 0.5 MB | < 1 ms |

A 1-minute recording takes ~4–12 s on a CPU-only container. A GPU isn't needed for the demo.

## 5. Where the model comes from

- **AST** ([willychenwii/pig-respiratory-ast](https://huggingface.co/willychenwii/pig-respiratory-ast)): `MIT/ast-finetuned-audioset-10-10-0.4593`, fine-tuned on stethoscope recordings of 16 pigs from a commercial
  farm (NPUST × NTU), checkpoint `20260209_n`. This is the model used in the paper *"Hybrid AST-XGBoost Architecture for
  Pig Respiratory Health Monitoring in Noisy Farm Environments"* (Chen, Huang, Xu, Jang; under review at *Computers and
  Electronics in Agriculture*). Code: <https://github.com/Willy-chen/pig-respiratory-monitor>.
- **Embedding** = mean over time of the average of the AST's last 3 hidden layers (768-d).
- **XGBoost head** (`model/xgb_head.json`) is trained on all 1,553 labelled 10 s clips with the paper's selected
  settings (class weights: normal ×3, abnormal ×5; 100 trees, depth 4). Re-create it with
  `python scripts/train_head.py features_3layer_mean.pkl` (the features are public on
  [Zenodo](https://doi.org/10.5281/zenodo.20084289)).
- **Accuracy** (paper, nested leave-one-pig-out, 16 pigs): 3-class macro F1 **0.871** (95% CI 0.774–0.909).
  Breathing-vs-no-breathing alone is the more reliable part. Normal-vs-abnormal is the part with the recording-setup
  caveat above.
- `transformers` is pinned to **4.45.0**: with it, this server reproduces the paper's embeddings exactly
  (cosine 1.000). Newer versions shift them slightly (cosine ≈ 0.98) and change the probabilities.

## 6. Known limitations

- Trained on **stethoscope** recordings. Phone-microphone or video audio is out of distribution, so expect more yellow
  results and less reliable colours.
- One recording = one pig. The server doesn't separate several pigs or detect which pig is which.
- No authentication: put it behind the web app (or Cloud Run IAM) before exposing it publicly.
- Sample audio is not included in this repo (farm data agreement). Use the files Prof. Huang provides. Any recording
  works for testing the plumbing.

## 7. Repository layout

```
app/main.py            HTTP API (FastAPI)
app/triage.py          model loading + audio → traffic light (thresholds at the top)
model/xgb_head.json    XGBoost classifier (the AST weights are downloaded from Hugging Face)
scripts/download_model.py  download + checksum-verify the AST weights into model/ast/
scripts/train_head.py  re-train the XGBoost head from the published embeddings
examples/test_client.py  send files to the server from the command line
examples/sample_response.json
Dockerfile, requirements.txt
```

Questions: Willy (陳威宇), ML side.
