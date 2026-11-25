# Pig respiratory audio triage (demo server)

A small Python server for the 晶創 platform: **upload a pig breathing recording → get back two results side by side**:

1. **Breathing**: is there clear pig breathing in the recording? *(reliable)*
2. **Disease**: does the breathing sound normal or abnormal? *(experimental, see the warning below)*

plus one combined traffic light (🟢 / 🟡 / 🔴) for a simple farmer view.
The web app (Firebase) calls this server over HTTP. The server holds the ML model, so the web app doesn't need Python.

> ⚠️ **The disease result is experimental, not a diagnosis.** Use wording like "abnormal-sounding breathing, please
> check / consult a vet". On 10 test clips from new recordings it flagged all 5 sick pigs but also 3 of 5 healthy
> pigs, and on our training data it can't yet separate normal from abnormal pigs recorded with the same device
> (details in §5). Breathing detection is not affected.

---

## 1. What it does

```
audio file (wav / mp3 / m4a / flac, any length, any sample rate)
  → mono, reduced to 4 kHz sampling (2 kHz bandwidth), cut into 10-second clips
  → fine-tuned Audio Spectrogram Transformer (AST)   → 768-number "embedding" per clip
  → breathing head (XGBoost)                        → P(breathing) per clip
  → disease head (XGBoost, experimental)            → P(abnormal) per clip
```

**Why 4 kHz?** Recordings come in at anything from 4 kHz (stethoscope) to 48 kHz (phones). In our training data the
abnormal recordings were mostly 16 kHz and the normal ones 4 kHz, so a model fed the full band learned to recognise the
*file format* (a sick pig recorded at 4 kHz came out green; the same file at 16 kHz came out red). Reducing every
input to the same 2 kHz bandwidth removes that shortcut, and the same recording now gets the same result whatever
format it arrives in. Breathing sounds sit well below 2 kHz.

### The two results

| Result | Rule | Values |
|---|---|---|
| **Breathing** | a clip is "breathing" if P(breathing) ≥ 0.70; the recording is `detected` if at least half its clips are | `detected` / `not_detected` |
| **Disease** *(experimental)* | mean P(abnormal) over the **breathing clips only** ≥ 0.25 → `abnormal` | `normal` / `abnormal` / `unknown` (no breathing clips) |

### Combined traffic light (`status`)

| Status | When | Suggested text for farmers |
|---|---|---|
| 🟡 `yellow` | breathing `not_detected` | No clear breathing recorded. Please re-record (closer, quieter) |
| 🔴 `red` | breathing detected **and** disease `abnormal` | Abnormal-sounding breathing, please check this pig |
| 🟢 `green` | breathing detected **and** disease `normal` | Breathing sounds normal |

For a comparison view (e.g. the researcher page), show `breathing` and `disease` separately rather than only `status`.
Every 10 s clip also gets its own `p_breathing` / `p_abnormal`, so the UI can show *where* in the recording things happen.
Thresholds live at the top of `app/triage.py`.

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
// res.breathing.status === "detected" | "not_detected"
// res.disease.status === "normal" | "abnormal" | "unknown"   (experimental)
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
  "status": "red",
  "breathing": {"status": "detected", "p_breathing": 0.9921, "breathing_clips": 2, "total_clips": 2},
  "disease":   {"status": "abnormal", "p_abnormal": 0.5027, "experimental": true},
  "probabilities": {"no_breathing": 0.0352, "normal": 0.4622, "abnormal": 0.5027},
  "duration_s": 12.0,
  "segments": [
    {"start_s": 0.0,  "end_s": 10.0, "breathing": true, "p_breathing": 0.9976, "p_abnormal": 0.9952,
     "probabilities": {"no_breathing": 0.002, "normal": 0.0028, "abnormal": 0.9952}},
    {"start_s": 10.0, "end_s": 12.0, "breathing": true, "p_breathing": 0.9867, "p_abnormal": 0.0101,
     "probabilities": {"no_breathing": 0.0684, "normal": 0.9216, "abnormal": 0.0101}}
  ],
  "processing_s": 1.4
}
```
Fields:
- `probabilities`: the disease head's 3-class output (no breathing / normal / abnormal), averaged over **all** clips
  (per clip in `segments`). Useful for charts and for the researcher view.
- `disease.p_abnormal`: P(abnormal) averaged over the **breathing clips only**, which is what the disease decision
  uses. It equals `probabilities.abnormal` when every clip contains breathing, and is higher when some clips are silent.
- `breathing.p_breathing`: from the separate breathing head, which is more reliable for "is there breathing" than
  `1 - probabilities.no_breathing`.

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
| `BREATHING_HEAD` | `model/breathing_head.json` | breathing classifier (included in this repo) |
| `DISEASE_HEAD` | `model/disease_head.json` | disease classifier (included in this repo) |
| `MAX_UPLOAD_MB` | `50` | Upload size limit |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set to the web app's domain in production |
| `PORT` | `8080` | (Docker only) listening port |

## 4. Speed and size

| | Size | Time per 10 s clip |
|---|---|---|
| AST (fine-tuned, 86 M parameters) | 345 MB | ~25 ms on GPU (RTX 4080), ~0.3–2 s on CPU |
| XGBoost heads (2) | 0.8 MB | < 1 ms |

A 1-minute recording takes ~4–12 s on a CPU-only container. A GPU isn't needed for the demo.

## 5. Where the model comes from

- **AST** ([willychenwii/pig-respiratory-ast](https://huggingface.co/willychenwii/pig-respiratory-ast)): `MIT/ast-finetuned-audioset-10-10-0.4593`, fine-tuned on stethoscope recordings of 16 pigs from a commercial
  farm (NPUST × NTU), checkpoint `20260209_n`. This is the model used in the paper *"Hybrid AST-XGBoost Architecture for
  Pig Respiratory Health Monitoring in Noisy Farm Environments"* (Chen, Huang, Xu, Jang; under review at *Computers and
  Electronics in Agriculture*). Code: <https://github.com/Willy-chen/pig-respiratory-monitor>.
- **Embedding** = mean over time of the average of the AST's last 3 hidden layers (768-d).
- **Heads**: two XGBoost classifiers trained on all 1,553 labelled 10 s clips (16 pigs), using **band-limited**
  embeddings (every clip reduced to 4 kHz first, as the server does). 100 trees, depth 4, learning rate 0.1.
  - `breathing_head.json`: breathing (normal or abnormal) vs no breathing; breathing class weight ×3.
  - `disease_head.json`: 3-class no-breathing / normal / abnormal; class weights normal ×3, abnormal ×5.
    The server uses its P(abnormal).
  - Re-create: `python scripts/extract_features.py segments.csv feats.pkl` then
    `python scripts/train_heads.py feats.pkl`. Needs the raw farm audio, which is not public (data-use agreement).
- **How well it works** (leave-one-pig-out: each pig scored by a model that never saw it):

  | | Result |
  |---|---|
  | Breathing vs no breathing | macro F1 **0.86**, AUC 0.92 (same as with full-band audio) |
  | Disease, normal vs abnormal pigs recorded with the **same** device | AUC 0.54, i.e. chance. Only 2 such abnormal pigs exist in the data |
  | Disease, 10 new clips (all 4 kHz) from Prof. Huang's lab | sick pigs: 5 / 5 abnormal. Healthy pigs: 3 / 5 wrongly abnormal |

  The paper's 3-class score (macro F1 0.871) used full-band audio and mostly reflects the recording-format difference
  described in §1, so it does not carry over to this server. The disease head needs normal and abnormal pigs
  recorded with the same device and procedure before it can be trusted.
- `transformers` is pinned to **4.45.0**: with it, this server reproduces the paper's embeddings exactly for 4 kHz recordings
  (cosine 1.000). Newer versions shift them slightly (cosine ≈ 0.98) and change the probabilities.

## 6. Known limitations

- **Disease result is experimental** (see §5). Breathing detection is the dependable part.
- Trained on **stethoscope** recordings. Phone-microphone or video audio is out of distribution, so expect more yellow
  results and less reliable colours. The 4 kHz reduction removes the file-format effect, but other differences between
  recording setups (background noise, gain, placement) can still influence the disease score.
- One recording = one pig. The server doesn't separate several pigs or detect which pig is which.
- No authentication: put it behind the web app (or Cloud Run IAM) before exposing it publicly.
- Sample audio is not included in this repo (farm data agreement). Use the files Prof. Huang provides. Any recording
  works for testing the plumbing.

## 7. Repository layout

```
app/main.py            HTTP API (FastAPI)
app/triage.py          model loading + audio → breathing / disease / traffic light (thresholds at the top)
model/breathing_head.json, model/disease_head.json   XGBoost heads (the AST weights come from Hugging Face)
scripts/download_model.py   download + checksum-verify the AST weights into model/ast/
scripts/extract_features.py band-limited AST embeddings from labelled clips (lab only: needs the raw audio)
scripts/train_heads.py      train both heads from those embeddings
examples/test_client.py  send files to the server from the command line
examples/sample_response.json
Dockerfile, requirements.txt
```

Questions: Willy (陳威宇), ML side.
