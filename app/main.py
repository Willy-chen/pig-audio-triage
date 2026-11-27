"""HTTP API for the pig respiratory audio triage demo."""
import os
import tempfile
import time

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from triage import Triage

MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024

app = FastAPI(title="Pig respiratory audio triage (demo)")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])
model = None


@app.on_event("startup")
def load():
    global model
    model = Triage()


@app.get("/health")
def health():
    return {"ok": model is not None, "device": model.device if model else None,
            "ast_model": model.ast_model if model else None}


def run(data: bytes, suffix: str):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        t0 = time.perf_counter()
        try:
            res = model.analyze(f.name)
        except Exception as e:  # unreadable / unsupported audio
            raise HTTPException(400, f"could not analyze audio ({type(e).__name__}: {e})")
    res["processing_s"] = round(time.perf_counter() - t0, 2)
    return res


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    return run(await file.read(), os.path.splitext(file.filename or "")[1] or ".wav")


class UrlReq(BaseModel):
    url: str


@app.post("/analyze_url")
def analyze_url(req: UrlReq):
    r = requests.get(req.url, timeout=60)
    if r.status_code != 200:
        raise HTTPException(400, f"download failed: HTTP {r.status_code}")
    suffix = os.path.splitext(req.url.split("?")[0])[1] or ".wav"
    return run(r.content, suffix)
