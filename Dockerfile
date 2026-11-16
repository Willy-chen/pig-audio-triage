FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
# CPU-only torch keeps the image ~3 GB instead of ~8 GB
RUN pip install --no-cache-dir torch==2.5.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY model/ model/
# AST weights are downloaded from Hugging Face at build time so the container starts fast
ARG AST_MODEL=willychenwii/pig-respiratory-ast
ENV AST_MODEL=${AST_MODEL} HF_HOME=/srv/hf
RUN python -c "import os; from transformers import ASTModel, ASTFeatureExtractor as F; m=os.environ['AST_MODEL']; F.from_pretrained(m); ASTModel.from_pretrained(m)"
ENV PORT=8080 HF_HUB_OFFLINE=1
CMD cd app && uvicorn main:app --host 0.0.0.0 --port ${PORT}
