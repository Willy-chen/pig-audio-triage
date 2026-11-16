"""Download the AST model weights from Hugging Face into model/ast/ and verify them.

Usage:
    python scripts/download_model.py            # -> model/ast/
    python scripts/download_model.py some/dir   # custom location

Then run the server with   AST_MODEL=model/ast   (or build Docker with
--build-arg AST_MODEL=/srv/model/ast) to use the local copy, e.g. without internet.
"""
import hashlib
import os
import sys

from huggingface_hub import snapshot_download

REPO = "willychenwii/pig-respiratory-ast"
SHA256 = "32d4b96d4741e88fd81748c6b62d84ac2af0a90dcd38e75024c20390e8354d45"  # model.safetensors (paper checkpoint)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "model", "ast")
out = os.path.abspath(out)
print(f"downloading {REPO} -> {out} (~345 MB)")
snapshot_download(REPO, local_dir=out,
                  allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])

h = hashlib.sha256()
with open(os.path.join(out, "model.safetensors"), "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != SHA256:
    sys.exit(f"checksum mismatch: got {h.hexdigest()}, expected {SHA256}")
print(f"OK: checksum verified. Use it with  AST_MODEL={out}")
