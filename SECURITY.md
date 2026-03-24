# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities by opening a **private security advisory**
on the GitHub repository (Security → Advisories → New draft advisory).
Do not open a public issue for security-sensitive findings.

---

## Dependency Vulnerability Status

The table below tracks known dependency vulnerabilities and their status in
AURORA-VISION.

### Fixed (patched versions already in `requirements.txt`)

| Package | Fixed version | Vulnerability |
|---|---|---|
| `torch` | 2.6.0 | RCE via `torch.load` with `weights_only=True` |
| `transformers` | 4.48.0 | Deserialization of untrusted data (3 CVEs) |
| `Pillow` | 12.1.1 | Out-of-bounds write loading PSD images |
| `aiohttp` | 3.13.3 | Zip bomb via HTTP `auto_decompress` |
| `yt-dlp` | 2026.2.21 | Arbitrary command injection + RCE via file-extension |
| `onnx` | 1.17.0 | Path traversal + arbitrary file overwrite |
| `nltk` | 3.9.3 | Zip Slip + unsafe deserialization |
| `black` | 26.3.1 | Arbitrary file writes from unsanitised cache file name |

---

### Residual — No upstream patch available

#### 1. `nltk` ≤ 3.9.3 — Unauthenticated remote shutdown (`nltk.app.wordnet_app`)

- **Advisory:** Unauthenticated HTTP POST to `nltk.app.wordnet_app` can shut
  down the server process.
- **Patched version:** Not available.
- **AURORA-VISION status:** ✅ **Not exploitable.**
  `nltk` is used exclusively for METEOR score tokenisation
  (`nltk.translate.meteor_score`). The vulnerable `nltk.app.wordnet_app`
  module is never imported or invoked anywhere in this codebase.
  Verify: `grep -r "wordnet_app\|nltk\.app" --include="*.py" .` returns
  nothing.
- **Mitigation:** Do not run `python -m nltk.app.wordnet_app` in any
  deployment or CI environment. The module is not required by AURORA-VISION.

---

#### 2. `onnx` ≤ 1.20.1 — Supply-chain attack via `onnx.hub.load(silent=True)`

- **Advisory:** `onnx.hub.load()` silently suppresses security warnings when
  `silent=True` is passed, allowing a malicious model repository to execute
  arbitrary code without user notification.
- **Patched version:** Not available.
- **AURORA-VISION status:** ✅ **Not exploitable.**
  AURORA-VISION produces ONNX files from local PyTorch weights using
  `torch.onnx.export()`. `onnx.hub.load()` is never called.
  Additionally, `deployment/onnx_exporter.py` contains a runtime guard that
  **replaces** `onnx.hub` with a stub that raises `RuntimeError` if
  `hub.load()` is ever called, preventing accidental future use.
- **Mitigation:** The guard in `deployment/onnx_exporter.py` enforces this
  invariant at runtime. Do not load ONNX models from untrusted remote
  repositories via `onnx.hub`.

---

#### 3. `wandb` ≤ 0.17.0 — SSRF (Withdrawn Advisory)

- **Advisory:** Server-Side Request Forgery in Weights & Biases.
- **Patched version:** Not available.
- **AURORA-VISION status:** ✅ **Not applicable.**
  This advisory has been **withdrawn** by the reporter and is considered
  a false positive. Furthermore, `wandb` is listed in `requirements.txt`
  as an optional experiment-tracking dependency but is **not imported** in
  any AURORA-VISION module.
  Verify: `grep -r "import wandb\|wandb\." --include="*.py" .` returns nothing.

---

## General Security Notes

- All model weights are loaded from **local files** or official HuggingFace
  model hubs over HTTPS. Never load weights from untrusted sources.
- API keys (`OPENAI_API_KEY`, `HUGGINGFACE_TOKEN`) must be stored in `.env`
  (excluded from git via `.gitignore`) and never committed.
- The FastAPI server (`api/server.py`) should be deployed behind a reverse
  proxy with TLS and should not be exposed directly to the internet without
  authentication.
- `torch.load()` calls use `weights_only=True` (enforced in `torch ≥ 2.6.0`)
  to prevent arbitrary code execution during checkpoint loading.
