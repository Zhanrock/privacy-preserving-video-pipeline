# PrivGrad — Privacy-Preserving Video Analytics

> **NTU CCDS Research Project** | Assoc Prof Chee Wei Tan | June 2026
> Target: NSDI 2026 / USENIX Security 2026

[![CI](https://github.com/Taishanrock/privacy-preserving-video-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Taishanrock/privacy-preserving-video-pipeline/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**PrivGrad** reframes GDPR Article 30 audit logs — legally mandatory records of processing activities — as an *upstream optimization signal* for a TextGrad self-improvement loop.

Instead of treating compliance records as overhead, PrivGrad uses them as a continuous loss function: detection counts, blur parameters, privacy/utility scores, and GDPR gaps are summarized as natural language and fed into TextGrad (LLM-based automatic differentiation) to iteratively refine the pipeline configuration.

**Key contributions:**
- GDPR Art.30 audit log as a textual loss signal (not just compliance overhead)
- TextGrad upstream loop: LLM backpropagation over pipeline configuration
- SHA3-256 tamper-evident hash chaining on all audit entries
- NemoBot agentic integration: `privacyAnalyze` + `textgradOptimize` tools
- Three-era benchmark: Pre-AI (2018) → CNN Baseline (2022) → PrivGrad

---

## Architecture

```
Raw Video Frame
      │
      ▼
┌─────────────────────────────────────────┐
│  1. Face Detection (Haar → RetinaFace   │
│     → NVIDIA NIM via NemoBot)           │
│  2. License Plate Detection (MSER)      │
│  3. Privacy Transforms (Gaussian blur)  │
│  4. AES-256-GCM Metadata Encryption     │
│  5. GDPR Art.30 Audit Log Entry         │
│     (SHA3-256 hash chain)               │
└─────────────────────────────────────────┘
      │
      ▼  ← Textual feedback (natural language summary)
┌─────────────────────────────────────────┐
│  TextGrad Upstream Loop                 │
│  • Loss = audit log feedback            │
│  • Gradient = LLM critique              │
│  • Update = refined pipeline config     │
│  Backend: Claude (Haiku) or NemoBot     │
└─────────────────────────────────────────┘
      │
      ▼
Anonymized Frame + Updated Config + Compliance Report
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Taishanrock/privacy-preserving-video-pipeline.git
cd privacy-preserving-video-pipeline
pip install numpy pandas opencv-python cryptography pyyaml Pillow scipy textgrad anthropic
```

### 2. Run the TextGrad Optimization Loop

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
python scripts/textgrad_pipeline_v2.py --optimize
# Results saved to: outputs/textgrad_results.json
```

### 3. Run the Base Pipeline (no API key needed)

```bash
python scripts/run_pipeline.py --frames 100
```

### 4. NemoBot Backend (when available)

```bash
export NEMOBOT_ENDPOINT="http://your-nemobot-server/v1"
python scripts/textgrad_pipeline_v2.py --optimize
```

---

## Benchmark: Three Eras

| Era | Detection | Privacy | Utility | P×U | Optimization |
|-----|-----------|---------|---------|-----|--------------|
| Pre-AI (~2018) | Fixed Gaussian blur | 0.85 | 0.45 | 0.38 | None |
| CNN Baseline (~2022) | Haar Cascade + fixed blur | 0.72 | 0.78 | 0.56 | Manual tuning |
| **PrivGrad (this work)** | RetinaFace / NVIDIA NIM + LoRA | TBD | TBD | TBD | TextGrad upstream loop |

---

## Project Structure

```
privacy_pipeline/
├── src/privacy_pipeline/          # Core library
│   ├── anonymizer/                # FaceDetector, LicensePlateDetector, RegionBlurrer
│   ├── crypto/                    # AES-256-GCM encryption, HMAC-SHA256 hasher
│   ├── database/                  # MySQL + SQLite connection, repositories
│   ├── pipeline/                  # PrivacyPipeline orchestrator, reporter
│   ├── api/                       # FastAPI REST endpoints
│   └── utils/                     # Config loader, structured logger
├── scripts/
│   ├── textgrad_pipeline_v2.py    # PrivGrad TextGrad optimization loop (main)
│   ├── run_pipeline.py            # Base pipeline runner
│   ├── api.py                     # API server entry point
│   └── nemobot_tools.js           # NemoBot tool definitions
├── tests/
│   ├── unit/                      # Unit tests (62)
│   ├── integration/               # Integration tests (43)
│   ├── test_audit.py              # Audit log + hash chain demo
│   └── test_detection.py          # Detection comparison demo
├── data/
│   ├── test_images/               # Sample frames for testing
│   └── detection_comparison.csv   # Three-era benchmark data
├── models/
│   └── face_detection_yunet_2023mar.onnx
├── outputs/
│   └── textgrad_results.json      # TextGrad optimization results
├── logs/                          # GDPR Art.30 audit log files (JSONL)
├── overleaf_paper/
│   └── main.tex                   # Research paper (LaTeX, latest version)
├── configs/
│   └── default.yaml
├── migrations/
│   └── 001_initial_schema.sql
└── pyproject.toml
```

---

## GDPR Compliance

| Requirement | Implementation |
|-------------|----------------|
| **Article 5** — Data minimisation | Only hashes and counts stored; no PII |
| **Article 17** — Right to erasure | LoRA adapter deletion = erasure of fine-tuned recognition |
| **Article 25** — Privacy by Design | Anonymization applied before data leaves edge |
| **Article 30** — Records of processing | SHA3-256 tamper-evident audit chain; also serves as TextGrad loss |
| **Recital 26** — Pseudonymisation | HMAC-SHA256 with secret key; computationally irreversible |

---

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
pytest tests/ -v -m "not mysql"
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
