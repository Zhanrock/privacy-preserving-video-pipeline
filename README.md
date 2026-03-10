# 🔒 Privacy-Preserving Video Pipeline

> **High-speed, Privacy-by-Design video anonymization with AES-256-GCM encryption, HMAC-SHA256 pseudonymisation, and MySQL audit logging — GDPR & PDPA compliant.**

[![CI](https://github.com/YOUR_USERNAME/privacy-preserving-video-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/privacy-preserving-video-pipeline/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange)](https://mysql.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen)]()

---

## Overview

The **Privacy-Preserving Video Pipeline** automatically anonymizes sensitive data in video streams before forwarding to cloud analytics:

- **Face blurring** — Haar Cascade detection → Gaussian/pixelate/solid blur
- **License plate masking** — MSER detection → solid black fill
- **HMAC-SHA256 pseudonymisation** — tracking IDs hashed with a secret key; originals never stored
- **AES-256-GCM encryption** — metadata encrypted at rest with authenticated encryption
- **MySQL audit logging** — immutable GDPR Article 30 / PDPA compliance trail
- **Sub-30ms target latency** — optimized for real-time safety monitoring

Enables monitoring of **300M+ square metres** while complying with GDPR, PDPA, and equivalent data protection laws.

---

## Architecture

```
Raw Video Frame
      │
      ▼
┌─────────────────────────────────┐
│  1. SHA-256 Frame Hash          │  ← Deduplication / tamper detection
│  2. Face Detection (Haar)       │  ← OpenCV CascadeClassifier
│  3. License Plate Detection     │  ← MSER + geometric filtering
│  4. Region Anonymization        │  ← Pixelate / Gaussian / Solid fill
│  5. HMAC-SHA256 ID Hash         │  ← Pseudonymise tracking IDs
│  6. MySQL Audit Log             │  ← detection_events + audit_trail
└─────────────────────────────────┘
      │
      ▼
Anonymized Frame → Cloud Analytics ✓ (No PII)
```

---

## MySQL Schema

```sql
anonymization_sessions   — One row per processing session
detection_events         — Per-frame results (no PII; SHA-256 hashes only)
tracking_id_map          — HMAC pseudonyms only (original IDs never stored)
audit_trail              — Append-only GDPR Article 30 compliance log
```

**Views:**
- `v_session_stats` — Live session statistics join
- `v_daily_detection_summary` — Daily aggregate for dashboards

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/privacy-preserving-video-pipeline.git
cd privacy-preserving-video-pipeline

# Core (no MySQL connector needed for SQLite mode)
pip install numpy pandas opencv-python cryptography pyyaml Pillow scipy

# With MySQL connector
pip install mysql-connector-python

# With REST API
pip install fastapi uvicorn python-multipart
```

### 2. Set up MySQL

```bash
# Create database and user
mysql -u root -p << 'SQL'
CREATE DATABASE IF NOT EXISTS privacy_pipeline_db CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'pp_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX
    ON privacy_pipeline_db.* TO 'pp_user'@'localhost';
FLUSH PRIVILEGES;
SQL

# Run migration
mysql -u pp_user -p privacy_pipeline_db < migrations/001_initial_schema.sql
```

### 3. Configure

```bash
# Set secrets via environment (never in YAML)
export PPVP_DB_PASSWORD="your_mysql_password"
export PPVP_HMAC_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export PPVP_DB_HOST="localhost"
```

### 4. Run

```bash
# Demo with synthetic frames (SQLite, no MySQL needed)
python scripts/run_pipeline.py --frames 100

# With real MySQL
python scripts/run_pipeline.py --mysql --frames 100 --source CAM-ENTRANCE-01

# Process a folder of images
ppvp-process --source /path/to/images/ --tracking-id CAM-001

# Generate compliance report
ppvp-report --session-id <session-uuid> --output reports/gdpr_report.json
```

---

## Python API

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.crypto.hasher import TrackingIDHasher
from privacy_pipeline.anonymizer.detectors import FaceDetector, LicensePlateDetector
from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
from privacy_pipeline.pipeline.processor import PrivacyPipeline
from privacy_pipeline.pipeline.reporter import PipelineReporter

# Connect to MySQL (falls back to SQLite if unavailable)
db = DatabaseManager.for_testing()   # or: DatabaseManager.from_config(cfg)
db.initialize_schema()

# Build pipeline
pipeline = PrivacyPipeline(
    db             = db,
    hasher         = TrackingIDHasher(secret_key=b"your-32-byte-key-here!!!!!!!!!!!"),
    face_detector  = FaceDetector(),
    plate_detector = LicensePlateDetector(),
    face_blurrer   = RegionBlurrer(method="pixelate", strength=25),
    plate_blurrer  = RegionBlurrer(method="solid"),
    source_identifier = "CAM-ENTRANCE-01",
)

# Process frames
with pipeline:
    for frame in video_source:
        result = pipeline.process_frame(frame, tracking_id="PERSON-001")
        # result.anonymized_frame  → send to cloud
        # result.face_count, result.plate_count, result.processing_latency_ms

# Reporting
reporter = PipelineReporter(db)
reporter.print_session_summary(pipeline.session_id)
compliance = reporter.compliance_report(session_id=pipeline.session_id)
print(f"GDPR Article 30 compliant: {compliance['gdpr_article_30_met']}")
```

---

## Cryptography

### HMAC-SHA256 Pseudonymisation

```python
from privacy_pipeline.crypto.hasher import TrackingIDHasher

hasher = TrackingIDHasher(secret_key=b"your-32-byte-secret-key-here!!!!!")
pseudo = hasher.pseudonymise("CAM-007:PERSON-1234")
# → "3a7f92b1c4e8d0f6..." (64 hex chars)

# Verify without storing original
hasher.verify("CAM-007:PERSON-1234", pseudo)  # True
```

### AES-256-GCM Metadata Encryption

```python
from privacy_pipeline.crypto.encryptor import MetadataEncryptor

enc   = MetadataEncryptor.generate()
token = enc.encrypt({"camera_id": "CAM-001", "zone": "A"})
data  = enc.decrypt(token)   # {"camera_id": "CAM-001", "zone": "A"}
```

---

## GDPR Compliance

| Requirement | Implementation |
|-------------|---------------|
| **Article 5** — Data minimisation | Only hashes and counts stored; no PII |
| **Article 17** — Right to erasure | `soft_delete_by_session()` + `GDPR-A17` audit tag |
| **Article 25** — Privacy by Design | Anonymization applied before any data leaves edge |
| **Article 30** — Records of processing | Immutable `audit_trail` table; all sessions logged |
| **Recital 26** — Pseudonymisation | HMAC-SHA256 with secret key; computationally irreversible |

---

## Project Structure

```
privacy-preserving-video-pipeline/
├── src/privacy_pipeline/
│   ├── anonymizer/
│   │   ├── detectors.py    # FaceDetector (Haar), LicensePlateDetector (MSER)
│   │   └── blurrer.py      # RegionBlurrer: gaussian | pixelate | solid
│   ├── crypto/
│   │   ├── hasher.py       # TrackingIDHasher (HMAC-SHA256)
│   │   └── encryptor.py    # MetadataEncryptor (AES-256-GCM)
│   ├── database/
│   │   ├── connection.py   # DatabaseManager (MySQL + SQLite fallback)
│   │   ├── schema.py       # DDL for all 4 tables
│   │   └── repositories.py # Session, Detection, TrackingID, Audit repos
│   ├── pipeline/
│   │   ├── processor.py    # PrivacyPipeline orchestrator
│   │   └── reporter.py     # PipelineReporter (session & compliance reports)
│   ├── api/
│   │   └── app.py          # FastAPI REST endpoints
│   └── utils/
│       ├── config.py       # YAML loader + env var overrides
│       └── logger.py       # Structured logging
├── tests/
│   ├── unit/               # 62 unit tests
│   └── integration/        # 43 integration tests
├── migrations/
│   └── 001_initial_schema.sql   # Complete MySQL DDL + views
├── configs/
│   └── default.yaml
└── scripts/
    └── run_pipeline.py
```

---

## Running Tests

```bash
# All 105 tests (SQLite — no MySQL required)
python -m unittest discover -s tests -p "test_*.py" -v

# With pytest
pytest tests/ -v -m "not mysql"

# MySQL integration tests (requires live MySQL)
PPVP_DB_HOST=localhost PPVP_DB_PASSWORD=yourpw pytest tests/ -v -m mysql
```

---

## FastAPI REST API

```bash
pip install fastapi uvicorn python-multipart
uvicorn "privacy_pipeline.api.app:create_app" --factory --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process/frame` | POST | Anonymize a single image frame |
| `/sessions` | GET | List recent sessions |
| `/sessions/{id}` | GET | Session details |
| `/sessions/{id}` | DELETE | GDPR erasure |
| `/audit` | GET | Recent audit trail |
| `/report/compliance` | GET | GDPR compliance report |
| `/health` | GET | DB connectivity check |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
