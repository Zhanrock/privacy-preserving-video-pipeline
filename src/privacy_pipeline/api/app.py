"""
api/app.py
-----------
FastAPI REST API for the Privacy-Preserving Video Pipeline.

Endpoints
---------
POST /process/frame    — Upload a single frame for anonymization
POST /process/batch    — Upload multiple frames
GET  /sessions/{id}    — Get session summary
GET  /sessions         — List recent sessions
GET  /audit            — Get recent audit trail
GET  /report/compliance — Generate compliance report
GET  /health           — Health check with DB connectivity test
DELETE /sessions/{id}  — GDPR right-to-erasure (soft-delete session data)

Install dependencies:
    pip install fastapi uvicorn python-multipart

Run:
    uvicorn privacy_pipeline.api.app:create_app --factory --reload --port 8000
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

# Guard: FastAPI may not be installed in environments that only run core pipeline
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

import cv2
import numpy as np

from privacy_pipeline.crypto.hasher import TrackingIDHasher
from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.database.repositories import AuditEvent, AuditRepository
from privacy_pipeline.pipeline.processor import PrivacyPipeline
from privacy_pipeline.pipeline.reporter import PipelineReporter
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


def create_app(db: Optional[DatabaseManager] = None) -> "FastAPI":
    """
    Application factory.

    Parameters
    ----------
    db: Pre-created DatabaseManager.  If None, creates a SQLite in-memory instance
        (for testing / quick-start).
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Run: pip install 'privacy-pipeline[api]'"
        )

    from fastapi import FastAPI

    _db = db or DatabaseManager.for_testing()
    _db.initialize_schema()
    reporter = PipelineReporter(_db)
    audit_repo = AuditRepository(_db)

    app = FastAPI(
        title       = "Privacy-Preserving Video Pipeline API",
        description = "Anonymize video frames and manage GDPR/PDPA audit logs",
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", tags=["System"])
    async def health_check():
        try:
            rows = _db.fetchall("SELECT 1 AS ok")
            db_ok = bool(rows)
        except Exception as exc:
            db_ok = False
            logger.error("DB health check failed: %s", exc)

        return {
            "status":    "healthy" if db_ok else "degraded",
            "database":  "ok" if db_ok else "error",
            "backend":   "mysql" if _db.is_mysql() else "sqlite",
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    @app.post("/process/frame", tags=["Processing"])
    async def process_frame(
        file:          UploadFile = File(..., description="Image file (JPEG/PNG)"),
        tracking_id:   Optional[str] = Form(None),
        source_label:  str = Form("api_upload"),
    ):
        """
        Anonymize a single video frame.

        Returns the detection metadata (anonymized frame is not returned in JSON
        to keep response lightweight — use a dedicated streaming endpoint for that).
        """
        raw = await file.read()
        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail      = "Could not decode image file.",
            )

        hasher   = TrackingIDHasher()   # ephemeral for this demo
        pipeline = PrivacyPipeline(
            db               = _db,
            hasher           = hasher,
            face_detector    = None,    # disabled for API (no cascade in container)
            plate_detector   = None,
            face_blurrer     = __import__(
                "privacy_pipeline.anonymizer.blurrer", fromlist=["RegionBlurrer"]
            ).RegionBlurrer(),
            plate_blurrer    = __import__(
                "privacy_pipeline.anonymizer.blurrer", fromlist=["RegionBlurrer"]
            ).RegionBlurrer(method="solid"),
            source_identifier = source_label,
        )

        result = pipeline.process_frame(frame, tracking_id=tracking_id)
        pipeline.close()

        return JSONResponse(content=result.to_dict())

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @app.get("/sessions", tags=["Sessions"])
    async def list_sessions(limit: int = 20):
        return reporter.all_sessions(limit=limit)

    @app.get("/sessions/{session_id}", tags=["Sessions"])
    async def get_session(session_id: str):
        summary = reporter.session_summary(session_id)
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        return summary

    @app.delete("/sessions/{session_id}", tags=["Sessions"], status_code=204)
    async def delete_session(session_id: str, actor: str = "api_user"):
        """
        GDPR Article 17 — Right to Erasure.
        Soft-deletes all detection events for a session and logs the action.
        """
        from privacy_pipeline.database.repositories import DetectionRepository
        det_repo = DetectionRepository(_db)
        count = det_repo.soft_delete_by_session(session_id)
        audit_repo.log_deletion(session_id, actor=actor, record_count=count)
        return None

    # ------------------------------------------------------------------
    # Audit & Compliance
    # ------------------------------------------------------------------

    @app.get("/audit", tags=["Compliance"])
    async def get_audit(limit: int = 100):
        """Return recent audit trail entries."""
        return audit_repo.get_recent(limit=limit)

    @app.get("/report/compliance", tags=["Compliance"])
    async def compliance_report(session_id: Optional[str] = None):
        """Generate a GDPR/PDPA compliance report."""
        report = reporter.compliance_report(session_id=session_id)
        # Strip verbose events list for the API response
        report.pop("events", None)
        return report

    return app


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _FASTAPI_AVAILABLE:
        print("Install: pip install fastapi uvicorn python-multipart")
    else:
        import uvicorn
        uvicorn.run(create_app, factory=True, host="0.0.0.0", port=8000, reload=True)
