"""
pipeline/reporter.py
---------------------
Generates compliance and operational reports from MySQL data.

Reports
-------
SessionSummaryReport   — Per-session frame counts, latency, detection stats
DailySummaryReport     — Aggregate daily detection metrics (last N days)
ComplianceAuditReport  — Full GDPR/PDPA audit trail export
LatencyReport          — Latency distribution and SLA adherence metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.database.repositories import (
    AuditRepository, DetectionRepository, SessionRepository,
)
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineReporter:
    """
    Generates operational and compliance reports from the MySQL audit database.

    Parameters
    ----------
    db: DatabaseManager instance (MySQL or SQLite).
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db         = db
        self._sessions   = SessionRepository(db)
        self._detections = DetectionRepository(db)
        self._audit      = AuditRepository(db)

    # ------------------------------------------------------------------
    # Session report
    # ------------------------------------------------------------------

    def session_summary(self, session_id: str) -> Dict[str, Any]:
        """Return a complete summary for a single session."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        stats = self._detections.get_session_summary(session_id)
        audit = self._audit.get_by_session(session_id)

        return {
            "session_id":         session_id,
            "source":             session.get("source_identifier"),
            "started_at":         session.get("started_at"),
            "ended_at":           session.get("ended_at"),
            "status":             session.get("status"),
            "total_frames":       stats.get("total_frames", 0),
            "faces_anonymized":   stats.get("total_faces", 0),
            "plates_anonymized":  stats.get("total_plates", 0),
            "mean_latency_ms":    round(float(stats.get("mean_latency_ms") or 0), 3),
            "max_latency_ms":     round(float(stats.get("max_latency_ms") or 0), 3),
            "audit_event_count":  len(audit),
            "compliance_tags":    list({e.get("compliance_tag") for e in audit
                                        if e.get("compliance_tag")}),
        }

    def all_sessions(self, limit: int = 50) -> List[Dict]:
        """Return summary of the most recent sessions."""
        return self._sessions.list_recent(limit=limit)

    # ------------------------------------------------------------------
    # Daily aggregate
    # ------------------------------------------------------------------

    def daily_summary(self, days: int = 7) -> List[Dict]:
        """Daily detection aggregate for the last N days."""
        return self._detections.get_daily_summary(days=days)

    # ------------------------------------------------------------------
    # Compliance / Audit report
    # ------------------------------------------------------------------

    def compliance_report(
        self,
        session_id: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a GDPR/PDPA compliance report.

        Parameters
        ----------
        session_id:  Scope to a single session, or None for all recent.
        output_path: If given, save report as JSON to this path.

        Returns
        -------
        Dict with compliance summary.
        """
        if session_id:
            events = self._audit.get_by_session(session_id)
        else:
            events = self._audit.get_recent(limit=5000)

        report = {
            "generated_at":       datetime.utcnow().isoformat(),
            "session_filter":     session_id,
            "total_events":       len(events),
            "event_type_counts":  {},
            "compliance_coverage": {},
            "gdpr_article_30_met": False,
            "events":             events,
        }

        # Count event types
        for ev in events:
            etype = ev.get("event_type", "unknown")
            report["event_type_counts"][etype] = (
                report["event_type_counts"].get(etype, 0) + 1
            )
            tag = ev.get("compliance_tag")
            if tag:
                report["compliance_coverage"][tag] = (
                    report["compliance_coverage"].get(tag, 0) + 1
                )

        # GDPR Article 30: processing activity log required
        report["gdpr_article_30_met"] = "GDPR-A30" in report["compliance_coverage"]

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Compliance report saved: %s", output_path)

        return report

    # ------------------------------------------------------------------
    # Pandas DataFrame reports
    # ------------------------------------------------------------------

    def detection_dataframe(self, session_id: str):
        """Return detection events as a pandas DataFrame."""
        return self._detections.to_dataframe(session_id)

    def audit_dataframe(self, session_id: Optional[str] = None):
        """Return audit trail as a pandas DataFrame."""
        return self._audit.to_dataframe(session_id)

    def print_session_summary(self, session_id: str) -> None:
        """Print a formatted session summary to stdout."""
        s  = self.session_summary(session_id)
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  SESSION REPORT — {s.get('session_id', '')[:16]}…")
        print(sep)
        print(f"  Source         : {s.get('source')}")
        print(f"  Status         : {s.get('status')}")
        print(f"  Started        : {s.get('started_at')}")
        print(f"  Ended          : {s.get('ended_at')}")
        print(f"  Frames         : {s.get('total_frames')}")
        print(f"  Faces blurred  : {s.get('faces_anonymized')}")
        print(f"  Plates masked  : {s.get('plates_anonymized')}")
        print(f"  Mean latency   : {s.get('mean_latency_ms')} ms")
        print(f"  Max latency    : {s.get('max_latency_ms')} ms")
        print(f"  Audit events   : {s.get('audit_event_count')}")
        print(f"  GDPR A30 met   : {s.get('compliance_coverage', {}).get('GDPR-A30', 0) > 0}")
        print(sep)
