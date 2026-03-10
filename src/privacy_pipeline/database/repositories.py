"""
database/repositories.py
-------------------------
Repository layer — all SQL queries in one place.
Business logic never writes raw SQL; it calls these methods.

Repositories
------------
SessionRepository      — anonymization_sessions table
DetectionRepository    — detection_events table
TrackingIDRepository   — tracking_id_map table
AuditRepository        — audit_trail table (append-only)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

_NOW = lambda: datetime.utcnow().isoformat(sep=" ", timespec="microseconds")


# ---------------------------------------------------------------------------
# Data classes (row DTOs)
# ---------------------------------------------------------------------------

@dataclass
class DetectionEvent:
    session_id:            str
    frame_sequence:        int
    captured_at:           str
    frame_hash_sha256:     str
    face_count:            int
    plate_count:           int
    processing_latency_ms: float
    anonymization_applied: bool = True
    detection_meta:        Optional[Dict] = None
    processed_at:          str = field(default_factory=_NOW)


@dataclass
class AuditEvent:
    event_type:    str
    action:        str
    actor:         str = "system"
    session_id:    Optional[str] = None
    resource_type: Optional[str] = None
    resource_id:   Optional[str] = None
    outcome:       str = "success"
    detail:        Optional[str] = None
    ip_address:    Optional[str] = None
    compliance_tag: Optional[str] = None
    event_id:      str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at:   str = field(default_factory=_NOW)


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------

class SessionRepository:

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def create(
        self,
        session_id: str,
        source_identifier: str,
        config_snapshot: Optional[Dict] = None,
    ) -> str:
        sql = """
            INSERT INTO anonymization_sessions
              (session_id, source_identifier, config_snapshot, started_at)
            VALUES (?, ?, ?, ?)
        """
        self._db.execute(
            sql,
            (
                session_id,
                source_identifier,
                json.dumps(config_snapshot) if config_snapshot else None,
                _NOW(),
            ),
        )
        logger.debug("Session created: %s", session_id)
        return session_id

    def update_stats(
        self,
        session_id: str,
        total_frames: int,
        frames_with_faces: int,
        frames_with_plates: int,
        mean_latency_ms: float,
        p99_latency_ms: float,
    ) -> None:
        sql = """
            UPDATE anonymization_sessions
            SET total_frames       = ?,
                frames_with_faces  = ?,
                frames_with_plates = ?,
                mean_latency_ms    = ?,
                p99_latency_ms     = ?
            WHERE session_id = ?
        """
        self._db.execute(
            sql,
            (
                total_frames, frames_with_faces, frames_with_plates,
                mean_latency_ms, p99_latency_ms, session_id,
            ),
        )

    def close_session(self, session_id: str, status: str = "completed") -> None:
        sql = """
            UPDATE anonymization_sessions
            SET ended_at = ?, status = ?
            WHERE session_id = ?
        """
        self._db.execute(sql, (_NOW(), status, session_id))
        logger.debug("Session closed: %s (%s)", session_id, status)

    def get(self, session_id: str) -> Optional[Dict]:
        return self._db.fetchone(
            "SELECT * FROM anonymization_sessions WHERE session_id = ?",
            (session_id,),
        )

    def list_recent(self, limit: int = 50) -> List[Dict]:
        return self._db.fetchall(
            "SELECT * FROM anonymization_sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )


# ---------------------------------------------------------------------------
# DetectionRepository
# ---------------------------------------------------------------------------

class DetectionRepository:

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def insert(self, event: DetectionEvent) -> None:
        sql = """
            INSERT INTO detection_events
              (session_id, frame_sequence, captured_at, processed_at,
               frame_hash_sha256, face_count, plate_count,
               processing_latency_ms, anonymization_applied, detection_meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._db.execute(
            sql,
            (
                event.session_id,
                event.frame_sequence,
                event.captured_at,
                event.processed_at,
                event.frame_hash_sha256,
                event.face_count,
                event.plate_count,
                event.processing_latency_ms,
                int(event.anonymization_applied),
                json.dumps(event.detection_meta) if event.detection_meta else None,
            ),
        )

    def bulk_insert(self, events: List[DetectionEvent]) -> None:
        """Batch insert for high-throughput scenarios."""
        sql = """
            INSERT INTO detection_events
              (session_id, frame_sequence, captured_at, processed_at,
               frame_hash_sha256, face_count, plate_count,
               processing_latency_ms, anonymization_applied, detection_meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (
                e.session_id, e.frame_sequence, e.captured_at, e.processed_at,
                e.frame_hash_sha256, e.face_count, e.plate_count,
                e.processing_latency_ms, int(e.anonymization_applied),
                json.dumps(e.detection_meta) if e.detection_meta else None,
            )
            for e in events
        ]
        self._db.executemany(sql, data)
        logger.debug("Bulk inserted %d detection events", len(events))

    def get_by_session(self, session_id: str, limit: int = 1000) -> List[Dict]:
        return self._db.fetchall(
            """SELECT * FROM detection_events
               WHERE session_id = ? AND deleted_at IS NULL
               ORDER BY frame_sequence
               LIMIT ?""",
            (session_id, limit),
        )

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        row = self._db.fetchone(
            """SELECT
                COUNT(*)         AS total_frames,
                SUM(face_count)  AS total_faces,
                SUM(plate_count) AS total_plates,
                AVG(processing_latency_ms) AS mean_latency_ms,
                MAX(processing_latency_ms) AS max_latency_ms,
                MIN(processing_latency_ms) AS min_latency_ms
               FROM detection_events
               WHERE session_id = ? AND deleted_at IS NULL""",
            (session_id,),
        )
        return row or {}

    def get_daily_summary(self, days: int = 7) -> List[Dict]:
        """Aggregate detections per day (last N days)."""
        return self._db.fetchall(
            """SELECT
                DATE(captured_at)  AS detection_date,
                COUNT(*)           AS total_frames,
                SUM(face_count)    AS total_faces,
                SUM(plate_count)   AS total_plates,
                AVG(processing_latency_ms) AS mean_latency_ms
               FROM detection_events
               WHERE deleted_at IS NULL
                 AND captured_at >= DATETIME('now', ? || ' days')
               GROUP BY DATE(captured_at)
               ORDER BY detection_date DESC""",
            (f"-{days}",),
        )

    def soft_delete_by_session(self, session_id: str) -> int:
        """GDPR right-to-erasure: soft-delete all records for a session."""
        cur = self._db.execute(
            "UPDATE detection_events SET deleted_at = ? WHERE session_id = ?",
            (_NOW(), session_id),
        )
        count = cur.rowcount
        logger.info("Soft-deleted %d detection events for session %s", count, session_id)
        return count

    def to_dataframe(self, session_id: str):
        """Return detection events as a pandas DataFrame for analysis/reporting."""
        import pandas as pd
        rows = self.get_by_session(session_id)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["captured_at"]  = pd.to_datetime(df["captured_at"])
        df["processed_at"] = pd.to_datetime(df["processed_at"])
        return df


# ---------------------------------------------------------------------------
# TrackingIDRepository
# ---------------------------------------------------------------------------

class TrackingIDRepository:
    """
    Manages pseudonymised tracking IDs.
    ONLY the HMAC hash is stored — never the original ID.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def upsert(self, pseudonym_hash: str, algorithm: str = "sha256") -> None:
        """Insert or update last_seen_at and increment session_count."""
        existing = self._db.fetchone(
            "SELECT id FROM tracking_id_map WHERE pseudonym_hash = ?",
            (pseudonym_hash,),
        )
        if existing:
            self._db.execute(
                """UPDATE tracking_id_map
                   SET last_seen_at = ?, session_count = session_count + 1
                   WHERE pseudonym_hash = ?""",
                (_NOW(), pseudonym_hash),
            )
        else:
            self._db.execute(
                """INSERT INTO tracking_id_map
                   (pseudonym_hash, hmac_algorithm, first_seen_at, last_seen_at)
                   VALUES (?, ?, ?, ?)""",
                (pseudonym_hash, algorithm, _NOW(), _NOW()),
            )

    def exists(self, pseudonym_hash: str) -> bool:
        row = self._db.fetchone(
            "SELECT 1 FROM tracking_id_map WHERE pseudonym_hash = ?",
            (pseudonym_hash,),
        )
        return row is not None

    def deactivate(self, pseudonym_hash: str) -> None:
        """Right-to-be-forgotten: deactivate a pseudonymised ID."""
        self._db.execute(
            "UPDATE tracking_id_map SET is_active = 0 WHERE pseudonym_hash = ?",
            (pseudonym_hash,),
        )

    def count_active(self) -> int:
        row = self._db.fetchone(
            "SELECT COUNT(*) AS cnt FROM tracking_id_map WHERE is_active = 1"
        )
        return int(row["cnt"]) if row else 0


# ---------------------------------------------------------------------------
# AuditRepository (append-only)
# ---------------------------------------------------------------------------

class AuditRepository:
    """
    Append-only audit trail for GDPR Article 30 / PDPA compliance.
    Records MUST NOT be updated or deleted through this repository.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def log(self, event: AuditEvent) -> str:
        """Append an audit event. Returns the event_id."""
        sql = """
            INSERT INTO audit_trail
              (event_id, event_type, actor, session_id, resource_type,
               resource_id, action, outcome, detail, ip_address,
               occurred_at, compliance_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._db.execute(
            sql,
            (
                event.event_id,
                event.event_type,
                event.actor,
                event.session_id,
                event.resource_type,
                event.resource_id,
                event.action,
                event.outcome,
                event.detail,
                event.ip_address,
                event.occurred_at,
                event.compliance_tag,
            ),
        )
        return event.event_id

    def log_anonymization(
        self,
        session_id: str,
        frame_count: int,
        face_count: int,
        plate_count: int,
    ) -> str:
        return self.log(AuditEvent(
            event_type    = "anonymization",
            action        = "apply_anonymization",
            session_id    = session_id,
            resource_type = "video_session",
            resource_id   = session_id,
            detail        = (
                f"frames={frame_count}, faces_blurred={face_count}, "
                f"plates_masked={plate_count}"
            ),
            compliance_tag = "GDPR-A30",
        ))

    def log_access(
        self,
        actor: str,
        resource: str,
        action: str,
        outcome: str = "success",
        ip_address: Optional[str] = None,
    ) -> str:
        return self.log(AuditEvent(
            event_type    = "access",
            actor         = actor,
            action        = action,
            resource_type = "video_data",
            resource_id   = resource,
            outcome       = outcome,
            ip_address    = ip_address,
            compliance_tag = "GDPR-A30",
        ))

    def log_deletion(self, session_id: str, actor: str, record_count: int) -> str:
        return self.log(AuditEvent(
            event_type    = "data_deletion",
            actor         = actor,
            action        = "soft_delete",
            session_id    = session_id,
            resource_type = "detection_events",
            resource_id   = session_id,
            detail        = f"records_deleted={record_count}",
            compliance_tag = "GDPR-A17",  # Right to erasure
        ))

    def get_by_session(self, session_id: str) -> List[Dict]:
        return self._db.fetchall(
            "SELECT * FROM audit_trail WHERE session_id = ? ORDER BY occurred_at",
            (session_id,),
        )

    def get_recent(self, limit: int = 100) -> List[Dict]:
        return self._db.fetchall(
            "SELECT * FROM audit_trail ORDER BY occurred_at DESC LIMIT ?",
            (limit,),
        )

    def to_dataframe(self, session_id: Optional[str] = None):
        """Return audit trail as a pandas DataFrame for compliance reporting."""
        import pandas as pd
        if session_id:
            rows = self.get_by_session(session_id)
        else:
            rows = self.get_recent(limit=10000)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["occurred_at"] = pd.to_datetime(df["occurred_at"])
        return df
