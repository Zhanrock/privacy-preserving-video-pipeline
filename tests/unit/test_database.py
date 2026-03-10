"""tests/unit/test_database.py — Database layer tests (SQLite in-memory)."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.database.repositories import (
    SessionRepository, DetectionRepository,
    TrackingIDRepository, AuditRepository,
    DetectionEvent, AuditEvent,
)

def make_db():
    return DatabaseManager.for_testing()

class TestDatabaseManager(unittest.TestCase):
    def test_is_sqlite(self):
        db = make_db()
        self.assertTrue(db.is_sqlite())
        self.assertFalse(db.is_mysql())

    def test_schema_tables_exist(self):
        db = make_db()
        tables = [r["name"] for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        for t in ["anonymization_sessions", "detection_events",
                  "tracking_id_map", "audit_trail"]:
            self.assertIn(t, tables, f"Table {t} missing")

    def test_context_manager(self):
        with make_db() as db:
            rows = db.fetchall("SELECT 1 AS ok")
            self.assertEqual(rows[0]["ok"], 1)

    def test_execute_and_fetch(self):
        db = make_db()
        rows = db.fetchall("SELECT 1 AS val")
        self.assertEqual(rows[0]["val"], 1)

    def test_fetchone_none_for_missing(self):
        db = make_db()
        row = db.fetchone(
            "SELECT * FROM anonymization_sessions WHERE session_id = ?",
            ("nonexistent",)
        )
        self.assertIsNone(row)


class TestSessionRepository(unittest.TestCase):
    def setUp(self):
        self.db   = make_db()
        self.repo = SessionRepository(self.db)

    def test_create_and_get(self):
        sid = "test-session-001"
        self.repo.create(sid, source_identifier="CAM-001")
        row = self.repo.get(sid)
        self.assertIsNotNone(row)
        self.assertEqual(row["session_id"], sid)
        self.assertEqual(row["source_identifier"], "CAM-001")

    def test_create_with_config(self):
        sid = "test-session-002"
        self.repo.create(sid, "CAM-002", config_snapshot={"epsilon": 0.03})
        row = self.repo.get(sid)
        self.assertIsNotNone(row)

    def test_update_stats(self):
        sid = "test-session-003"
        self.repo.create(sid, "CAM-003")
        self.repo.update_stats(sid, 100, 30, 5, 18.5, 28.9)
        row = self.repo.get(sid)
        self.assertEqual(row["total_frames"], 100)
        self.assertAlmostEqual(row["mean_latency_ms"], 18.5, places=1)

    def test_close_session(self):
        sid = "test-session-004"
        self.repo.create(sid, "CAM-004")
        self.repo.close_session(sid, status="completed")
        row = self.repo.get(sid)
        self.assertEqual(row["status"], "completed")
        self.assertIsNotNone(row["ended_at"])

    def test_list_recent(self):
        for i in range(3):
            self.repo.create(f"sess-{i}", f"CAM-{i}")
        rows = self.repo.list_recent(limit=10)
        self.assertGreaterEqual(len(rows), 3)

    def test_get_nonexistent_returns_none(self):
        row = self.repo.get("does-not-exist")
        self.assertIsNone(row)


class TestDetectionRepository(unittest.TestCase):
    def setUp(self):
        self.db   = make_db()
        self.sid  = "det-session-001"
        SessionRepository(self.db).create(self.sid, "CAM-DET")
        self.repo = DetectionRepository(self.db)

    def _event(self, seq=1, faces=2, plates=0, latency=15.3):
        return DetectionEvent(
            session_id=self.sid, frame_sequence=seq,
            captured_at="2024-12-01 10:00:00",
            frame_hash_sha256="a" * 64,
            face_count=faces, plate_count=plates,
            processing_latency_ms=latency,
        )

    def test_insert_and_retrieve(self):
        self.repo.insert(self._event())
        rows = self.repo.get_by_session(self.sid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["face_count"], 2)

    def test_bulk_insert(self):
        events = [self._event(seq=i, faces=i%3) for i in range(1, 11)]
        self.repo.bulk_insert(events)
        rows = self.repo.get_by_session(self.sid)
        self.assertEqual(len(rows), 10)

    def test_session_summary(self):
        events = [self._event(seq=i, faces=1, plates=0, latency=10.0 + i)
                  for i in range(1, 6)]
        self.repo.bulk_insert(events)
        summary = self.repo.get_session_summary(self.sid)
        self.assertEqual(int(summary["total_frames"]), 5)
        self.assertEqual(int(summary["total_faces"]), 5)

    def test_soft_delete(self):
        for i in range(1, 4):
            self.repo.insert(self._event(seq=i))
        count = self.repo.soft_delete_by_session(self.sid)
        self.assertEqual(count, 3)
        rows = self.repo.get_by_session(self.sid)
        self.assertEqual(len(rows), 0)

    def test_to_dataframe(self):
        import pandas as pd
        self.repo.insert(self._event())
        df = self.repo.to_dataframe(self.sid)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("face_count", df.columns)

    def test_empty_dataframe(self):
        import pandas as pd
        df = self.repo.to_dataframe("nonexistent-session")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)


class TestTrackingIDRepository(unittest.TestCase):
    def setUp(self):
        self.db   = make_db()
        self.repo = TrackingIDRepository(self.db)

    def test_upsert_new(self):
        self.repo.upsert("a" * 64)
        self.assertTrue(self.repo.exists("a" * 64))

    def test_upsert_existing_increments(self):
        h = "b" * 64
        self.repo.upsert(h)
        self.repo.upsert(h)
        row = self.db.fetchone(
            "SELECT session_count FROM tracking_id_map WHERE pseudonym_hash = ?", (h,)
        )
        self.assertEqual(row["session_count"], 2)

    def test_exists_false_for_unknown(self):
        self.assertFalse(self.repo.exists("c" * 64))

    def test_deactivate(self):
        h = "d" * 64
        self.repo.upsert(h)
        self.repo.deactivate(h)
        row = self.db.fetchone(
            "SELECT is_active FROM tracking_id_map WHERE pseudonym_hash = ?", (h,)
        )
        self.assertEqual(row["is_active"], 0)

    def test_count_active(self):
        initial = self.repo.count_active()
        self.repo.upsert("e" * 64)
        self.repo.upsert("f" * 64)
        self.assertEqual(self.repo.count_active(), initial + 2)


class TestAuditRepository(unittest.TestCase):
    def setUp(self):
        self.db   = make_db()
        sid = "audit-session-001"
        SessionRepository(self.db).create(sid, "CAM-AUDIT")
        self.sid  = sid
        self.repo = AuditRepository(self.db)

    def test_log_event(self):
        event_id = self.repo.log(AuditEvent(
            event_type="test", action="unit_test", session_id=self.sid
        ))
        self.assertIsNotNone(event_id)
        rows = self.repo.get_by_session(self.sid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_type"], "test")

    def test_log_anonymization(self):
        eid = self.repo.log_anonymization(self.sid, 100, 30, 5)
        self.assertIsNotNone(eid)
        rows = self.repo.get_by_session(self.sid)
        self.assertTrue(any(r["action"] == "apply_anonymization" for r in rows))

    def test_log_access(self):
        self.repo.log_access("user_1", "video_data", "read")
        recent = self.repo.get_recent(limit=5)
        self.assertTrue(any(r["event_type"] == "access" for r in recent))

    def test_log_deletion(self):
        self.repo.log_deletion(self.sid, "admin", 50)
        rows = self.repo.get_by_session(self.sid)
        self.assertTrue(any(r["compliance_tag"] == "GDPR-A17" for r in rows))

    def test_compliance_tag_present(self):
        self.repo.log_anonymization(self.sid, 10, 5, 0)
        rows = self.repo.get_by_session(self.sid)
        tags = [r.get("compliance_tag") for r in rows]
        self.assertIn("GDPR-A30", tags)

    def test_get_recent(self):
        for i in range(5):
            self.repo.log(AuditEvent(event_type="bulk", action=f"action_{i}",
                                     session_id=self.sid))
        recent = self.repo.get_recent(limit=10)
        self.assertGreaterEqual(len(recent), 5)

    def test_to_dataframe(self):
        import pandas as pd
        self.repo.log_anonymization(self.sid, 10, 3, 1)
        df = self.repo.to_dataframe(self.sid)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("event_type", df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
