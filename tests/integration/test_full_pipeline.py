"""tests/integration/test_full_pipeline.py — Full pipeline integration tests."""
import sys, os, json, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.database.repositories import (
    SessionRepository, DetectionRepository, AuditRepository,
)
from privacy_pipeline.crypto.hasher import TrackingIDHasher
from privacy_pipeline.crypto.encryptor import MetadataEncryptor
from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
from privacy_pipeline.anonymizer.detectors import FaceDetector, LicensePlateDetector
from privacy_pipeline.pipeline.processor import PrivacyPipeline
from privacy_pipeline.pipeline.reporter import PipelineReporter

KEY = b"z" * 32

def solid_frame(h=64, w=64, color=(100, 150, 200)):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = color
    return f

def make_pipeline(db, source="integration_test"):
    return PrivacyPipeline(
        db=db,
        hasher=TrackingIDHasher(secret_key=KEY),
        face_detector=FaceDetector(),
        plate_detector=LicensePlateDetector(),
        face_blurrer=RegionBlurrer(method="pixelate", strength=20),
        plate_blurrer=RegionBlurrer(method="solid"),
        source_identifier=source,
        log_to_db=True,
    )

class TestFullProcessingCycle(unittest.TestCase):
    def test_process_50_frames(self):
        db = DatabaseManager.for_testing()
        p  = make_pipeline(db)
        sid = p.session_id
        frames = [solid_frame() for _ in range(50)]
        results = p.process_batch(frames)
        stats   = p.close()

        self.assertEqual(len(results), 50)
        self.assertEqual(stats["total_frames"], 50)
        self.assertGreater(stats["mean_latency_ms"], 0)

        rows = db.fetchall(
            "SELECT * FROM detection_events WHERE session_id = ?", (sid,)
        )
        self.assertEqual(len(rows), 50)

    def test_gdpr_audit_trail_complete(self):
        db   = DatabaseManager.for_testing()
        p    = make_pipeline(db)
        sid  = p.session_id
        p.process_batch([solid_frame() for _ in range(5)])
        p.close()

        audit = AuditRepository(db)
        events = audit.get_by_session(sid)
        event_types = [e["event_type"] for e in events]
        self.assertIn("session",       event_types)
        self.assertIn("anonymization", event_types)
        tags = [e.get("compliance_tag") for e in events]
        self.assertIn("GDPR-A30", tags)

    def test_right_to_erasure(self):
        db  = DatabaseManager.for_testing()
        p   = make_pipeline(db)
        sid = p.session_id
        for _ in range(5):
            p.process_frame(solid_frame())
        p.close()

        # Exercise GDPR Article 17 soft delete
        det_repo   = DetectionRepository(db)
        audit_repo = AuditRepository(db)
        count = det_repo.soft_delete_by_session(sid)
        audit_repo.log_deletion(sid, "test_admin", count)

        remaining = det_repo.get_by_session(sid)
        self.assertEqual(len(remaining), 0)

        audit_events = audit_repo.get_by_session(sid)
        self.assertTrue(any(e["compliance_tag"] == "GDPR-A17" for e in audit_events))

    def test_tracking_id_pseudonymisation_persisted(self):
        db  = DatabaseManager.for_testing()
        p   = make_pipeline(db)
        for i in range(3):
            p.process_frame(solid_frame(), tracking_id=f"PERSON-{i:03d}")
        p.close()

        count = db.fetchone("SELECT COUNT(*) AS cnt FROM tracking_id_map")["cnt"]
        self.assertEqual(count, 3)

    def test_same_tracking_id_increments_count(self):
        db  = DatabaseManager.for_testing()
        p   = make_pipeline(db)
        for _ in range(5):
            p.process_frame(solid_frame(), tracking_id="PERSON-001")
        p.close()

        row = db.fetchone(
            "SELECT session_count FROM tracking_id_map"
        )
        self.assertEqual(row["session_count"], 5)

    def test_crypto_roundtrip_in_pipeline(self):
        enc   = MetadataEncryptor.generate()
        data  = {"camera_id": "CAM-007", "zone": "A", "sensitivity": "high"}
        token = enc.encrypt(data)
        result = enc.decrypt(token)
        self.assertEqual(result["camera_id"], "CAM-007")
        self.assertEqual(result["zone"], "A")

class TestPipelineReporter(unittest.TestCase):
    def _setup_session(self, n_frames=10):
        db  = DatabaseManager.for_testing()
        p   = make_pipeline(db, source="reporter_test")
        sid = p.session_id
        for _ in range(n_frames):
            p.process_frame(solid_frame())
        p.close()
        return db, sid

    def test_session_summary_has_keys(self):
        db, sid = self._setup_session(5)
        reporter = PipelineReporter(db)
        s = reporter.session_summary(sid)
        for k in ["session_id", "total_frames", "faces_anonymized",
                  "plates_anonymized", "mean_latency_ms", "audit_event_count"]:
            self.assertIn(k, s)

    def test_session_summary_frame_count(self):
        db, sid = self._setup_session(15)
        reporter = PipelineReporter(db)
        s = reporter.session_summary(sid)
        self.assertEqual(s["total_frames"], 15)

    def test_compliance_report_gdpr_flag(self):
        db, sid = self._setup_session(5)
        reporter = PipelineReporter(db)
        r = reporter.compliance_report(session_id=sid)
        self.assertIn("gdpr_article_30_met", r)
        self.assertTrue(r["gdpr_article_30_met"])

    def test_compliance_report_saved_to_file(self):
        db, sid = self._setup_session(3)
        reporter = PipelineReporter(db)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.json")
            reporter.compliance_report(session_id=sid, output_path=path)
            self.assertTrue(os.path.exists(path))
            data = json.load(open(path))
            self.assertIn("total_events", data)

    def test_all_sessions_list(self):
        db, _ = self._setup_session(2)
        reporter = PipelineReporter(db)
        sessions = reporter.all_sessions()
        self.assertGreaterEqual(len(sessions), 1)

    def test_detection_dataframe_shape(self):
        import pandas as pd
        db, sid = self._setup_session(8)
        reporter = PipelineReporter(db)
        df = reporter.detection_dataframe(sid)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 8)

    def test_audit_dataframe_not_empty(self):
        import pandas as pd
        db, sid = self._setup_session(3)
        reporter = PipelineReporter(db)
        df = reporter.audit_dataframe(session_id=sid)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
