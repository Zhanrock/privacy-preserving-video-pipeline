"""tests/unit/test_pipeline.py — PrivacyPipeline unit tests."""
import sys, os, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.crypto.hasher import TrackingIDHasher
from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
from privacy_pipeline.pipeline.processor import PrivacyPipeline, FrameResult

KEY = b"x" * 32

def make_pipeline(log=True):
    db = DatabaseManager.for_testing()
    return PrivacyPipeline(
        db=db,
        hasher=TrackingIDHasher(secret_key=KEY),
        face_detector=None,
        plate_detector=None,
        face_blurrer=RegionBlurrer(method="pixelate", strength=25),
        plate_blurrer=RegionBlurrer(method="solid"),
        source_identifier="test_cam",
        log_to_db=log,
    )

def solid_frame(h=64, w=64):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = 100
    return f

class TestPrivacyPipeline(unittest.TestCase):
    def test_process_frame_returns_result(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame())
        self.assertIsInstance(r, FrameResult)

    def test_anonymized_frame_shape(self):
        p = make_pipeline()
        f = solid_frame(128, 192)
        r = p.process_frame(f)
        self.assertEqual(r.anonymized_frame.shape, f.shape)

    def test_anonymized_frame_is_copy(self):
        p = make_pipeline()
        f = solid_frame()
        r = p.process_frame(f)
        self.assertIsNot(r.anonymized_frame, f)

    def test_frame_hash_is_64_hex(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame())
        self.assertEqual(len(r.frame_hash_sha256), 64)

    def test_frame_sequence_increments(self):
        p = make_pipeline()
        r1 = p.process_frame(solid_frame())
        r2 = p.process_frame(solid_frame())
        self.assertEqual(r1.frame_sequence, 1)
        self.assertEqual(r2.frame_sequence, 2)

    def test_no_detections_without_detectors(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame())
        self.assertEqual(r.face_count, 0)
        self.assertEqual(r.plate_count, 0)

    def test_pseudonym_returned_when_tracking_id_given(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame(), tracking_id="CAM-001")
        self.assertIsNotNone(r.pseudonym_hash)
        self.assertEqual(len(r.pseudonym_hash), 64)

    def test_pseudonym_none_when_no_tracking_id(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame())
        self.assertIsNone(r.pseudonym_hash)

    def test_latency_is_positive(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame())
        self.assertGreater(r.processing_latency_ms, 0.0)

    def test_to_dict_has_required_keys(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame())
        d = r.to_dict()
        for k in ["frame_sequence", "face_count", "plate_count",
                  "frame_hash_sha256", "processing_latency_ms", "session_id"]:
            self.assertIn(k, d)

    def test_process_batch(self):
        p = make_pipeline()
        frames = [solid_frame() for _ in range(5)]
        results = p.process_batch(frames)
        self.assertEqual(len(results), 5)
        seqs = [r.frame_sequence for r in results]
        self.assertEqual(seqs, [1, 2, 3, 4, 5])

    def test_get_stats_after_processing(self):
        p = make_pipeline()
        for _ in range(10):
            p.process_frame(solid_frame())
        stats = p.get_stats()
        self.assertEqual(stats["total_frames"], 10)
        self.assertGreater(stats["mean_latency_ms"], 0)
        self.assertIn("under_target_pct", stats)

    def test_close_returns_stats(self):
        p = make_pipeline()
        p.process_frame(solid_frame())
        stats = p.close()
        self.assertIn("total_frames", stats)
        self.assertEqual(stats["total_frames"], 1)

    def test_context_manager(self):
        db = DatabaseManager.for_testing()
        with PrivacyPipeline(
            db=db, hasher=TrackingIDHasher(KEY),
            face_detector=None, plate_detector=None,
            face_blurrer=RegionBlurrer(), plate_blurrer=RegionBlurrer(method="solid"),
            source_identifier="ctx_test",
        ) as p:
            p.process_frame(solid_frame())
        # No exception = pass

    def test_session_written_to_db(self):
        db = DatabaseManager.for_testing()
        p  = PrivacyPipeline(
            db=db, hasher=TrackingIDHasher(KEY),
            face_detector=None, plate_detector=None,
            face_blurrer=RegionBlurrer(), plate_blurrer=RegionBlurrer(method="solid"),
            source_identifier="db_test", log_to_db=True,
        )
        sid = p.session_id
        p.process_frame(solid_frame())
        p.close()
        row = db.fetchone(
            "SELECT * FROM anonymization_sessions WHERE session_id = ?", (sid,)
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["source_identifier"], "db_test")

    def test_detection_events_written(self):
        db = DatabaseManager.for_testing()
        p  = PrivacyPipeline(
            db=db, hasher=TrackingIDHasher(KEY),
            face_detector=None, plate_detector=None,
            face_blurrer=RegionBlurrer(), plate_blurrer=RegionBlurrer(method="solid"),
            source_identifier="db_det_test", log_to_db=True,
        )
        sid = p.session_id
        for _ in range(3):
            p.process_frame(solid_frame())
        p.close()
        rows = db.fetchall(
            "SELECT * FROM detection_events WHERE session_id = ?", (sid,)
        )
        self.assertEqual(len(rows), 3)

    def test_has_detections_property(self):
        p = make_pipeline()
        r = p.process_frame(solid_frame())
        self.assertFalse(r.has_detections)

if __name__ == "__main__":
    unittest.main(verbosity=2)
