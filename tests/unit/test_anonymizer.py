"""tests/unit/test_anonymizer.py — Anonymizer module tests."""
import sys, os, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from privacy_pipeline.anonymizer.detectors import DetectedRegion, FaceDetector, LicensePlateDetector
from privacy_pipeline.anonymizer.blurrer import RegionBlurrer

def solid_frame(h=128, w=128, color=(120, 80, 60)):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame

class TestDetectedRegion(unittest.TestCase):
    def test_x2_y2(self):
        r = DetectedRegion(10, 20, 50, 60, "face")
        self.assertEqual(r.x2, 60)
        self.assertEqual(r.y2, 80)

    def test_with_padding_clamps(self):
        r = DetectedRegion(5, 5, 20, 20, "face")
        padded = r.with_padding(10, 100, 100)
        self.assertGreaterEqual(padded.x, 0)
        self.assertGreaterEqual(padded.y, 0)

    def test_to_dict(self):
        r = DetectedRegion(1, 2, 3, 4, "license_plate", confidence=0.8)
        d = r.to_dict()
        self.assertEqual(d["type"], "license_plate")
        self.assertAlmostEqual(d["confidence"], 0.8)

class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        self.det = FaceDetector()

    def test_detects_on_blank_frame(self):
        frame = solid_frame()
        regions = self.det.detect(frame)
        # No face expected in solid frame — just assert it runs
        self.assertIsInstance(regions, list)

    def test_all_regions_type_face(self):
        frame   = solid_frame()
        regions = self.det.detect(frame)
        for r in regions:
            self.assertEqual(r.region_type, "face")

    def test_empty_frame_returns_empty(self):
        empty   = np.zeros((0, 0, 3), dtype=np.uint8)
        regions = self.det.detect(empty)
        self.assertEqual(regions, [])

    def test_none_returns_empty(self):
        regions = self.det.detect(None)
        self.assertEqual(regions, [])

class TestLicensePlateDetector(unittest.TestCase):
    def setUp(self):
        self.det = LicensePlateDetector()

    def test_returns_list(self):
        frame   = solid_frame(256, 256)
        regions = self.det.detect(frame)
        self.assertIsInstance(regions, list)

    def test_all_regions_type_plate(self):
        frame   = solid_frame(256, 256)
        regions = self.det.detect(frame)
        for r in regions:
            self.assertEqual(r.region_type, "license_plate")

    def test_empty_frame_returns_empty(self):
        self.assertEqual(self.det.detect(np.zeros((0, 0, 3), dtype=np.uint8)), [])

class TestRegionBlurrer(unittest.TestCase):
    def _regions(self):
        return [DetectedRegion(10, 10, 40, 40, "face")]

    def test_gaussian_preserves_shape(self):
        frame  = solid_frame()
        b      = RegionBlurrer(method="gaussian", strength=15)
        result = b.apply(frame, self._regions())
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)

    def test_pixelate_preserves_shape(self):
        frame  = solid_frame()
        b      = RegionBlurrer(method="pixelate", strength=10)
        result = b.apply(frame, self._regions())
        self.assertEqual(result.shape, frame.shape)

    def test_solid_preserves_shape(self):
        frame  = solid_frame()
        b      = RegionBlurrer(method="solid")
        result = b.apply(frame, self._regions())
        self.assertEqual(result.shape, frame.shape)

    def test_solid_fills_black(self):
        frame  = solid_frame(color=(200, 200, 200))
        b      = RegionBlurrer(method="solid", fill_color=(0, 0, 0))
        result = b.apply(frame, self._regions())
        roi    = result[10:50, 10:50]
        self.assertEqual(roi.max(), 0)

    def test_no_regions_returns_copy(self):
        frame  = solid_frame()
        b      = RegionBlurrer()
        result = b.apply(frame, [])
        np.testing.assert_array_equal(result, frame)
        self.assertIsNot(result, frame)  # Must be a copy

    def test_does_not_modify_original(self):
        frame  = solid_frame()
        orig   = frame.copy()
        b      = RegionBlurrer(method="solid")
        b.apply(frame, self._regions())
        np.testing.assert_array_equal(frame, orig)

    def test_gaussian_blurs_region(self):
        frame  = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        b      = RegionBlurrer(method="gaussian", strength=31)
        result = b.apply(frame, self._regions())
        # The blurred region should differ from original (unless extremely lucky)
        self.assertFalse(np.array_equal(result[10:50, 10:50], frame[10:50, 10:50]))

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            RegionBlurrer(method="xyzzy")

    def test_invalid_strength_raises(self):
        with self.assertRaises(ValueError):
            RegionBlurrer(strength=0)

    def test_clamps_out_of_bounds_region(self):
        frame  = solid_frame(50, 50)
        b      = RegionBlurrer(method="solid")
        huge   = [DetectedRegion(0, 0, 200, 200, "face")]
        result = b.apply(frame, huge)  # Must not raise
        self.assertEqual(result.shape, (50, 50, 3))

    def test_repr(self):
        b = RegionBlurrer(method="pixelate", strength=20)
        self.assertIn("pixelate", repr(b))

if __name__ == "__main__":
    unittest.main(verbosity=2)
