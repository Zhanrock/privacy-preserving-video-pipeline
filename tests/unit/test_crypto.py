"""tests/unit/test_crypto.py — Crypto module tests."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from privacy_pipeline.crypto.hasher import TrackingIDHasher
from privacy_pipeline.crypto.encryptor import MetadataEncryptor, EncryptionError

KEY_32 = b"a" * 32

class TestTrackingIDHasher(unittest.TestCase):
    def setUp(self):
        self.h = TrackingIDHasher(secret_key=KEY_32)

    def test_pseudonymise_returns_64_hex(self):
        p = self.h.pseudonymise("CAM-001")
        self.assertEqual(len(p), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in p))

    def test_same_input_same_output(self):
        p1 = self.h.pseudonymise("CAM-001")
        p2 = self.h.pseudonymise("CAM-001")
        self.assertEqual(p1, p2)

    def test_different_input_different_output(self):
        p1 = self.h.pseudonymise("CAM-001")
        p2 = self.h.pseudonymise("CAM-002")
        self.assertNotEqual(p1, p2)

    def test_verify_correct(self):
        p = self.h.pseudonymise("ID-123")
        self.assertTrue(self.h.verify("ID-123", p))

    def test_verify_wrong_input(self):
        p = self.h.pseudonymise("ID-123")
        self.assertFalse(self.h.verify("ID-999", p))

    def test_verify_tampered_hash(self):
        p = self.h.pseudonymise("ID-123")
        self.assertFalse(self.h.verify("ID-123", "a" * 64))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            self.h.pseudonymise("")

    def test_short_key_raises(self):
        with self.assertRaises(ValueError):
            TrackingIDHasher(secret_key=b"tooshort")

    def test_different_keys_different_hashes(self):
        h2 = TrackingIDHasher(secret_key=b"b" * 32)
        p1 = self.h.pseudonymise("CAM-001")
        p2 = h2.pseudonymise("CAM-001")
        self.assertNotEqual(p1, p2)

    def test_batch_pseudonymise(self):
        ids = ["A", "B", "C"]
        results = self.h.pseudonymise_batch(ids)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(set(results)), 3)

    def test_hash_frame_bytes(self):
        h = TrackingIDHasher.hash_frame(b"some_frame_data")
        self.assertEqual(len(h), 64)

    def test_hash_numpy_frame(self):
        import numpy as np
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        h = TrackingIDHasher.hash_numpy_frame(frame)
        self.assertEqual(len(h), 64)

    def test_same_frame_same_hash(self):
        import numpy as np
        frame = np.ones((32, 32, 3), dtype=np.uint8) * 128
        h1 = TrackingIDHasher.hash_numpy_frame(frame)
        h2 = TrackingIDHasher.hash_numpy_frame(frame)
        self.assertEqual(h1, h2)

    def test_generate_key(self):
        k = TrackingIDHasher.generate_key(32)
        self.assertEqual(len(k), 32)

    def test_generate_key_hex(self):
        k = TrackingIDHasher.generate_key_hex(32)
        self.assertEqual(len(k), 64)
        bytes.fromhex(k)  # Must be valid hex


class TestMetadataEncryptor(unittest.TestCase):
    def setUp(self):
        self.enc = MetadataEncryptor.generate()

    def test_encrypt_decrypt_dict(self):
        data = {"camera": "CAM-001", "zone": "A"}
        token = self.enc.encrypt(data)
        result = self.enc.decrypt(token)
        self.assertEqual(result["camera"], "CAM-001")
        self.assertEqual(result["zone"], "A")

    def test_encrypt_decrypt_string(self):
        token  = self.enc.encrypt("hello world")
        result = self.enc.decrypt(token)
        self.assertIn("hello", str(result))

    def test_encrypt_decrypt_bytes(self):
        data   = b"raw bytes"
        token  = self.enc.encrypt(data)
        result = self.enc.decrypt(token)
        self.assertIn(b"raw", result if isinstance(result, bytes) else result.encode())

    def test_different_encryptions_of_same_data(self):
        """GCM uses random nonce — same plaintext → different ciphertext each time."""
        t1 = self.enc.encrypt({"x": 1})
        t2 = self.enc.encrypt({"x": 1})
        self.assertNotEqual(t1, t2)

    def test_tampered_token_raises(self):
        token = self.enc.encrypt({"x": 1})
        bad   = token[:-4] + "AAAA"
        with self.assertRaises(EncryptionError):
            self.enc.decrypt(bad)

    def test_wrong_key_raises(self):
        enc2  = MetadataEncryptor.generate()
        token = self.enc.encrypt({"x": 1})
        with self.assertRaises(EncryptionError):
            enc2.decrypt(token)

    def test_invalid_key_size_raises(self):
        with self.assertRaises(ValueError):
            MetadataEncryptor(b"tooshort")

    def test_from_hex_key(self):
        key = MetadataEncryptor.generate()
        enc2 = MetadataEncryptor.from_hex_key(key.key_hex)
        data  = {"test": True}
        token = key.encrypt(data)
        result = enc2.decrypt(token)
        self.assertEqual(result["test"], True)

    def test_repr(self):
        self.assertIn("AES-256-GCM", repr(self.enc))

    def test_encrypt_string_helper(self):
        token  = self.enc.encrypt_string("hello")
        result = self.enc.decrypt_string(token)
        self.assertEqual(result, "hello")


if __name__ == "__main__":
    unittest.main(verbosity=2)
