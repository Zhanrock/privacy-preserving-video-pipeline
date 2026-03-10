"""
crypto/hasher.py
-----------------
Cryptographic pseudonymisation of tracking IDs using HMAC-SHA256.

Why HMAC instead of plain SHA-256?
  - Plain SHA-256 is a one-way hash but is susceptible to rainbow table
    attacks if the input space is small (e.g. numeric camera IDs).
  - HMAC-SHA256 requires the secret key to reverse, making pseudonymisation
    computationally infeasible without the key.
  - This satisfies GDPR Recital 26: pseudonymisation "considerably reduces
    the risks to the data subjects concerned."

Usage
-----
>>> hasher = TrackingIDHasher(secret_key=b"32-byte-secret-key-here!!")
>>> pseudo = hasher.pseudonymise("CAM-007:PERSON-1234")
>>> hasher.verify("CAM-007:PERSON-1234", pseudo)
True

Security notes
--------------
- The secret key MUST be at least 32 bytes and stored outside source code
  (use environment variable PPVP_HMAC_KEY or a secrets manager).
- Key rotation: use HasherKeyRotation to create a new hasher and re-hash
  all stored pseudonyms when the key changes.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Optional

from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class TrackingIDHasher:
    """
    HMAC-SHA256 pseudonymiser for tracking IDs.

    Parameters
    ----------
    secret_key: Bytes secret key (≥ 32 bytes).  Generate with:
                  python -c "import secrets; print(secrets.token_hex(32))"
    algorithm:  Hash algorithm (default: sha256).
    """

    MIN_KEY_LENGTH = 32

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        algorithm: str = "sha256",
    ) -> None:
        if secret_key is None:
            # Auto-generate ephemeral key (useful for single-session mode)
            secret_key = secrets.token_bytes(self.MIN_KEY_LENGTH)
            logger.warning(
                "No HMAC secret key provided — using ephemeral key. "
                "Hashes will NOT be consistent across sessions. "
                "Set PPVP_HMAC_KEY in production."
            )

        if len(secret_key) < self.MIN_KEY_LENGTH:
            raise ValueError(
                f"secret_key must be at least {self.MIN_KEY_LENGTH} bytes, "
                f"got {len(secret_key)}"
            )

        self._key       = secret_key
        self._algorithm = algorithm

        # Validate algorithm is available
        if algorithm not in hashlib.algorithms_guaranteed:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        logger.debug(
            "TrackingIDHasher initialised: algorithm=%s, key_length=%d bytes",
            algorithm, len(secret_key),
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def pseudonymise(self, tracking_id: str) -> str:
        """
        Compute the HMAC-SHA256 pseudonym for a tracking ID.

        Parameters
        ----------
        tracking_id: The original identifier (camera ID, person token, etc.)

        Returns
        -------
        64-character lowercase hex string (256-bit HMAC digest).
        """
        if not tracking_id:
            raise ValueError("tracking_id must not be empty")

        digest = hmac.new(
            self._key,
            tracking_id.encode("utf-8"),
            self._algorithm,
        ).hexdigest()

        logger.debug("Pseudonymised tracking ID (first 8 chars): %s…", digest[:8])
        return digest

    def verify(self, tracking_id: str, expected_hash: str) -> bool:
        """
        Constant-time verification — prevents timing attacks.

        Parameters
        ----------
        tracking_id:   Original ID to verify.
        expected_hash: Previously stored HMAC digest.

        Returns
        -------
        True if the ID matches the hash, False otherwise.
        """
        computed = self.pseudonymise(tracking_id)
        return hmac.compare_digest(computed, expected_hash)

    def pseudonymise_batch(self, tracking_ids: list[str]) -> list[str]:
        """Pseudonymise a list of IDs in one call."""
        return [self.pseudonymise(tid) for tid in tracking_ids]

    # ------------------------------------------------------------------
    # Frame hashing (SHA-256, no key — for deduplication not privacy)
    # ------------------------------------------------------------------

    @staticmethod
    def hash_frame(frame_bytes: bytes) -> str:
        """
        Compute a SHA-256 hash of raw frame bytes.
        Used for frame deduplication and tamper detection — not for PII.

        Parameters
        ----------
        frame_bytes: Raw image bytes (e.g. JPEG-encoded frame).

        Returns
        -------
        64-character hex string.
        """
        return hashlib.sha256(frame_bytes).hexdigest()

    @staticmethod
    def hash_numpy_frame(frame) -> str:
        """
        Compute a SHA-256 hash of a numpy array frame.
        Array is converted to bytes via tobytes() before hashing.
        """
        import numpy as np
        arr = np.ascontiguousarray(frame)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    # ------------------------------------------------------------------
    # Key management helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key(length: int = 32) -> bytes:
        """Generate a cryptographically secure random key."""
        return secrets.token_bytes(length)

    @staticmethod
    def generate_key_hex(length: int = 32) -> str:
        """Generate a hex-encoded key suitable for storage in env vars."""
        return secrets.token_hex(length)

    @classmethod
    def from_env(cls, env_var: str = "PPVP_HMAC_KEY") -> "TrackingIDHasher":
        """
        Load secret key from an environment variable.

        The env var may contain either:
          - A raw string key (interpreted as UTF-8 bytes)
          - A hex-encoded key (64+ characters)
        """
        raw = os.environ.get(env_var, "")
        if not raw:
            logger.warning("Env var %s not set — using ephemeral key", env_var)
            return cls()

        # If it looks like a hex string, decode it
        if len(raw) >= 64 and all(c in "0123456789abcdefABCDEF" for c in raw):
            key = bytes.fromhex(raw)
        else:
            key = raw.encode("utf-8")

        return cls(secret_key=key)

    def __repr__(self) -> str:
        return (
            f"TrackingIDHasher(algorithm={self._algorithm}, "
            f"key_length={len(self._key)} bytes)"
        )
