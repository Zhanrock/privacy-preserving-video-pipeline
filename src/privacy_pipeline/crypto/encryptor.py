"""
crypto/encryptor.py
--------------------
AES-256-GCM authenticated encryption for metadata at rest.

Why AES-256-GCM?
  - AES-256: 256-bit key, NIST-approved, computationally infeasible to brute-force.
  - GCM mode: Authenticated encryption — provides BOTH confidentiality AND
    integrity guarantees.  Any tampered ciphertext is rejected before decryption.
  - The GCM authentication tag acts as a MAC, preventing undetected modification.

Wire format (base64-encoded JSON):
  {
    "v":    1,               # schema version for future key rotation
    "nonce": "<12-byte hex>",
    "tag":   "<16-byte hex>",
    "ct":    "<ciphertext hex>"
  }

Usage
-----
>>> enc = MetadataEncryptor.generate()
>>> token = enc.encrypt({"camera_id": "CAM-001", "location": "Zone-A"})
>>> data  = enc.decrypt(token)
>>> data["camera_id"]
'CAM-001'
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from typing import Any, Dict, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""


class MetadataEncryptor:
    """
    AES-256-GCM authenticated encryption for JSON metadata.

    Parameters
    ----------
    key: 32-byte (256-bit) AES key.
    """

    KEY_SIZE    = 32   # bytes → 256-bit AES
    NONCE_SIZE  = 12   # bytes → 96-bit GCM nonce (NIST recommended)
    SCHEMA_VER  = 1

    def __init__(self, key: bytes) -> None:
        if len(key) != self.KEY_SIZE:
            raise ValueError(
                f"AES-256 key must be exactly {self.KEY_SIZE} bytes, "
                f"got {len(key)}"
            )
        self._key  = key
        self._aesgcm = AESGCM(key)
        logger.debug("MetadataEncryptor initialised (AES-256-GCM)")

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------

    def encrypt(self, data: Union[Dict[str, Any], str, bytes]) -> str:
        """
        Encrypt data and return a base64-encoded token.

        Parameters
        ----------
        data: Dict (serialised to JSON), str, or raw bytes.

        Returns
        -------
        Base64-encoded string containing nonce + tag + ciphertext.
        """
        if isinstance(data, dict):
            plaintext = json.dumps(data, separators=(",", ":")).encode("utf-8")
        elif isinstance(data, str):
            plaintext = data.encode("utf-8")
        elif isinstance(data, bytes):
            plaintext = data
        else:
            raise TypeError(f"Unsupported plaintext type: {type(data)}")

        nonce = secrets.token_bytes(self.NONCE_SIZE)

        # AESGCM.encrypt returns ciphertext + 16-byte GCM tag appended
        ct_with_tag = self._aesgcm.encrypt(nonce, plaintext, associated_data=None)
        ciphertext  = ct_with_tag[:-16]
        tag         = ct_with_tag[-16:]

        payload = json.dumps({
            "v":     self.SCHEMA_VER,
            "nonce": nonce.hex(),
            "tag":   tag.hex(),
            "ct":    ciphertext.hex(),
        }, separators=(",", ":"))

        return base64.b64encode(payload.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> Any:
        """
        Decrypt a token produced by encrypt().

        Returns
        -------
        Original dict (if JSON), str, or bytes depending on what was encrypted.

        Raises
        ------
        EncryptionError: If the token is invalid or the ciphertext has been tampered.
        """
        try:
            payload_bytes = base64.b64decode(token.encode("ascii"))
            payload       = json.loads(payload_bytes)

            nonce      = bytes.fromhex(payload["nonce"])
            tag        = bytes.fromhex(payload["tag"])
            ciphertext = bytes.fromhex(payload["ct"])

            # Re-assemble ciphertext + tag for AESGCM.decrypt
            ct_with_tag = ciphertext + tag
            plaintext   = self._aesgcm.decrypt(nonce, ct_with_tag, associated_data=None)

            # Try to deserialise as JSON
            try:
                return json.loads(plaintext.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return plaintext

        except Exception as exc:
            raise EncryptionError(f"Decryption failed: {exc}") from exc

    def encrypt_string(self, value: str) -> str:
        """Convenience: encrypt a plain string."""
        return self.encrypt(value.encode("utf-8"))

    def decrypt_string(self, token: str) -> str:
        """Convenience: decrypt to a plain string."""
        result = self.decrypt(token)
        if isinstance(result, bytes):
            return result.decode("utf-8")
        return str(result)

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    @classmethod
    def generate(cls) -> "MetadataEncryptor":
        """Create a new encryptor with a randomly generated AES-256 key."""
        key = secrets.token_bytes(cls.KEY_SIZE)
        logger.debug("Generated new AES-256-GCM key")
        return cls(key)

    @classmethod
    def from_hex_key(cls, hex_key: str) -> "MetadataEncryptor":
        """Create from a 64-character hex-encoded key."""
        return cls(bytes.fromhex(hex_key))

    @classmethod
    def from_env(cls, env_var: str = "PPVP_AES_KEY") -> "MetadataEncryptor":
        """Load AES key from environment variable (hex-encoded)."""
        raw = os.environ.get(env_var, "")
        if raw:
            return cls.from_hex_key(raw)
        logger.warning("Env var %s not set — generating ephemeral AES key", env_var)
        return cls.generate()

    @property
    def key_hex(self) -> str:
        """Return the key as a hex string (for secure storage)."""
        return self._key.hex()

    def __repr__(self) -> str:
        return f"MetadataEncryptor(AES-{self.KEY_SIZE * 8}-GCM)"
