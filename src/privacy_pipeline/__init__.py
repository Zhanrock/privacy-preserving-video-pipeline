"""
Privacy-Preserving Video Pipeline
===================================
High-speed video anonymization pipeline with:
- Face blurring and license plate masking (OpenCV)
- Cryptographic HMAC-SHA256 tracking ID pseudonymisation
- AES-256-GCM metadata encryption at rest
- MySQL audit logging (GDPR Article 30 / PDPA compliant)
- Sub-30ms per-frame latency target
"""
__version__ = "1.0.0"
__author__  = "Privacy Pipeline Team"
