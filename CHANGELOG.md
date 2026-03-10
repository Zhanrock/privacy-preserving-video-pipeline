# Changelog

## [1.0.0] — 2024-12-01

### Added
- FaceDetector (Haar Cascade) + LicensePlateDetector (MSER)
- RegionBlurrer: gaussian, pixelate, solid modes
- TrackingIDHasher — HMAC-SHA256 pseudonymisation
- MetadataEncryptor — AES-256-GCM authenticated encryption
- DatabaseManager — MySQL (production) + SQLite (development/CI) backends
- 4-table MySQL schema with GDPR-compliant design
- SessionRepository, DetectionRepository, TrackingIDRepository, AuditRepository
- PrivacyPipeline — end-to-end orchestrator with sub-30ms latency target
- PipelineReporter — compliance + operational reports, pandas DataFrames
- FastAPI REST API with GDPR erasure endpoint
- 105 unit + integration tests (all passing without MySQL)
- GitHub Actions CI: unit tests (3 Python versions) + MySQL service container
- migrations/001_initial_schema.sql — complete MySQL DDL with views
