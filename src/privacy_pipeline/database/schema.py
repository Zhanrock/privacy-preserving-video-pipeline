"""
database/schema.py
-------------------
MySQL DDL for all Privacy Pipeline tables.

Tables
------
detection_events   — Per-frame detection metadata (faces, plates found)
audit_trail        — Immutable compliance audit log (GDPR Article 30 / PDPA)
anonymization_sessions — Session-level summary (frames processed, latency stats)
tracking_id_map    — Pseudonymised tracking ID registry (HMAC-SHA256 hashes only)

Design principles:
  - No raw PII stored (names, faces, plate numbers) — only hashes and counts
  - Immutable audit_trail: no UPDATE/DELETE allowed via application layer
  - Soft deletes with deleted_at timestamps for detection_events
  - JSON columns for extensible metadata without schema migrations
  - All datetime columns use DATETIME(6) for microsecond precision
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# All DDL statements — executed in order by DatabaseManager.initialize_schema()
# ---------------------------------------------------------------------------

SCHEMA_DDL: list[str] = [

    # ── Session table ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS `anonymization_sessions` (
        `id`                    INTEGER PRIMARY KEY AUTO_INCREMENT,
        `session_id`            VARCHAR(36)  NOT NULL UNIQUE,
        `started_at`            DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
        `ended_at`              DATETIME(6)  NULL,
        `source_identifier`     VARCHAR(255) NOT NULL,
        `total_frames`          INTEGER      NOT NULL DEFAULT 0,
        `frames_with_faces`     INTEGER      NOT NULL DEFAULT 0,
        `frames_with_plates`    INTEGER      NOT NULL DEFAULT 0,
        `mean_latency_ms`       FLOAT        NULL,
        `p99_latency_ms`        FLOAT        NULL,
        `config_snapshot`       JSON         NULL,
        `status`                VARCHAR(20)  NOT NULL DEFAULT 'running',
        `created_at`            DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
    """,

    # ── Per-frame detection events ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS `detection_events` (
        `id`                    INTEGER PRIMARY KEY AUTO_INCREMENT,
        `session_id`            VARCHAR(36)  NOT NULL,
        `frame_sequence`        INTEGER      NOT NULL,
        `captured_at`           DATETIME(6)  NOT NULL,
        `processed_at`          DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
        `frame_hash_sha256`     VARCHAR(64)  NOT NULL,
        `face_count`            TINYINT(1)   NOT NULL DEFAULT 0,
        `plate_count`           TINYINT(1)   NOT NULL DEFAULT 0,
        `processing_latency_ms` FLOAT        NOT NULL,
        `anonymization_applied` TINYINT(1)   NOT NULL DEFAULT 1,
        `detection_meta`        JSON         NULL,
        `deleted_at`            DATETIME(6)  NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
    """,

    # ── Pseudonymised tracking ID map ───────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS `tracking_id_map` (
        `id`                    INTEGER PRIMARY KEY AUTO_INCREMENT,
        `pseudonym_hash`        VARCHAR(64)  NOT NULL UNIQUE,
        `hmac_algorithm`        VARCHAR(20)  NOT NULL DEFAULT 'sha256',
        `first_seen_at`         DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
        `last_seen_at`          DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
        `session_count`         INTEGER      NOT NULL DEFAULT 1,
        `is_active`             TINYINT(1)   NOT NULL DEFAULT 1
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
    """,

    # ── Immutable audit trail ───────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS `audit_trail` (
        `id`                    INTEGER PRIMARY KEY AUTO_INCREMENT,
        `event_id`              VARCHAR(36)  NOT NULL UNIQUE,
        `event_type`            VARCHAR(50)  NOT NULL,
        `actor`                 VARCHAR(100) NOT NULL DEFAULT 'system',
        `session_id`            VARCHAR(36)  NULL,
        `resource_type`         VARCHAR(50)  NULL,
        `resource_id`           VARCHAR(100) NULL,
        `action`                VARCHAR(50)  NOT NULL,
        `outcome`               VARCHAR(20)  NOT NULL DEFAULT 'success',
        `detail`                MEDIUMTEXT   NULL,
        `ip_address`            VARCHAR(45)  NULL,
        `occurred_at`           DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
        `compliance_tag`        VARCHAR(20)  NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
    """,

    # ── Indexes ─────────────────────────────────────────────────────────────
    "CREATE INDEX IF NOT EXISTS idx_de_session  ON detection_events (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_de_captured ON detection_events (captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_de_hash     ON detection_events (frame_hash_sha256)",
    "CREATE INDEX IF NOT EXISTS idx_at_session  ON audit_trail (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_at_type     ON audit_trail (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_at_occurred ON audit_trail (occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_ti_hash     ON tracking_id_map (pseudonym_hash)",
]
