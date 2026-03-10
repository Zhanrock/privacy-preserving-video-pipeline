-- =============================================================================
-- Migration 001: Initial Schema
-- Privacy-Preserving Video Pipeline — MySQL
-- Run: mysql -u pp_user -p privacy_pipeline_db < migrations/001_initial_schema.sql
-- =============================================================================

-- Create database (run as root/admin)
-- CREATE DATABASE IF NOT EXISTS privacy_pipeline_db
--     CHARACTER SET utf8mb4
--     COLLATE utf8mb4_unicode_ci;

-- CREATE USER IF NOT EXISTS 'pp_user'@'localhost' IDENTIFIED BY 'your_password';
-- GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX ON privacy_pipeline_db.* TO 'pp_user'@'localhost';
-- FLUSH PRIVILEGES;

USE privacy_pipeline_db;

-- ---------------------------------------------------------------------------
-- Session table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `anonymization_sessions` (
    `id`                    INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `session_id`            VARCHAR(36)     NOT NULL,
    `started_at`            DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `ended_at`              DATETIME(6)     NULL,
    `source_identifier`     VARCHAR(255)    NOT NULL,
    `total_frames`          INT UNSIGNED    NOT NULL DEFAULT 0,
    `frames_with_faces`     INT UNSIGNED    NOT NULL DEFAULT 0,
    `frames_with_plates`    INT UNSIGNED    NOT NULL DEFAULT 0,
    `mean_latency_ms`       FLOAT           NULL COMMENT 'Mean per-frame processing latency',
    `p99_latency_ms`        FLOAT           NULL COMMENT '99th percentile latency',
    `config_snapshot`       JSON            NULL     COMMENT 'Config state at session start',
    `status`                ENUM('running','completed','failed','aborted')
                                            NOT NULL DEFAULT 'running',
    `created_at`            DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_session_id` (`session_id`),
    KEY `idx_started_at` (`started_at`),
    KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='One row per video processing session';

-- ---------------------------------------------------------------------------
-- Per-frame detection events
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `detection_events` (
    `id`                    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `session_id`            VARCHAR(36)     NOT NULL,
    `frame_sequence`        INT UNSIGNED    NOT NULL COMMENT 'Frame number within session',
    `captured_at`           DATETIME(6)     NOT NULL COMMENT 'When the frame was captured',
    `processed_at`          DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `frame_hash_sha256`     CHAR(64)        NOT NULL COMMENT 'SHA-256 of raw frame bytes',
    `face_count`            TINYINT UNSIGNED NOT NULL DEFAULT 0,
    `plate_count`           TINYINT UNSIGNED NOT NULL DEFAULT 0,
    `processing_latency_ms` FLOAT           NOT NULL COMMENT 'Total anonymization latency',
    `anonymization_applied` TINYINT(1)      NOT NULL DEFAULT 1,
    `detection_meta`        JSON            NULL     COMMENT 'Region bounding boxes (no PII)',
    `deleted_at`            DATETIME(6)     NULL     COMMENT 'Soft delete for GDPR Article 17',
    PRIMARY KEY (`id`),
    KEY `idx_de_session`    (`session_id`),
    KEY `idx_de_captured`   (`captured_at`),
    KEY `idx_de_hash`       (`frame_hash_sha256`),
    KEY `idx_de_deleted`    (`deleted_at`),
    CONSTRAINT `fk_de_session` FOREIGN KEY (`session_id`)
        REFERENCES `anonymization_sessions` (`session_id`)
        ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Per-frame anonymization results — no PII stored';

-- ---------------------------------------------------------------------------
-- Pseudonymised tracking ID registry
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `tracking_id_map` (
    `id`                    INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `pseudonym_hash`        CHAR(64)        NOT NULL COMMENT 'HMAC-SHA256 of original ID',
    `hmac_algorithm`        VARCHAR(20)     NOT NULL DEFAULT 'sha256',
    `first_seen_at`         DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `last_seen_at`          DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                                            ON UPDATE CURRENT_TIMESTAMP(6),
    `session_count`         INT UNSIGNED    NOT NULL DEFAULT 1,
    `is_active`             TINYINT(1)      NOT NULL DEFAULT 1
                                            COMMENT '0 = GDPR erasure requested',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_pseudonym_hash` (`pseudonym_hash`),
    KEY `idx_ti_active` (`is_active`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='HMAC pseudonyms only — original IDs never stored';

-- ---------------------------------------------------------------------------
-- Immutable audit trail (GDPR Article 30 / PDPA)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `audit_trail` (
    `id`                    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `event_id`              VARCHAR(36)     NOT NULL COMMENT 'UUID v4',
    `event_type`            VARCHAR(50)     NOT NULL COMMENT 'session|anonymization|access|deletion',
    `actor`                 VARCHAR(100)    NOT NULL DEFAULT 'system',
    `session_id`            VARCHAR(36)     NULL,
    `resource_type`         VARCHAR(50)     NULL,
    `resource_id`           VARCHAR(100)    NULL,
    `action`                VARCHAR(50)     NOT NULL,
    `outcome`               ENUM('success','failure','warning')
                                            NOT NULL DEFAULT 'success',
    `detail`                MEDIUMTEXT      NULL,
    `ip_address`            VARCHAR(45)     NULL,
    `occurred_at`           DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `compliance_tag`        VARCHAR(20)     NULL COMMENT 'e.g. GDPR-A30, GDPR-A17, PDPA',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_event_id` (`event_id`),
    KEY `idx_at_session`    (`session_id`),
    KEY `idx_at_type`       (`event_type`),
    KEY `idx_at_occurred`   (`occurred_at`),
    KEY `idx_at_compliance` (`compliance_tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Append-only compliance audit log — no UPDATE/DELETE via application';

-- ---------------------------------------------------------------------------
-- Useful views
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW `v_session_stats` AS
SELECT
    s.session_id,
    s.source_identifier,
    s.started_at,
    s.ended_at,
    s.status,
    COUNT(d.id)              AS total_frames,
    SUM(d.face_count)        AS total_faces,
    SUM(d.plate_count)       AS total_plates,
    AVG(d.processing_latency_ms) AS mean_latency_ms,
    MAX(d.processing_latency_ms) AS max_latency_ms,
    SUM(CASE WHEN d.processing_latency_ms <= 30 THEN 1 ELSE 0 END) AS frames_under_30ms
FROM anonymization_sessions s
LEFT JOIN detection_events d
    ON d.session_id = s.session_id AND d.deleted_at IS NULL
GROUP BY s.id;

CREATE OR REPLACE VIEW `v_daily_detection_summary` AS
SELECT
    DATE(captured_at)               AS detection_date,
    COUNT(*)                        AS total_frames,
    SUM(face_count)                 AS total_faces,
    SUM(plate_count)                AS total_plates,
    AVG(processing_latency_ms)      AS mean_latency_ms,
    MAX(processing_latency_ms)      AS max_latency_ms,
    COUNT(DISTINCT session_id)      AS unique_sessions
FROM detection_events
WHERE deleted_at IS NULL
GROUP BY DATE(captured_at)
ORDER BY detection_date DESC;
