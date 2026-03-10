#!/usr/bin/env python3
"""
scripts/run_pipeline.py
------------------------
End-to-end demo: generates synthetic frames, runs the full anonymization
pipeline with MySQL (or SQLite fallback), and prints a compliance summary.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --frames 200 --source CAM-NORTH-01
    python scripts/run_pipeline.py --mysql --host localhost --db ppvp_db
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2

from privacy_pipeline.database.connection import DatabaseManager, ConnectionConfig
from privacy_pipeline.crypto.hasher import TrackingIDHasher
from privacy_pipeline.anonymizer.detectors import FaceDetector, LicensePlateDetector
from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
from privacy_pipeline.pipeline.processor import PrivacyPipeline
from privacy_pipeline.pipeline.reporter import PipelineReporter
from privacy_pipeline.utils.logger import configure_logging, get_logger


def generate_synthetic_frame(h=480, w=640, rng=None) -> np.ndarray:
    """Generate a random BGR frame for demo purposes."""
    if rng is None:
        rng = np.random.default_rng()
    frame = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Privacy Pipeline Demo")
    parser.add_argument("--frames",  type=int,  default=50)
    parser.add_argument("--source",  default="DEMO-CAM-001")
    parser.add_argument("--verbose", action="store_true")
    # MySQL options (uses SQLite by default)
    parser.add_argument("--mysql",    action="store_true")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=3306)
    parser.add_argument("--db-name",  default="privacy_pipeline_db")
    parser.add_argument("--user",     default="pp_user")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    configure_logging("DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)

    # Database
    if args.mysql:
        cfg = ConnectionConfig(
            host=args.host, port=args.port,
            database=args.db_name, user=args.user, password=args.password,
            fallback_to_sqlite=True,
        )
        db = DatabaseManager._create(cfg)
    else:
        db = DatabaseManager.for_testing()

    db.initialize_schema()
    backend = "MySQL" if db.is_mysql() else "SQLite (in-memory)"
    logger.info("Database backend: %s", backend)

    # Pipeline
    hasher = TrackingIDHasher(
        secret_key=os.environ.get("PPVP_HMAC_KEY", "demo_key_32_chars_minimum!!!!!").encode()
    )
    pipeline = PrivacyPipeline(
        db               = db,
        hasher           = hasher,
        face_detector    = FaceDetector(),
        plate_detector   = LicensePlateDetector(),
        face_blurrer     = RegionBlurrer(method="pixelate", strength=25),
        plate_blurrer    = RegionBlurrer(method="solid"),
        source_identifier = args.source,
        max_latency_ms   = 30.0,
    )

    logger.info("Processing %d synthetic frames...", args.frames)
    rng = np.random.default_rng(42)
    t0  = time.monotonic()

    for i in range(args.frames):
        frame = generate_synthetic_frame(rng=rng)
        result = pipeline.process_frame(
            frame,
            tracking_id=f"PERSON-{(i % 10):03d}",
        )
        if args.verbose:
            logger.debug(
                "Frame %3d: faces=%d plates=%d latency=%.2fms",
                i + 1, result.face_count, result.plate_count,
                result.processing_latency_ms,
            )

    elapsed = time.monotonic() - t0
    stats   = pipeline.close()

    # Report
    reporter = PipelineReporter(db)
    reporter.print_session_summary(pipeline.session_id)

    compliance = reporter.compliance_report(session_id=pipeline.session_id)
    print(f"\n  GDPR Article 30 compliant : {compliance['gdpr_article_30_met']}")
    print(f"  Total audit events        : {compliance['total_events']}")
    print(f"  Wall time                 : {elapsed:.2f}s")
    print(f"  Throughput                : {args.frames / elapsed:.1f} fps")
    print(f"  Mean latency              : {stats['mean_latency_ms']:.2f} ms")
    print(f"  Frames under 30ms target  : {stats['under_target_pct']:.1f}%")
    print()


if __name__ == "__main__":
    main()
