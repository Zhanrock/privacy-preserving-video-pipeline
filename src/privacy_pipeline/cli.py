"""cli.py — Command-line entry points."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_db(args=None):
    """Helper: load config and return a DatabaseManager."""
    from privacy_pipeline.utils.config import load_config
    from privacy_pipeline.database.connection import DatabaseManager

    config_path = getattr(args, "config", "configs/default.yaml")
    if not os.path.exists(config_path):
        return DatabaseManager.for_testing()

    cfg = load_config(config_path)
    return DatabaseManager.from_config(cfg)


def migrate():
    """Entry point: ppvp-migrate — initialise or migrate the database schema."""
    parser = argparse.ArgumentParser(prog="ppvp-migrate")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-sqlite", action="store_true")
    args = parser.parse_args()

    from privacy_pipeline.utils.logger import configure_logging
    configure_logging("INFO")
    from privacy_pipeline.utils.logger import get_logger
    logger = get_logger(__name__)

    from privacy_pipeline.utils.config import load_config
    from privacy_pipeline.database.connection import DatabaseManager

    if os.path.exists(args.config):
        cfg = load_config(args.config)
        db  = DatabaseManager.from_config(cfg, force_sqlite=args.force_sqlite)
    else:
        logger.warning("Config not found at %s, using SQLite", args.config)
        db = DatabaseManager.for_testing()

    db.initialize_schema()
    logger.info("Migration complete. Backend: %s",
                "MySQL" if db.is_mysql() else "SQLite")


def process():
    """Entry point: ppvp-process — process a video file or image folder."""
    parser = argparse.ArgumentParser(prog="ppvp-process")
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--source",   required=True, help="Path to image or folder")
    parser.add_argument("--tracking-id", default=None)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    from privacy_pipeline.utils.logger import configure_logging
    configure_logging("DEBUG" if args.verbose else "INFO")
    from privacy_pipeline.utils.logger import get_logger
    logger = get_logger(__name__)

    import cv2
    import numpy as np
    from pathlib import Path
    from privacy_pipeline.utils.config import load_config
    from privacy_pipeline.database.connection import DatabaseManager
    from privacy_pipeline.pipeline.processor import PrivacyPipeline

    cfg = load_config(args.config) if os.path.exists(args.config) else None

    if cfg:
        db = DatabaseManager.from_config(cfg)
    else:
        db = DatabaseManager.for_testing()

    db.initialize_schema()

    source = Path(args.source)
    if source.is_file():
        image_files = [source]
    elif source.is_dir():
        image_files = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
    else:
        logger.error("Source not found: %s", source)
        sys.exit(1)

    if cfg:
        pipeline = PrivacyPipeline.from_config(
            cfg, db, source_identifier=str(source)
        )
    else:
        from privacy_pipeline.crypto.hasher import TrackingIDHasher
        from privacy_pipeline.anonymizer.detectors import FaceDetector, LicensePlateDetector
        from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
        pipeline = PrivacyPipeline(
            db=db,
            hasher=TrackingIDHasher(),
            face_detector=FaceDetector(),
            plate_detector=LicensePlateDetector(),
            face_blurrer=RegionBlurrer(method="pixelate", strength=25),
            plate_blurrer=RegionBlurrer(method="solid"),
            source_identifier=str(source),
        )

    with pipeline:
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Could not read: %s", img_path)
                continue
            result = pipeline.process_frame(frame, tracking_id=args.tracking_id)
            logger.info(
                "[%s] faces=%d plates=%d latency=%.2fms",
                img_path.name,
                result.face_count,
                result.plate_count,
                result.processing_latency_ms,
            )

    stats = pipeline.get_stats()
    logger.info("Done. Stats: %s", stats)


def report():
    """Entry point: ppvp-report — generate compliance report."""
    parser = argparse.ArgumentParser(prog="ppvp-report")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output",     default="reports/compliance.json")
    parser.add_argument("--days",       type=int, default=7)
    args = parser.parse_args()

    from privacy_pipeline.utils.logger import configure_logging
    configure_logging("INFO")
    from privacy_pipeline.utils.logger import get_logger
    logger = get_logger(__name__)

    db = _load_db(args)
    db.initialize_schema()

    from privacy_pipeline.pipeline.reporter import PipelineReporter
    reporter = PipelineReporter(db)

    report_data = reporter.compliance_report(
        session_id  = args.session_id,
        output_path = args.output,
    )
    logger.info(
        "Compliance report: total_events=%d, GDPR_A30=%s",
        report_data["total_events"],
        report_data.get("gdpr_article_30_met"),
    )
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    process()
