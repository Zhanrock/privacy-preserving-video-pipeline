"""utils/logger.py — Centralised logging factory."""
from __future__ import annotations
import logging, sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOG_FORMAT  = "%(asctime)s | %(name)-45s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured  = False


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = False,
) -> None:
    global _configured
    if _configured:
        return
    numeric = getattr(logging, level.upper(), logging.INFO)
    root    = logging.getLogger()
    root.setLevel(numeric)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    ch  = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_to_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(log_dir) / f"ppvp_{ts}.log"
        fh   = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    for noisy in ("PIL", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
