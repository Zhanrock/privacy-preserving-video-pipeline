"""Shared test fixtures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

def make_frame(h=64, w=64, color=(120, 80, 60)):
    """Create a solid-colour BGR test frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame

def make_db():
    """Create a fresh in-memory SQLite DatabaseManager."""
    from privacy_pipeline.database.connection import DatabaseManager
    return DatabaseManager.for_testing()
