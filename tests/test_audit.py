import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from textgrad_pipeline_v2 import AuditLog, AuditEntry
import datetime

log = AuditLog()

# Simulate 5 frames being processed
for i in range(5):
    entry = AuditEntry(
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        frame_id=f"frame_{i:04d}",
        faces_detected=2 + (i % 3),
        plates_detected=i % 2,
        blur_kernel=31,
        privacy_score=0.70 + (i * 0.02),
        utility_score=0.80 - (i * 0.01),
        gdpr_compliant=True,
        detection_model="haar_cascade",
        issues=["partial face missed"] if i % 2 == 0 else []
    )
    log.add_entry(entry)
    print(f"Frame {i}: privacy={entry.privacy_score:.2f}, "
          f"utility={entry.utility_score:.2f}, "
          f"hash={entry.entry_hash[:16]}...")

print()
print("Chain verified:", log.verify_chain())
print()
print("=== TEXTUAL FEEDBACK (this becomes the TextGrad loss signal) ===")
print(log.generate_textual_feedback())