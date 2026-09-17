"""
TextGrad Integration for Privacy-Preserving Video Analytics Pipeline
=====================================================================
Research implementation — NTU CCDS
Advisor: Assoc Prof Chee Wei Tan
Student: Krittaphas Thanaphongphaisan (Year 1, Computer Engineering)
Target venue: USENIX Security / NSDI 2026

Architecture (three layers):
  DOWNSTREAM  — Pre-trained models (Haar Cascade → RetinaFace / NVIDIA NIM)
                LoRA adapter for deployment-specific fine-tuning (planned)
  UPSTREAM    — Tamper-evident GDPR audit log (Art.30 / Art.17)
                SHA3-256 hash chain, natural language feedback generation
  TEXTGRAD    — Upstream validation / optimization loop
                LLM backend: Claude API (default) OR NemoBot local LLM (planned)

How to run:
  Demo only (no API key):
      python textgrad_pipeline.py

  Full TextGrad optimization:
      export ANTHROPIC_API_KEY=sk-ant-...
      python textgrad_pipeline.py --optimize

  With NemoBot local LLM (once NemoBot access confirmed with Prof Tan):
      export NEMOBOT_ENDPOINT=http://localhost:8080
      python textgrad_pipeline.py --optimize --backend nemobot

Updates (April 2026):
  - Detection model: Haar Cascade → RetinaFace (drop-in upgrade, see Section 6)
  - Added NemoBot backend stub (pending API documentation from Prof Tan)
  - Added LoRA adapter notes for Art.17 erasure argument
  - Benchmark table updated with three-era framing
"""

import os
import re
import sys
import json
import time
import hashlib
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT / "src"), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# textgrad, and the detector/WiderFace grounding helpers (benchmark_detectors,
# privacy_pipeline.anonymizer.detectors), are imported lazily inside the
# functions that need them, so importing this module (for AuditLog,
# AuditEntry, PrivacyOracle) does not block on their slower initialization.


# ---------------------------------------------------------------------------
# 1. PIPELINE CONFIGURATION (TextGrad Variable — the thing we optimize)
# ---------------------------------------------------------------------------

INITIAL_PIPELINE_CONFIG = """
Privacy-Preserving Video Analytics Pipeline Configuration
==========================================================

Detection Model:
  - Current: Haar Cascade (cv2.CascadeClassifier) — 2001 baseline, CPU-only
  - Planned upgrade: RetinaFace — current SOTA, handles occluded/angled faces
  - Prof Tan recommendation: NVIDIA NIM detection model via NemoBot API
  - LoRA adapter: optional fine-tuning layer for deployment-specific context
    (adapter file is separable — enables Art.17 erasure by deleting adapter)

Face Detection Parameters:
  - Blur strength: Gaussian kernel size = 31
  - Blur sigma: 15
  - Detection confidence threshold: 0.6
  - Padding around detected face: 10px

License Plate Detection:
  - Method: MSER (Maximally Stable Extremal Regions)
  - Blur kernel size: 25

Metadata Encryption:
  - Algorithm: AES-256-GCM
  - Key derivation: Argon2id
  - Pseudonymisation: HMAC-SHA256

Audit Logging (Upstream Layer — GDPR Art.30 / Art.17):
  - Format: Structured JSON with ISO 8601 timestamps
  - Chain of custody: SHA3-256 hash chaining (tamper-evident)
  - Art.30 fields: processing purpose, data subject category, retention period
  - Art.17 fields: deletion request tracking, erasure confirmation
  - Feedback: natural language summary fed into TextGrad as loss signal

Processing Pipeline Order:
  1. Ingest raw video frame
  2. Detect faces (Haar Cascade / RetinaFace / NVIDIA NIM)
  3. Detect license plates (MSER)
  4. Apply privacy-preserving transformations (blur/mask)
  5. Encrypt and pseudonymise metadata
  6. Write tamper-evident audit log entry (SHA3-256 hash chain)
  7. Upstream: generate textual feedback from audit log
  8. TextGrad: receive feedback, compute loss, apply textual gradient
  9. Output anonymised frame + encrypted metadata + updated config
"""


# ---------------------------------------------------------------------------
# 2. DETECTION MODEL BACKENDS
# ---------------------------------------------------------------------------

class DetectionBackend:
    """
    Abstraction layer for face detection models.
    Swap between Haar Cascade (baseline), RetinaFace (upgrade),
    and NVIDIA NIM (Prof Tan recommendation) without changing pipeline logic.
    """

    @staticmethod
    def get_haar_cascade():
        """
        Original baseline — Haar Cascade (Viola-Jones, 2001).
        Fast on CPU, poor on occluded/angled/distant faces.
        Suitable as the ~2022 CNN baseline comparison point.
        """
        import cv2
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def get_retinaface():
        """
        Recommended upgrade — RetinaFace via InsightFace (CVPR 2020).
        Handles occluded faces, varied angles, multiple scales.
        Install: pip install insightface onnxruntime
        Note: retina-face (pip install retina-face) requires TensorFlow which
        has no Python 3.14 build. insightface uses ONNX Runtime instead.

        Usage:
            detector = DetectionBackend.get_retinaface()
            faces = detector.get(img)   # img = BGR numpy array (cv2.imread)
            count = len(faces)          # each face has .bbox, .kps, .det_score
        """
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except ImportError:
            print("[WARNING] insightface not installed. Run: pip install insightface onnxruntime")
            print("[FALLBACK] Using Haar Cascade instead.")
            return DetectionBackend.get_haar_cascade()

    @staticmethod
    def get_nvidia_nim(endpoint: str = None):
        """
        NVIDIA NIM detection model — Prof Tan's recommendation.
        Requires access to Prof Tan's DGX Spark or NVIDIA API endpoint.

        TODO (Taishan): Ask Prof Tan which specific NVIDIA NIM model he
        recommends for face detection. Get the endpoint URL.
        Likely candidate: NVIDIA Metropolis / TAO face detection model.

        Once endpoint is confirmed:
            endpoint = "http://ntu-dgx-endpoint/face-detect"
        """
        if endpoint is None:
            endpoint = os.environ.get("NVIDIA_NIM_ENDPOINT", "")
        if not endpoint:
            print("[WARNING] NVIDIA_NIM_ENDPOINT not set. Falling back to RetinaFace.")
            return DetectionBackend.get_retinaface()
        # Stub: real implementation calls the NIM REST API
        print(f"[INFO] NVIDIA NIM endpoint configured: {endpoint}")
        return {"type": "nvidia_nim", "endpoint": endpoint}


# ---------------------------------------------------------------------------
# 3. NEMOBOT LLM BACKEND (planned — pending API docs from Prof Tan)
# ---------------------------------------------------------------------------

class NemoBotEngine:
    """
    NemoBot — Prof Tan's lab system (NTU CCDS, developed since 2018).
    Enables LLMs to communicate local↔cloud via API.

    Why this matters for the paper:
    - Using Claude API for TextGrad optimization means audit log data
      (which contains personal data metadata) leaves the operator's
      infrastructure. This is a GDPR concern.
    - NemoBot routes the same calls to a LOCAL LLM, keeping everything
      on-premise. This makes the privacy story fully self-contained.
    - This is a strong contribution: the entire system (pipeline + optimizer)
      runs without any external API dependency.

    TODO (Taishan): Ask Prof Tan for:
      1. NemoBot API documentation or GitHub repo
      2. Example of pointing a TextGrad engine at a NemoBot endpoint
      3. Which local model he recommends for the optimization loop
         (DeepSeek? Llama 3? Something from NVIDIA NeMo?)

    Integration target:
        tg.set_backward_engine(NemoBotEngine(endpoint="http://..."))
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Stub: real implementation POSTs to NemoBot endpoint.
        NemoBot routes to local or cloud LLM based on its configuration.
        """
        import urllib.request
        payload = json.dumps({"prompt": prompt, "max_tokens": 1000}).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())["response"]
        except Exception as e:
            print(f"[NemoBot] Connection failed: {e}")
            print("[NemoBot] Falling back to Claude API")
            return None


# ---------------------------------------------------------------------------
# 4. AUDIT LOG — upstream feedback signal
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    timestamp: str
    frame_id: str
    faces_detected: int
    plates_detected: int
    blur_kernel: int
    privacy_score: float        # 0.0 = fully exposed, 1.0 = fully private
    utility_score: float        # 0.0 = useless, 1.0 = full CV utility
    gdpr_compliant: bool
    detection_model: str = "haar_cascade"  # track which model was used
    issues: list = field(default_factory=list)
    prev_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        content = json.dumps(
            {k: v for k, v in asdict(self).items() if k != "entry_hash"},
            sort_keys=True
        )
        return hashlib.sha3_256(content.encode()).hexdigest()


class AuditLog:
    """
    Tamper-evident audit log using SHA3-256 hash chaining.
    GDPR Art.30: records of processing activities.
    GDPR Art.17: erasure request tracking.

    This is the UPSTREAM component.
    generate_textual_feedback() converts the log into the TextGrad loss signal.
    """

    def __init__(self):
        self.entries: list[AuditEntry] = []
        self._prev_hash = "0" * 64

    def add_entry(self, entry: AuditEntry) -> AuditEntry:
        entry.prev_hash = self._prev_hash
        entry.entry_hash = entry.compute_hash()
        self._prev_hash = entry.entry_hash
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.entries:
            if e.prev_hash != prev:
                return False
            if e.entry_hash != e.compute_hash():
                return False
            prev = e.entry_hash
        return True

    def generate_textual_feedback(self) -> str:
        """
        Convert audit log into natural language for TextGrad.
        This is the loss signal — the thing the LLM reads to understand
        how well the current config is performing.
        """
        if not self.entries:
            return "No audit entries yet."

        recent = self.entries[-min(5, len(self.entries)):]
        avg_privacy = sum(e.privacy_score for e in recent) / len(recent)
        avg_utility = sum(e.utility_score for e in recent) / len(recent)
        all_issues = [issue for e in recent for issue in e.issues]
        models_used = list(set(e.detection_model for e in recent))

        return f"""
Audit Log Analysis (last {len(recent)} frames):

Privacy Score:   {avg_privacy:.2f}/1.00  {'[GOOD]' if avg_privacy >= 0.85 else '[NEEDS IMPROVEMENT — target >= 0.90]'}
Utility Score:   {avg_utility:.2f}/1.00  {'[GOOD]' if avg_utility >= 0.70 else '[NEEDS IMPROVEMENT — target >= 0.75]'}
P×U Product:     {avg_privacy * avg_utility:.2f}  (higher = better balance)
Chain Integrity: {'VERIFIED' if self.verify_chain() else 'COMPROMISED — GDPR incident'}
Detection Model: {', '.join(models_used)}

Issues detected:
{chr(10).join(f'  - {i}' for i in all_issues) if all_issues else '  None'}

GDPR Compliance:
  Art.30 (processing records): {'COMPLIANT' if all(e.gdpr_compliant for e in recent) else 'NON-COMPLIANT'}
  Art.17 (erasure tracking): present in schema
  Chain of custody: {'INTACT' if self.verify_chain() else 'BROKEN'}

Optimization target:
  Raise privacy to >= 0.90 while keeping utility >= 0.75.
  Ensure all Art.30 required fields are present in every entry.
""".strip()


# ---------------------------------------------------------------------------
# 5. ORACLE EVALUATOR
# ---------------------------------------------------------------------------

class PrivacyOracle:
    """
    Oracle = theoretical upper bound / gold standard.
    In the paper: oracle = human expert evaluation (expensive).
    TextGrad approximates oracle feedback via LLM.
    Future option: NemoBot local LLM as oracle judge (no external API).
    """

    @staticmethod
    def evaluate_config(config_text: str, audit_feedback: str) -> dict:
        metrics = {"gdpr_gaps": [], "optimization_targets": []}
        c = config_text.lower()

        if "kernel size = 31" in c or "kernel size=31" in c:
            metrics["optimization_targets"].append(
                "Fixed blur kernel 31 — consider adaptive sizing based on face distance."
            )
        if "haar cascade" in c and "retinaface" not in c:
            metrics["optimization_targets"].append(
                "Haar Cascade misses occluded and angled faces — upgrade to RetinaFace."
            )
        if "confidence threshold: 0.6" in c:
            metrics["optimization_targets"].append(
                "Detection threshold 0.6 may miss partial faces — consider lowering to 0.45."
            )
        if "retention period" not in c:
            metrics["gdpr_gaps"].append(
                "Art.30 requires explicit retention period — missing from config."
            )
        if "data subject category" not in c:
            metrics["gdpr_gaps"].append(
                "Art.30 requires data subject category classification."
            )
        return metrics

    @staticmethod
    def build_loss_description(audit_feedback: str, oracle_eval: dict) -> str:
        gaps = "\n".join(f"  - {g}" for g in oracle_eval["gdpr_gaps"]) or "  None"
        targets = "\n".join(f"  - {t}" for t in oracle_eval["optimization_targets"]) or "  None"
        return f"""
You are evaluating a privacy-preserving video analytics pipeline configuration.
Optimize for best privacy-utility tradeoff while ensuring full GDPR compliance.
The system operator can ONLY control upstream configuration — not the base model weights.

Current performance (from audit log):
{audit_feedback}

GDPR compliance gaps:
{gaps}

Optimization targets:
{targets}

Provide specific, actionable configuration changes to:
1. Raise privacy score to >= 0.90
2. Keep utility score at >= 0.75
3. Achieve full GDPR Art.30/17 compliance
Be concrete — suggest exact parameter values.
""".strip()


# ---------------------------------------------------------------------------
# 6. REAL-PIPELINE GROUNDING
# ---------------------------------------------------------------------------
# The TextGrad loop rewrites INITIAL_PIPELINE_CONFIG as free-text natural
# language. Everything in this section turns that text back into a runnable
# detector, executes it against real WiderFace images, and returns MEASURED
# metrics — replacing the previous hardcoded-formula AuditEntry values
# (privacy_score=min(0.95, 0.72 + iter*0.07), etc.) with numbers a detector
# actually produced. Without this, the loop optimizes against fiction.

def _load_grounding_sample(sample_size: int, seed: int = 42):
    """Fixed WiderFace image sample reused across every iteration, so
    iteration-to-iteration score changes reflect the config change, not a
    different sample."""
    from benchmark_detectors import parse_widerface_gt, load_entries, WIDER_VAL_GT, WIDER_VAL_IMAGES

    if not WIDER_VAL_GT.exists() or not WIDER_VAL_IMAGES.exists():
        raise FileNotFoundError(
            f"WiderFace dataset not found under {WIDER_VAL_IMAGES.parent}. "
            "The TextGrad loop measures real detector performance every "
            "iteration and needs this dataset — see benchmark_detectors.py "
            "for the expected layout."
        )
    gt = parse_widerface_gt(WIDER_VAL_GT)
    return load_entries(gt, WIDER_VAL_IMAGES, max_images=sample_size, seed=seed)


def parse_config_to_detector_spec(config_text: str) -> dict:
    """
    Extract concrete detector parameters from TextGrad's rewritten config
    text. TextGrad edits INITIAL_PIPELINE_CONFIG as natural language (it is
    explicitly prompted to "suggest exact parameter values"), so this is a
    best-effort regex extraction, not a structured parser — free-text
    rewrites can phrase a value in ways the patterns below miss. Anything
    not found falls back to the current default rather than raising, and
    the extracted spec is logged with every iteration so the paper can show
    parser fidelity across a run.
    """
    c = config_text.lower()

    if "retinaface" in c:
        detector_type = "retinaface"
    elif "yunet" in c:
        detector_type = "yunet"
    elif "nvidia nim" in c or "nemobot api" in c:
        # Not runnable locally — ground against the strongest detector we
        # can actually execute rather than skip measurement.
        detector_type = "yunet"
    else:
        detector_type = "haar_cascade"

    conf_match = re.search(r"confidence threshold[^\d]*?([01]\.\d+)", c)
    conf_threshold = float(conf_match.group(1)) if conf_match else 0.45

    pad_match = re.search(r"padding[^\d]*?(\d+)\s*px", c)
    padding = int(pad_match.group(1)) if pad_match else 15

    kernel_match = re.search(r"kernel size[^\d]*?(\d+)", c)
    blur_kernel = int(kernel_match.group(1)) if kernel_match else 31

    return {
        "detector_type": detector_type,
        "conf_threshold": conf_threshold,
        "padding": padding,
        "blur_kernel": blur_kernel,
    }


def measure_real_performance(config_text: str, sample, verbose: bool = False) -> dict:
    """
    Instantiate the detector implied by `config_text` and run it over the
    fixed WiderFace `sample`, returning real aggregate metrics (recall, FNR,
    privacy/utility score, latency) via benchmark_detectors.run_detector —
    the same scoring path used for the paper's WiderFace benchmark table, so
    TextGrad-loop numbers and benchmark-table numbers are directly comparable.
    """
    from benchmark_detectors import run_detector
    from privacy_pipeline.anonymizer.detectors import create_face_detector, LicensePlateDetector
    import cv2

    spec = parse_config_to_detector_spec(config_text)
    model_path = str(REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx")

    # padding=0 for the measurement detector: IoU-based recall/FNR must be
    # scored against the raw detection box, not the blur-padded one — padding
    # a near-MIN_FACE_PX box before matching it to tight WiderFace ground
    # truth can push IoU below the 0.5 threshold and misscore a correct
    # detection as a miss (see benchmark_detectors.py for the same fix and
    # the measurement that found it: padding=15 -> recall=0.51 vs
    # padding=0 -> recall=0.92 on an identical sample). The config's actual
    # padding value is still recorded in `spec` for the audit trail — it's a
    # real anonymization-margin parameter, just not one that should affect
    # whether a face counts as "found."
    def _build(detector_type: str):
        if detector_type == "yunet":
            return create_face_detector(
                "yunet", model_path=model_path,
                conf_threshold=spec["conf_threshold"], padding=0,
            )
        if detector_type == "retinaface":
            return create_face_detector(
                "retinaface",
                conf_threshold=spec["conf_threshold"], padding=0,
            )
        return create_face_detector("haar_cascade", padding=0)

    try:
        detector = _build(spec["detector_type"])
    except ImportError as e:
        if verbose:
            print(f"[WARNING] {spec['detector_type']} unavailable ({e}); measuring with YuNet instead.")
        spec["detector_type"] = "yunet"
        detector = _build("yunet")

    outcome = run_detector(detector, sample)
    summary = outcome.get("summary", {})
    per_image = outcome.get("per_image", [])
    if not summary:
        raise RuntimeError("Detector produced no measurable output on the grounding sample.")

    # Real license-plate candidate count. WiderFace has no plate ground truth,
    # so this is a raw detection count from the actual MSER detector — not an
    # accuracy metric — capped to a few frames to keep iterations fast.
    plate_detector = LicensePlateDetector()
    plate_counts = []
    for img_path, _gt in sample[:10]:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            plate_counts.append(len(plate_detector.detect(frame)))
    avg_plates = round(sum(plate_counts) / len(plate_counts)) if plate_counts else 0
    avg_faces = round(sum(m["detected"] for m in per_image) / len(per_image)) if per_image else 0

    issues = []
    if summary["overall_fnr"] > 0.10:
        issues.append(
            f"{summary['overall_fnr']:.1%} of ground-truth faces missed on this "
            f"iteration's {summary['images_tested']}-image sample "
            f"({summary['total_fn']}/{summary['total_gt_faces']})"
        )

    return {
        "spec": spec,
        "detection_model": spec["detector_type"],
        "faces_detected": avg_faces,
        "plates_detected": avg_plates,
        "privacy_score": summary["mean_privacy"],
        "utility_score": summary["mean_utility"],
        "recall": summary["overall_recall"],
        "fnr": summary["overall_fnr"],
        "latency_ms": summary["mean_latency_ms"],
        "images_tested": summary["images_tested"],
        "issues": issues,
    }


def _seed_real_audit_entries(audit_log: "AuditLog", results: dict, sample, n: int = 5) -> None:
    """Seed the log with `n` real per-frame measurements from the INITIAL
    config's detector (Haar Cascade), replacing the old
    ``privacy_score=0.72 + i*0.01``-style hardcoded formula."""
    from benchmark_detectors import run_detector
    from privacy_pipeline.anonymizer.detectors import create_face_detector

    detector = create_face_detector("haar_cascade", padding=0)  # see measure_real_performance
    outcome = run_detector(detector, sample[:n])

    for m in outcome.get("per_image", []):
        entry = AuditEntry(
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            frame_id=Path(m["image"]).stem,
            faces_detected=m["detected"],
            plates_detected=0,
            blur_kernel=31,
            privacy_score=m["privacy_score"],
            utility_score=m["utility_score"],
            gdpr_compliant=True,
            detection_model="haar_cascade",
            issues=(
                [f"{m['fnr']:.0%} FNR on this frame ({m['fn']}/{m['total_gt']} faces missed)"]
                if m["fn"] > 0 else []
            ),
        )
        audit_log.add_entry(entry)
        results["audit_log_entries"].append(asdict(entry))


# ---------------------------------------------------------------------------
# 7. TEXTGRAD OPTIMIZATION LOOP
# ---------------------------------------------------------------------------

def run_textgrad_optimization(
    anthropic_api_key: str,
    n_iterations: int = 3,
    verbose: bool = True,
    backend: str = "claude",   # "claude" | "nemobot"
    nemobot_endpoint: str = "",
    grounding_sample_size: int = 40,
    grounding_seed: int = 42,
) -> dict:
    """
    Core TextGrad loop.

    backend="claude":   uses Claude API (current default)
    backend="nemobot":  uses NemoBot local LLM (planned — needs Prof Tan's docs)
                        This is the GDPR-strong version: no data leaves premises.

    Every iteration's AuditEntry is built from `measure_real_performance()` —
    an actual detector run over a fixed WiderFace sample — not a formula.
    """

    import textgrad as tg  # lazy import — only needed for the optimization loop

    results = {
        "backend": backend,
        "iterations": [],
        "initial_config": INITIAL_PIPELINE_CONFIG,
        "final_config": None,
        "audit_log_entries": [],
        "optimization_successful": False,
    }

    # Setup backend
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if backend == "nemobot" and nemobot_endpoint:
        print(f"[INFO] Using NemoBot backend: {nemobot_endpoint}")
        # TODO: wire NemoBotEngine into tg.set_backward_engine once API confirmed
        # For now fall through to Claude
        print("[WARNING] NemoBot TextGrad integration pending. Using Claude.")

    tg.set_backward_engine("claude-haiku-4-5-20251001", override=True)
    engine = tg.get_engine("claude-haiku-4-5-20251001")

    # Pipeline config as TextGrad Variable
    pipeline_config = tg.Variable(
        value=INITIAL_PIPELINE_CONFIG,
        requires_grad=True,
        role_description=(
            "Privacy-preserving video analytics pipeline configuration. "
            "Controls face detection model, blur parameters, and GDPR audit logging. "
            "Optimize for maximum privacy-utility tradeoff while maintaining compliance."
        ),
    )

    audit_log = AuditLog()
    oracle = PrivacyOracle()
    optimizer = tg.TGD(parameters=[pipeline_config])

    sample = _load_grounding_sample(grounding_sample_size, grounding_seed)
    if verbose:
        print(f"[Grounding] {len(sample)} WiderFace images loaded "
              f"(seed={grounding_seed}) — reused across all {n_iterations} iterations.")

    # Seed the log with real measurements from the INITIAL config's detector,
    # instead of a hardcoded formula.
    _seed_real_audit_entries(audit_log, results, sample, n=5)

    # TextGrad loop
    for iteration in range(n_iterations):
        t0 = time.time()
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{n_iterations}")
            print(f"{'='*60}")

        audit_feedback = audit_log.generate_textual_feedback()
        oracle_eval = oracle.evaluate_config(pipeline_config.value, audit_feedback)
        loss_desc = oracle.build_loss_description(audit_feedback, oracle_eval)

        loss_fn = tg.TextLoss(eval_system_prompt=loss_desc, engine=engine)
        loss = loss_fn(pipeline_config)

        if verbose:
            print(f"\n[Loss]:\n{loss.value[:600]}...")

        optimizer.zero_grad()
        loss.backward()

        if verbose and pipeline_config.gradients:
            grad = list(pipeline_config.gradients)[0]
            print(f"\n[Textual Gradient]:\n{str(grad)[:600]}...")

        optimizer.step()

        # Add a post-optimization audit entry built from a real detector run
        # over the fixed grounding sample, using whatever parameters
        # TextGrad's rewritten config now specifies.
        measured = measure_real_performance(pipeline_config.value, sample, verbose=verbose)
        new_entry = AuditEntry(
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            frame_id=f"post_iter{iteration+1}",
            faces_detected=measured["faces_detected"],
            plates_detected=measured["plates_detected"],
            blur_kernel=measured["spec"]["blur_kernel"],
            privacy_score=measured["privacy_score"],
            utility_score=measured["utility_score"],
            gdpr_compliant=not oracle_eval["gdpr_gaps"],
            detection_model=measured["detection_model"],
            issues=measured["issues"],
        )
        audit_log.add_entry(new_entry)

        if verbose:
            print(f"\n[Measured — {measured['detection_model']}, "
                  f"{measured['images_tested']} images]: "
                  f"privacy={measured['privacy_score']:.3f} "
                  f"utility={measured['utility_score']:.3f} "
                  f"recall={measured['recall']:.3f} fnr={measured['fnr']:.3f}")

        results["iterations"].append({
            "iteration": iteration + 1,
            "duration_seconds": round(time.time() - t0, 2),
            "updated_config_preview": pipeline_config.value[:400],
            "measured": {k: v for k, v in measured.items() if k != "spec"},
            "detector_spec": measured["spec"],
        })

        if verbose:
            print(f"\n[Updated Config Preview]:\n{pipeline_config.value[:400]}...")

    results["final_config"] = pipeline_config.value
    results["audit_chain_verified"] = audit_log.verify_chain()
    results["optimization_successful"] = True
    return results


# ---------------------------------------------------------------------------
# 7. BENCHMARK TABLE
# ---------------------------------------------------------------------------

def generate_benchmark() -> dict:
    return {
        "description": (
            "Three-era comparison: pre-AI rule-based (~2018), "
            "AI CNN baseline (~2022), PrivGrad TextGrad-optimized (this work)."
        ),
        "eras": [
            {
                "era": "Pre-AI (~2018)",
                "detection": "Fixed Gaussian blur, no ML",
                "privacy": 0.85, "utility": 0.45, "pxu": 0.38,
                "gdpr": "Partial (manual records)",
                "optimization": "None",
            },
            {
                "era": "CNN Baseline (~2022)",
                "detection": "Haar Cascade + fixed blur",
                "privacy": 0.72, "utility": 0.78, "pxu": 0.56,
                "gdpr": "Partial (automated, incomplete)",
                "optimization": "Manual tuning only",
            },
            {
                "era": "PrivGrad (This Work)",
                "detection": "RetinaFace / NVIDIA NIM + LoRA adapter",
                "privacy": "TBD", "utility": "TBD", "pxu": "TBD",
                "gdpr": "Full Art.30 + Art.17 automated",
                "optimization": "TextGrad upstream loop (NemoBot or Claude backend)",
            },
        ],
    }


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("PrivGrad — Privacy-Preserving Video Analytics")
    print("NTU CCDS | Assoc Prof Chee Wei Tan | April 2026")
    print("=" * 60)

    # Always run: benchmark + audit log demo
    bench = generate_benchmark()
    print("\n[Benchmark: Three Eras]")
    for era in bench["eras"]:
        print(f"  {era['era']}")
        print(f"    Detection:   {era['detection']}")
        print(f"    Privacy:     {era['privacy']}  Utility: {era['utility']}  P×U: {era['pxu']}")
        print(f"    Optimization:{era['optimization']}")
        print()

    print("[Audit Log Chain Integrity Demo]")
    log = AuditLog()
    for i in range(3):
        e = AuditEntry(
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            frame_id=f"frame_{i:04d}",
            faces_detected=2, plates_detected=0,
            blur_kernel=31, privacy_score=0.80 + i * 0.05,
            utility_score=0.75, gdpr_compliant=True,
            detection_model="haar_cascade",
        )
        log.add_entry(e)
    print(f"  Entries: {len(log.entries)} | Chain verified: {log.verify_chain()}")
    print()
    print("[Textual Feedback Preview (this becomes the TextGrad loss)]:")
    print(log.generate_textual_feedback())

    def _get_int_arg(flag: str, default: int) -> int:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                try:
                    return int(sys.argv[idx + 1])
                except ValueError:
                    pass
        return default

    # Full optimization loop (requires API key)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    nemobot_ep = os.environ.get("NEMOBOT_ENDPOINT", "")
    backend = "nemobot" if nemobot_ep else "claude"

    if "--test-grounding" in sys.argv:
        # Exercises the real-detector grounding path with no API key and no
        # TextGrad/LLM calls — for verifying the measurement code in isolation.
        sample_size = _get_int_arg("--sample-size", 40)
        print(f"\n[Testing grounding — {sample_size} WiderFace images, no API key needed]")
        sample = _load_grounding_sample(sample_size, seed=42)
        measured = measure_real_performance(INITIAL_PIPELINE_CONFIG, sample, verbose=True)
        print(json.dumps(measured, indent=2, default=str))
    elif api_key and "--optimize" in sys.argv:
        n_iter = _get_int_arg("--iterations", 10)
        sample_size = _get_int_arg("--sample-size", 40)
        print(f"\n[Running TextGrad Optimization — backend: {backend}, "
              f"{n_iter} iterations, grounding sample={sample_size} WiderFace images]")
        results = run_textgrad_optimization(
            api_key, n_iterations=n_iter, verbose=True,
            backend=backend, nemobot_endpoint=nemobot_ep,
            grounding_sample_size=sample_size,
        )
        out = os.path.join(os.path.dirname(__file__), "..", "outputs", "textgrad_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {out}")
    else:
        print("\n[TextGrad loop ready]")
        print("  To optimize: export ANTHROPIC_API_KEY=... && python textgrad_pipeline_v2.py --optimize --iterations 10")
        print("  For NemoBot: export NEMOBOT_ENDPOINT=http://... && python textgrad_pipeline_v2.py --optimize")
        print("  To test grounding only (no API key): python textgrad_pipeline_v2.py --test-grounding")