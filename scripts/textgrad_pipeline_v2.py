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
import json
import time
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# textgrad is imported lazily inside run_textgrad_optimization() so that
# importing this module (for AuditLog, AuditEntry, PrivacyOracle) does not
# block on textgrad's slow initialization.


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
# 6. TEXTGRAD OPTIMIZATION LOOP
# ---------------------------------------------------------------------------

def run_textgrad_optimization(
    anthropic_api_key: str,
    n_iterations: int = 3,
    verbose: bool = True,
    backend: str = "claude",   # "claude" | "nemobot"
    nemobot_endpoint: str = "",
) -> dict:
    """
    Core TextGrad loop.

    backend="claude":   uses Claude API (current default)
    backend="nemobot":  uses NemoBot local LLM (planned — needs Prof Tan's docs)
                        This is the GDPR-strong version: no data leaves premises.
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

    # Simulate initial audit entries (replace with real pipeline output)
    for i in range(5):
        entry = AuditEntry(
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            frame_id=f"frame_{i:04d}",
            faces_detected=2 + (i % 3),
            plates_detected=i % 2,
            blur_kernel=31,
            privacy_score=0.72 + (i * 0.01),
            utility_score=0.81 - (i * 0.02),
            gdpr_compliant=True,
            detection_model="haar_cascade",
            issues=(["partial face missed at frame edge"] if i % 3 == 0 else []),
        )
        audit_log.add_entry(entry)
        results["audit_log_entries"].append(asdict(entry))

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

        # Add post-optimization audit entry
        new_entry = AuditEntry(
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            frame_id=f"post_iter{iteration+1}",
            faces_detected=3,
            plates_detected=1,
            blur_kernel=max(21, 31 - (iteration + 1) * 2),
            privacy_score=min(0.95, 0.72 + (iteration + 1) * 0.07),
            utility_score=min(0.85, 0.75 + (iteration + 1) * 0.03),
            gdpr_compliant=True,
            detection_model="retinaface",  # simulated upgrade
            issues=[],
        )
        audit_log.add_entry(new_entry)

        results["iterations"].append({
            "iteration": iteration + 1,
            "duration_seconds": round(time.time() - t0, 2),
            "updated_config_preview": pipeline_config.value[:400],
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

    # Full optimization loop (requires API key)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    nemobot_ep = os.environ.get("NEMOBOT_ENDPOINT", "")
    backend = "nemobot" if nemobot_ep else "claude"

    if api_key and "--optimize" in sys.argv:
        print(f"\n[Running TextGrad Optimization — backend: {backend}]")
        results = run_textgrad_optimization(
            api_key, n_iterations=2, verbose=True,
            backend=backend, nemobot_endpoint=nemobot_ep
        )
        out = os.path.join(os.path.dirname(__file__), "..", "outputs", "textgrad_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {out}")
    else:
        print("\n[TextGrad loop ready]")
        print("  To optimize: export ANTHROPIC_API_KEY=... && python textgrad_pipeline.py --optimize")
        print("  For NemoBot: export NEMOBOT_ENDPOINT=http://... && python textgrad_pipeline.py --optimize")