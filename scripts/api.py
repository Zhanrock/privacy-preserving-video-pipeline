"""
PrivGrad TextGrad API
=====================
FastAPI wrapper exposing the audit log and TextGrad optimization loop.

Endpoints:
  GET  /            — health check
  POST /analyze     — add an audit entry, return textual feedback
  POST /optimize    — run TextGrad loop (needs ANTHROPIC_API_KEY) or mock

Run:
  cd scripts/
  uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger UI.
"""

import datetime
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow importing textgrad_pipeline_v2 from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from textgrad_pipeline_v2 import (
    INITIAL_PIPELINE_CONFIG,
    AuditEntry,
    AuditLog,
    PrivacyOracle,
)

app = FastAPI(
    title="PrivGrad TextGrad API",
    description=(
        "GDPR audit log + TextGrad optimization loop. "
        "NTU CCDS — Assoc Prof Chee Wei Tan."
    ),
    version="1.0.0",
)

# NemoBot's postprocess JS runs in the browser tab, on the NemoBot page's own
# origin (e.g. https://<something>.vercel.app) — its fetch() to localhost:8000
# is cross-origin from the browser's perspective, so without CORS headers the
# browser blocks the response even though this server is reachable. Local-only
# dev API, not exposed publicly, so allow_origins=["*"] is fine here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level audit log — persists across requests for the lifetime of the server
_audit_log = AuditLog()
_oracle = PrivacyOracle()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    frame_id: str = Field(..., example="frame_0001")
    faces_detected: int = Field(..., ge=0, example=3)
    privacy_score: float = Field(..., ge=0.0, le=1.0, example=0.75)
    utility_score: float = Field(..., ge=0.0, le=1.0, example=0.82)
    blur_kernel: int = Field(31, ge=1, example=31)
    issues: list[str] = Field(default_factory=list, example=["partial face missed"])


class AnalyzeResponse(BaseModel):
    textual_feedback: str
    chain_verified: bool
    entry_hash: str
    total_entries: int


class OptimizeRequest(BaseModel):
    audit_feedback: str = Field(..., example="Privacy Score: 0.74/1.00 [NEEDS IMPROVEMENT]")


class OptimizeResponse(BaseModel):
    updated_config: str
    optimization_ran: bool
    backend: str
    message: str
    iterations: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "service": "PrivGrad TextGrad API",
        "version": "1.0.0",
        "audit_entries": len(_audit_log.entries),
        "chain_verified": _audit_log.verify_chain(),
        "textgrad_ready": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Audit"])
async def analyze(req: AnalyzeRequest):
    """
    Add a processed frame's metrics to the audit log.
    Returns the textual feedback (TextGrad loss signal) and hash chain status.
    """
    entry = AuditEntry(
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        frame_id=req.frame_id,
        faces_detected=req.faces_detected,
        plates_detected=0,
        blur_kernel=req.blur_kernel,
        privacy_score=req.privacy_score,
        utility_score=req.utility_score,
        gdpr_compliant=True,
        detection_model="haar_cascade",
        issues=req.issues,
    )
    _audit_log.add_entry(entry)

    return AnalyzeResponse(
        textual_feedback=_audit_log.generate_textual_feedback(),
        chain_verified=_audit_log.verify_chain(),
        entry_hash=entry.entry_hash,
        total_entries=len(_audit_log.entries),
    )


@app.post("/optimize", response_model=OptimizeResponse, tags=["TextGrad"])
async def optimize(req: OptimizeRequest):
    """
    Run one TextGrad optimization step using the provided audit feedback.
    Requires ANTHROPIC_API_KEY to run the real loop; returns a mock otherwise.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        return OptimizeResponse(
            updated_config=_mock_optimized_config(req.audit_feedback),
            optimization_ran=False,
            backend="none",
            message=(
                "ANTHROPIC_API_KEY not set — returning structured mock response. "
                "Set the key and call again to run the real TextGrad loop."
            ),
            iterations=0,
        )

    try:
        import textgrad as tg

        tg.set_backward_engine("claude-haiku-4-5-20251001", override=True)
        engine = tg.get_engine("claude-haiku-4-5-20251001")

        pipeline_config = tg.Variable(
            value=INITIAL_PIPELINE_CONFIG,
            requires_grad=True,
            role_description=(
                "Privacy-preserving video analytics pipeline configuration. "
                "Optimize for privacy >= 0.90 and utility >= 0.75."
            ),
        )
        oracle_eval = _oracle.evaluate_config(INITIAL_PIPELINE_CONFIG, req.audit_feedback)
        loss_desc = _oracle.build_loss_description(req.audit_feedback, oracle_eval)

        loss_fn = tg.TextLoss(eval_system_prompt=loss_desc, engine=engine)
        loss = loss_fn(pipeline_config)

        optimizer = tg.TGD(parameters=[pipeline_config])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return OptimizeResponse(
            updated_config=pipeline_config.value,
            optimization_ran=True,
            backend="claude-haiku-4-5-20251001",
            message="TextGrad optimization completed successfully (1 iteration).",
            iterations=1,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TextGrad error: {exc}") from exc


# ---------------------------------------------------------------------------
# Mock helper
# ---------------------------------------------------------------------------

def _mock_optimized_config(feedback: str) -> str:
    return f"""
Privacy-Preserving Video Analytics Pipeline Configuration (TextGrad Mock Update)
==================================================================================
[MOCK — set ANTHROPIC_API_KEY to run the real optimization loop]

Based on audit feedback:
{feedback[:300]}...

Suggested changes (what TextGrad would produce):
  Detection Model:    Upgrade Haar Cascade -> RetinaFace (catches occluded faces)
  Blur kernel:        Increase 31 -> 45 to raise privacy score above 0.90
  Conf threshold:     Lower 0.60 -> 0.45 to catch partial faces at frame edges
  GDPR retention:     Add explicit "retention_period: 30 days" to Art.30 record
  Data subject cat:   Add "data_subject_category: public_space_individual"

Expected outcome:
  Privacy score:  0.74 -> ~0.92 (+0.18)
  Utility score:  0.78 -> ~0.76 (-0.02, acceptable)
  PxU product:    0.58 -> ~0.70 (+0.12)
""".strip()
