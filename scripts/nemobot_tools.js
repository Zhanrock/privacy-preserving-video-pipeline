/**
 * NemoBot LLM Function Definitions — PrivGrad TextGrad API
 * =========================================================
 * How to use:
 *   1. Make sure api.py is running: cd scripts/ && uvicorn api:app --port 8000
 *   2. Open NemoBot at https://nemobot-neue-experiment.vercel.app
 *   3. Go to "Functions" or "Tools" panel
 *   4. Paste each function definition below as a new LLM Function
 *   5. The NemoBot agent will call these automatically when it processes video frames
 *
 * What these do:
 *   analyzeFrame     -- sends frame metrics to the audit log, gets back the
 *                       textual feedback that becomes the TextGrad loss signal
 *   textgradOptimize -- sends audit feedback to the optimizer; if ANTHROPIC_API_KEY
 *                       is set on the server it runs a real TextGrad step,
 *                       otherwise returns a structured mock showing what would change
 */

// ---------------------------------------------------------------------------
// Function 1: analyzeFrame
// ---------------------------------------------------------------------------
// Purpose: After the pipeline processes a video frame, call this to log the
// privacy/utility metrics and get back the GDPR audit feedback.
// The returned textual_feedback is what you pass to textgradOptimize.
// ---------------------------------------------------------------------------
async function analyzeFrame(frame_id, faces_detected, privacy_score, utility_score) {
  const response = await fetch("http://localhost:8000/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frame_id: frame_id,
      faces_detected: faces_detected,
      privacy_score: privacy_score,
      utility_score: utility_score,
      blur_kernel: 31,
      issues: [],
    }),
  });

  if (!response.ok) {
    throw new Error(`analyzeFrame failed: ${response.status} ${response.statusText}`);
  }

  return await response.json();
  // Returns: { textual_feedback, chain_verified, entry_hash, total_entries }
}

// ---------------------------------------------------------------------------
// Function 2: textgradOptimize
// ---------------------------------------------------------------------------
// Purpose: Feed the audit log feedback into the TextGrad optimization loop.
// If ANTHROPIC_API_KEY is set on the server, this runs a real optimization
// step and returns an improved pipeline config.
// If not set, it returns a structured mock showing exactly what would change.
// ---------------------------------------------------------------------------
async function textgradOptimize(audit_feedback) {
  const response = await fetch("http://localhost:8000/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audit_feedback: audit_feedback,
    }),
  });

  if (!response.ok) {
    throw new Error(`textgradOptimize failed: ${response.status} ${response.statusText}`);
  }

  return await response.json();
  // Returns: { updated_config, optimization_ran, backend, message, iterations }
}

// ---------------------------------------------------------------------------
// Example NemoBot workflow (paste this in the agent system prompt or notes):
// ---------------------------------------------------------------------------
//
// 1. For each frame processed by the pipeline:
//      const audit = await analyzeFrame("frame_0001", 3, 0.74, 0.82);
//      console.log("Chain verified:", audit.chain_verified);
//      console.log("Feedback:", audit.textual_feedback);
//
// 2. When you have enough frames logged (e.g. every 10 frames):
//      const result = await textgradOptimize(audit.textual_feedback);
//      if (result.optimization_ran) {
//        console.log("Updated config:", result.updated_config);
//      } else {
//        console.log("Mock mode:", result.message);
//      }
//
// Targets: privacy_score >= 0.90, utility_score >= 0.75
// ---------------------------------------------------------------------------
