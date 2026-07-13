/**
 * NemoBot LLM Function Definitions — PrivGrad TextGrad API
 * =========================================================
 * NTU CCDS | Assoc Prof Chee Wei Tan | Krittaphas Thanaphongphaisan
 *
 * HOW TO USE:
 *   1. Run local backend:  cd scripts && uvicorn api:app --port 8000
 *   2. In NemoBot -> "Create New LLM Function" -- make TWO functions below.
 *      NemoBot runs postprocess JS in the browser, so localhost:8000 works
 *      directly -- no ngrok or public URL required.
 *
 * USING ANTHROPIC IN NEMOBOT (OpenAI-compatible endpoint):
 *   LLM_CHAT_COMPLETION_API_TYPE  : OpenAI        <- keep as-is
 *   LLM_CHAT_COMPLETION_API_URL   : https://api.anthropic.com/v1
 *   LLM_CHAT_COMPLETION_API_KEY   : <your Anthropic API key>
 *   LLM_CHAT_COMPLETION_MODEL     : claude-haiku-4-5-20251001
 *
 *   Anthropic's /v1/chat/completions endpoint is OpenAI-compatible,
 *   so NemoBot's OpenAI client works against it directly.
 *
 * ============================================================
 * FUNCTION 1: analyzePrivacyFrame
 * ============================================================
 *
 * BASIC INFORMATION TAB
 *   Name:        async function analyzePrivacyFrame
 *   Description: Call this tool to log a processed video frame's privacy and
 *                utility metrics into the GDPR Art.30 audit trail. Returns
 *                the textual feedback that becomes the TextGrad loss signal.
 *                Call it after each frame or when the user provides frame stats.
 *
 *   Input Format (JSON Schema):
 *     {
 *       "type": "object",
 *       "properties": {
 *         "frame_id":       { "type": "string" },
 *         "faces_detected": { "type": "number" },
 *         "privacy_score":  { "type": "number" },
 *         "utility_score":  { "type": "number" }
 *       },
 *       "required": ["frame_id", "faces_detected", "privacy_score", "utility_score"]
 *     }
 *
 *   Output Format (JSON Schema):
 *     {
 *       "type": "object",
 *       "properties": {
 *         "textual_feedback": { "type": "string" },
 *         "chain_verified":   { "type": "boolean" },
 *         "entry_hash":       { "type": "string" },
 *         "total_entries":    { "type": "number" }
 *       }
 *     }
 *
 *   Environment Variables to add:
 *     Key: PRIVGRAD_API_URL
 *     Value: http://localhost:8000  (NemoBot runs JS in the browser, so
 *            localhost points to your machine -- no ngrok needed)
 *
 * ---- PRE PROCESSING TAB (paste the function body below) ----
 */

async function preprocess(input, environment) {
  // Pass structured input straight through to the LLM
  return { input };
}

/**
 * ---- POST PROCESSING TAB for analyzePrivacyFrame (paste the function body below) ----
 */

async function postprocess(llmOutput, environment) {
  // Step 1: Extract structured metrics from LLM output
  let outputStr = typeof llmOutput === "string" ? llmOutput : JSON.stringify(llmOutput);
  let metrics = { frame_id: "frame_unknown", faces_detected: 0, privacy_score: 0.0, utility_score: 0.0 };

  try {
    const match = outputStr.match(/\{[\s\S]*?\}/);
    if (match) {
      const parsed = JSON.parse(match[0]);
      metrics = { ...metrics, ...parsed };
    }
  } catch (err) {
    console.error("Metrics parse failed:", err);
  }

  // Step 2: Call PrivGrad audit API
  const apiUrl = environment.PRIVGRAD_API_URL || "http://localhost:8000";
  try {
    const response = await fetch(`${apiUrl}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_id:       String(metrics.frame_id),
        faces_detected: Number(metrics.faces_detected) || 0,
        privacy_score:  Number(metrics.privacy_score)  || 0.0,
        utility_score:  Number(metrics.utility_score)  || 0.0,
        blur_kernel:    31,
        issues:         [],
      }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    return {
      textual_feedback: data.textual_feedback,
      chain_verified:   data.chain_verified,
      entry_hash:       data.entry_hash,
      total_entries:    data.total_entries,
    };
  } catch (err) {
    console.error("analyzePrivacyFrame API call failed:", err);
    return {
      textual_feedback: `[API unavailable] Frame ${metrics.frame_id}: privacy=${metrics.privacy_score}, utility=${metrics.utility_score}`,
      chain_verified:   false,
      entry_hash:       "unavailable",
      total_entries:    0,
    };
  }
}

/**
 * ============================================================
 * FUNCTION 2: textgradOptimize
 * ============================================================
 *
 * BASIC INFORMATION TAB
 *   Name:        async function textgradOptimize
 *   Description: Call this tool to run the TextGrad optimization loop using
 *                GDPR audit feedback as the loss signal. Returns an improved
 *                pipeline configuration. Call it after analyzePrivacyFrame
 *                has logged enough frames, or when the user asks to optimize.
 *
 *   Input Format (JSON Schema):
 *     {
 *       "type": "object",
 *       "properties": {
 *         "audit_feedback": { "type": "string" }
 *       },
 *       "required": ["audit_feedback"]
 *     }
 *
 *   Output Format (JSON Schema):
 *     {
 *       "type": "object",
 *       "properties": {
 *         "updated_config":    { "type": "string" },
 *         "optimization_ran":  { "type": "boolean" },
 *         "backend":           { "type": "string" },
 *         "message":           { "type": "string" },
 *         "iterations":        { "type": "number" }
 *       }
 *     }
 *
 *   Environment Variables to add:
 *     Key: PRIVGRAD_API_URL
 *     Value: http://localhost:8000
 *
 * ---- PRE PROCESSING TAB (paste the function body below) ----
 */

async function preprocess(input, environment) {
  return { input };
}

/**
 * ---- POST PROCESSING TAB for textgradOptimize (paste the function body below) ----
 */

async function postprocess(llmOutput, environment) {
  // Extract audit_feedback string from LLM output
  let outputStr = typeof llmOutput === "string" ? llmOutput : JSON.stringify(llmOutput);
  let auditFeedback = outputStr;

  try {
    const match = outputStr.match(/\{[\s\S]*?\}/);
    if (match) {
      const parsed = JSON.parse(match[0]);
      auditFeedback = parsed.audit_feedback || outputStr;
    }
  } catch (err) {
    // Use raw string as feedback
  }

  const apiUrl = environment.PRIVGRAD_API_URL || "http://localhost:8000";
  try {
    const response = await fetch(`${apiUrl}/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audit_feedback: auditFeedback }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    return {
      updated_config:   data.updated_config,
      optimization_ran: data.optimization_ran,
      backend:          data.backend,
      message:          data.message,
      iterations:       data.iterations,
    };
  } catch (err) {
    console.error("textgradOptimize API call failed:", err);
    return {
      updated_config:   "[API unavailable -- is the backend running with ngrok?]",
      optimization_ran: false,
      backend:          "none",
      message:          String(err),
      iterations:       0,
    };
  }
}

/**
 * ============================================================
 * SYSTEM PROMPT -- paste into NemoBot chat settings
 * ============================================================
 *
 * You are PrivGrad, an AI assistant for privacy-preserving video analytics
 * at NTU CCDS. You help users log video frame metrics, run GDPR Art.30
 * audit checks, and optimize the pipeline using TextGrad.
 *
 * When a user provides frame statistics (faces detected, privacy score,
 * utility score), call analyzePrivacyFrame to log them.
 *
 * When a user asks to optimize, or after 5+ frames are logged, call
 * textgradOptimize with the most recent textual_feedback.
 *
 * Targets: privacy_score >= 0.90, utility_score >= 0.75.
 * ============================================================
 */
