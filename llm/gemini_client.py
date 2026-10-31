import json
import logging
import os
import re

logger = logging.getLogger("GeminiLayer")


class GeminiClient:
    """Optional advisory veto (plan.md principle 4: the LLM is advisory, never authoritative).
    It can turn a trade away ("AVOID"); it can never create, size or approve one. Any error
    means no veto."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"')
        self.enabled = False
        if not api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini Layer will be disabled.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip().strip('"'))
            self.enabled = True
        except Exception as e:
            logger.warning(f"Gemini unavailable: {e}")

    def evaluate_trade_candidate(self, candidate_context: dict) -> dict:
        if not self.enabled:
            return {"recommendation": "ALLOW", "reason_codes": ["GEMINI_DISABLED"]}

        prompt = f"""
        You are a risk reviewer for an intraday NSE equity trade that already passed deterministic
        risk checks. Look only for context that makes it unusually dangerous today (results day,
        regulatory action, fraud allegations, a halt, a block deal). Respond in valid JSON ONLY.
        No prose or markdown fences.

        Context:
        {json.dumps(candidate_context, indent=2, default=str)}

        Expected JSON format:
        {{
            "context_risk": float, // 0.0 to 1.0
            "anomaly": boolean,
            "recommendation": "ALLOW" | "AVOID",
            "reason_codes": [string]
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            text = re.sub(r"^```(json)?|```$", "", response.text.strip(), flags=re.M).strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"recommendation": "ALLOW", "reason_codes": ["API_ERROR"]}
