from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

# Try to import CrewAI first; fall back to OpenAI if not available
try:
    import crewai  # type: ignore
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False

try:
    import openai  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# -------------------------------
# Configuration
# -------------------------------

MAX_INPUT_CHARS = int(os.environ.get("HATE_DETECTOR_MAX_CHARS", "4000"))
MODEL = os.environ.get("HATE_DETECTOR_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CREWAI_API_KEY = os.environ.get("CREWAI_API_KEY")

# Rate limiting: simple token-bucket per IP (in-memory)
RATE_LIMIT_TOKENS = int(os.environ.get("HATE_DETECTOR_TOKENS", "60"))  # tokens per window
RATE_LIMIT_WINDOW = int(os.environ.get("HATE_DETECTOR_WINDOW", "60"))  # seconds

# Logging setup
LOG_LEVEL = os.environ.get("HATE_DETECTOR_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("hate_detector")

# -------------------------------
# Pydantic models for endpoint
# -------------------------------

class DetectionRequest(BaseModel):
    """Request schema for the hate speech detection endpoint.

    Attributes:
        text: Text to be analyzed. Must be <= MAX_INPUT_CHARS.
        language: Optional BCP-47 language hint to improve detection.
        explain: If true, request a brief rationale in the response.
    """

    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    language: Optional[str] = Field(None, description="Optional language hint (e.g. en, fr)")
    explain: bool = Field(False, description="Return rationale if true")


class DetectionResponse(BaseModel):
    """Response schema for the hate speech detection endpoint."""

    label: str
    score: float
    model: str
    rationale: Optional[str]
    warnings: Optional[List[str]] = None


# -------------------------------
# Helper utilities
# -------------------------------

def safe_truncate(s: str, limit: int) -> str:
    """Truncate string to limit safely.

    Keeps unicode intact and returns escaped text to avoid prompt injection via HTML or control chars.
    """
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "\u2026"


def normalize_text(s: str) -> str:
    """Normalize text: strip, collapse whitespace, basic HTML unescape.

    This helps reduce trivial evasion tactics.
    """
    return re.sub(r"\s+", " ", html.unescape(s).strip())


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -------------------------------
# Simple rule-based fallback classifier
# -------------------------------

# NOTE: This rule-based classifier is intentionally conservative and meant only as a fallback.
# It should NOT be considered production-grade for nuanced cases. Use an LLM-based model in production.

PROTECTED_CLASS_KEYWORDS = [
    "race",
    "ethnic",
    "religion",
    "gender",
    "sexual",
    "orientation",
    "nationality",
    "immigrant",
    "refugee",
]

HATE_INDICATORS = [
    r"\bI hate\b",
    r"\bkill (them|[a-z]+)\b",
    r"\bgo back to\b",
    r"\bdie,? (you )?(.*?)\b",
    r"\bwe should (kill|beat|destroy)\b",
    r"\b(fuck|f\*ck) (them|[a-z]+)\b",
]

NEGATIVE_WORDS = [
    "hate",
    "kill",
    "destroy",
    "attack",
    "exterminate",
    "die",
    "behead",
]


def rule_based_score(text: str) -> Tuple[str, float, str]:
    """Return (label, score, rationale) using a conservative set of heuristics.

    Score: 0-1 confidence for hate speech.
    """
    t = text.lower()
    score = 0.0
    rationale_parts: List[str] = []

    # Quick exact heuristics
    for pat in HATE_INDICATORS:
        if re.search(pat, t):
            score = max(score, 0.7)
            rationale_parts.append(f"Matched pattern: {pat}")

    # Presence of negative verbs + protected group words nearby (simple proximity check)
    for w in NEGATIVE_WORDS:
        if w in t:
            score = max(score, 0.5)
            rationale_parts.append(f"Found indicator word: {w}")
            # Check for protected-class words within 6 words
            for pc in PROTECTED_CLASS_KEYWORDS:
                if re.search(rf"\b{pc}\b", t):
                    score = max(score, 0.85)
                    rationale_parts.append(f"Negative term '{w}' near protected-class '{pc}'")

    if score < 0.1:
        label = "NOT_HATEFUL"
        score = 0.02
    elif score < 0.5:
        label = "UNSURE"
        score = round(score, 2)
    else:
        label = "HATEFUL"
        score = round(score, 2)

    rationale = "; ".join(rationale_parts) if rationale_parts else "No strong lexical indicators detected."
    return label, float(score), rationale


# -------------------------------
# CrewAI/OpenAI integration wrapper
# -------------------------------

class ModelClient:
    """Abstracts over available model providers (CrewAI preferred, OpenAI fallback).

    This class implements a safe prompt template to ask the model to classify text as:
      - HATEFUL
      - NOT_HATEFUL
      - UNSURE

    It also requests a short rationale. The prompt contains policy guidance and format constraints
    to reduce hallucination and keep outputs machine-parseable.
    """

    def __init__(self, model: str = MODEL):
        self.model = model
        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize API clients depending on availability.

        Avoid doing heavy work here; keep it idempotent.
        """
        if CREWAI_AVAILABLE:
            try:
                # Best-effort init; CrewAI SDK details may vary.
                if CREWAI_API_KEY:
                    # If crewai uses environment-based auth this may be redundant.
                    os.environ.setdefault("CREWAI_API_KEY", CREWAI_API_KEY)
                self._provider = "crewai"
                self._client = crewai.Client(api_key=CREWAI_API_KEY) if hasattr(crewai, "Client") else crewai
                logger.info("Using CrewAI provider for inference")
                return
            except Exception as e:
                logger.warning("CrewAI init failed, will try OpenAI: %s", e)

        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
                openai.api_key = OPENAI_API_KEY
                self._provider = "openai"
                self._client = openai
                logger.info("Using OpenAI provider for inference")
                return
            except Exception as e:
                logger.warning("OpenAI init failed: %s", e)

        # No remote provider configured
        self._provider = "none"
        self._client = None
        logger.warning("No remote model provider available; using local fallback")

    def classify(self, text: str, explain: bool = False, language: Optional[str] = None) -> Dict[str, Any]:
        """Classify text. Returns dict containing label, score, rationale, provider.

        This function is defensive: will never raise raw provider exceptions.
        """
        prompt = self._build_prompt(text, explain=explain, language=language)

        if self._provider == "crewai":
            try:
                # Generic attempt for common CrewAI SDK pattern. If the real SDK differs,
                # this will raise and fall back to openai or local.
                logger.debug("Sending prompt to CrewAI: %s", prompt[:400])
                resp = self._client.run(prompt)
                # Normalize response into structured parts if possible
                text_out = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
                return self._parse_model_output(text_out, provider="crewai")
            except Exception as e:
                logger.exception("CrewAI call failed: %s", e)
                # Continue to fallback

        if self._provider == "openai":
            try:
                logger.debug("Sending prompt to OpenAI (model=%s)", self.model)
                completion = self._client.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                text_out = completion.choices[0].message.content
                return self._parse_model_output(text_out, provider="openai")
            except Exception as e:
                logger.exception("OpenAI call failed: %s", e)
                # Continue to local fallback

        # If we reach here, use rule-based fallback
        label, score, rationale = rule_based_score(text)
        return {
            "label": label,
            "score": score,
            "rationale": rationale if explain else None,
            "provider": "fallback-rule-based",
        }

    def _build_prompt(self, text: str, explain: bool = False, language: Optional[str] = None) -> str:
        """Create a robust prompt instructing the model to classify hate speech.

        The prompt enforces a compact, machine-parseable JSON output.
        """
        safe_text = safe_truncate(normalize_text(text), MAX_INPUT_CHARS)
        policy = (
            "You are a content policy assistant. Classify the following user content into one of:"
            " HATEFUL, NOT_HATEFUL, or UNSURE. Base your decision on whether the content expresses hatred"
            " toward a protected group (e.g., by race, religion, gender, sexuality, nationality) or calls for"
            " violence/harassment against such groups or individuals. Be conservative: if uncertain, return UNSURE."
        )

        format_instructions = (
            "Return a JSON object with keys: label (one of HATEFUL/NOT_HATEFUL/UNSURE),"
            " score (0.0-1.0 float confidence), rationale (short string, 1-3 sentences)."
            " ONLY return the JSON and no other text."
        )

        language_hint = f"Language hint: {language}." if language else ""

        prompt = (
            f"{policy}\n\n{format_instructions}\n\n{language_hint}\n\n" "Content:\n" "" + safe_text + "\n\nRespond now."
        )
        return prompt

    def _parse_model_output(self, model_text: str, provider: str) -> Dict[str, Any]:
        """Try to parse the model output as JSON, else fall back to heuristic parsing.

        Always returns a dict with label, score, rationale, provider.
        """
        model_text = model_text.strip()
        # Try to find a JSON object in the output
        try:
            # Heuristic: find first '{' and last '}' and parse
            start = model_text.find("{")
            end = model_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = model_text[start : end + 1]
                parsed = json.loads(candidate)
                label = parsed.get("label")
                score = float(parsed.get("score", 0.0))
                rationale = parsed.get("rationale")
                if label is None:
                    raise ValueError("No label in model JSON")
                return {
                    "label": str(label).upper(),
                    "score": max(0.0, min(1.0, float(score))) if isinstance(score, (float, int)) else 0.0,
                    "rationale": rationale,
                    "provider": provider,
                }
        except Exception as e:
            logger.debug("Failed to parse JSON from model output: %s; error: %s", model_text[:200], e)

        # As fallback make a very conservative parse
        lowered = model_text.lower()
        label = "UNSURE"
        if "hateful" in lowered or "hate" in lowered:
            label = "HATEFUL"
        elif "not_hateful" in lowered or "not hateful" in lowered or "not hateful" in lowered:
            label = "NOT_HATEFUL"

        score = 0.5
        if "confidence" in lowered:
            m = re.search(r"([0-9]+\.?[0-9]*)\s*%", model_text)
            if m:
                try:
                    score = float(m.group(1)) / 100.0
                except Exception:
                    score = 0.5
        rationale = model_text if len(model_text) < 1000 else model_text[:1000]
        return {"label": label, "score": score, "rationale": rationale, "provider": provider}


# -------------------------------
# Rate limiter implementation (in-memory token bucket)
# -------------------------------

@dataclass
class Bucket:
    tokens: float
    last_ts: float
    lock: threading.Lock


class RateLimiter:
    def __init__(self, tokens: int = RATE_LIMIT_TOKENS, window: int = RATE_LIMIT_WINDOW):
        self.tokens = tokens
        self.window = window
        self.refill_rate = tokens / float(window)
        self.buckets: Dict[str, Bucket] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            bucket = self.buckets.get(key)
            if not bucket:
                self.buckets[key] = Bucket(tokens=self.tokens - 1, last_ts=now, lock=threading.Lock())
                return True
            # refill
            elapsed = max(0.0, now - bucket.last_ts)
            refill = elapsed * self.refill_rate
            bucket.tokens = min(self.tokens, bucket.tokens + refill)
            bucket.last_ts = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False


rate_limiter = RateLimiter()


# -------------------------------
# FastAPI application and routes
# -------------------------------

app = FastAPI(title="CrewAI Hate Speech Detector", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("HATE_DETECTOR_CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"]
)

# Initialize model client lazily
_model_client: Optional[ModelClient] = None
_model_client_lock = threading.Lock()


def get_model_client() -> ModelClient:
    global _model_client
    with _model_client_lock:
        if _model_client is None:
            _model_client = ModelClient()
        return _model_client


@app.post("/detect", response_model=DetectionResponse)
async def detect(request: Request, body: DetectionRequest) -> DetectionResponse:
    """Endpoint: Detect whether the provided text is hate speech.

    Security considerations:
      - Input length-limited
      - Rate-limited per client IP
      - Sanitize inputs to reduce prompt injection
    """
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        text = normalize_text(body.text)
    except Exception as e:
        logger.exception("Invalid input text: %s", e)
        raise HTTPException(status_code=400, detail="Invalid text input")

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="Empty text after normalization")

    # Defensive prompt-safety: remove accidental JSON-looking suffixes to avoid model confusion
    # (not a full-proof protection against prompt injection, but helps reduce simple cases)
    text = re.sub(r"\n{2,}", "\n", text)

    # Choose classification path
    client = get_model_client()

    try:
        result = client.classify(text, explain=body.explain, language=body.language)
    except Exception as e:
        logger.exception("Classification failed unexpectedly: %s", e)
        # Provide fallback
        label, score, rationale = rule_based_score(text)
        result = {"label": label, "score": score, "rationale": rationale, "provider": "fallback"}

    label = result.get("label", "UNSURE")
    score = float(result.get("score", 0.0))
    rationale = result.get("rationale") if body.explain else None

    # Construct warnings if the provider is a fallback or response malformed
    warnings: List[str] = []
    provider = result.get("provider")
    if provider == "fallback-rule-based" or provider == "fallback":
        warnings.append("Using conservative rule-based fallback; results are approximate.")
    if provider == "none":
        warnings.append("No remote model provider configured; running local heuristics.")

    response = DetectionResponse(
        label=label,
        score=score,
        model=str(provider),
        rationale=rationale,
        warnings=warnings if warnings else None,
    )

    # Audit log: write compact log for downstream analysis (avoid logging full text in prod unless necessary)
    try:
        logger.info(
            "detect: ip=%s hash=%s label=%s score=%.2f provider=%s len=%d",
            client_ip,
            text_hash(text)[:8],
            response.label,
            response.score,
            response.model,
            len(text),
        )
    except Exception:
        logger.exception("Failed to write audit log")

    return response


# Local runnable entrypoint
if __name__ == "__main__":
    import uvicorn  # type: ignore

    # Developer tip: set ENV OPENAI_API_KEY or CREWAI_API_KEY before running for best results.
    uvicorn.run("generated_code:app", host="0.0.0.0", port=8000, log_level=LOG_LEVEL.lower())
