"""Emotion detection via the HuggingFace Inference API."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{EMOTION_MODEL_NAME}"
SUPPORTED_EMOTIONS = {
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "neutral",
}


def _extract_emotions(result: Any) -> List[Dict[str, Any]]:
    """Extract the list of emotion predictions from a HF Inference API response."""

    try:
        # Common shape: [[{"label": "joy", "score": 0.8}, ...]]
        if isinstance(result, list) and result:
            if isinstance(result[0], list):
                emotions = result[0]
            else:
                emotions = result
        else:
            emotions = []

        if not isinstance(emotions, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in emotions:
            if isinstance(item, dict) and "label" in item and "score" in item:
                normalized.append(item)
        return normalized
    except Exception:
        return []


def _call_hf_inference(text: str, token: str, *, retry_on_loading: bool) -> Any:
    """Call the HF Inference API and optionally retry once if the model is loading."""

    try:
        response = requests.post(
            HF_INFERENCE_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text},
            timeout=20,
        )

        payload: Any
        try:
            payload = response.json()
        except Exception:
            payload = None

        # When the model is spinning up, API often returns a dict with estimated_time.
        if isinstance(payload, dict) and "estimated_time" in payload and retry_on_loading:
            try:
                import time

                wait_seconds = float(payload.get("estimated_time") or 0.0)
                time.sleep(max(0.0, min(wait_seconds, 12.0)))
            except Exception:
                pass
            return _call_hf_inference(text, token, retry_on_loading=False)

        if response.status_code != 200:
            return payload

        return payload
    except Exception:
        return None


def detect_emotion(text: str) -> Dict[str, float | str]:
    """Detect the dominant emotion label for text using the HuggingFace Inference API."""

    if not isinstance(text, str) or not text.strip():
        return {"emotion": "neutral", "confidence": 0.0}

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return {"emotion": "neutral", "confidence": 0.0}

    try:
        result = _call_hf_inference(text, token, retry_on_loading=True)

        if isinstance(result, dict) and result.get("error"):
            return {"emotion": "neutral", "confidence": 0.0}

        emotions = _extract_emotions(result)
        if not emotions:
            return {"emotion": "neutral", "confidence": 0.0}

        top = max(emotions, key=lambda x: float(x.get("score", 0.0)))
        label = str(top.get("label", "neutral")).lower().strip()
        confidence = float(top.get("score", 0.0))

        if label not in SUPPORTED_EMOTIONS:
            label = "neutral"

        return {"emotion": label, "confidence": round(confidence, 4)}
    except Exception:
        return {"emotion": "neutral", "confidence": 0.0}
