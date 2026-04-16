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


def _normalize_inference_response(payload: Any) -> List[Dict[str, Any]]:
    """Normalize HuggingFace Inference API responses into a flat list of predictions."""

    try:
        if isinstance(payload, list) and payload:
            first_item = payload[0]
            if isinstance(first_item, dict):
                return payload
            if isinstance(first_item, list):
                return first_item
    except Exception:
        return []

    return []


def detect_emotion(text: str) -> Dict[str, float | str]:
    """Detect the dominant emotion label for text using the HuggingFace Inference API."""

    if not isinstance(text, str) or not text.strip():
        return {"emotion": "neutral", "confidence": 0.0}

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return {"emotion": "neutral", "confidence": 0.0}

    try:
        response = requests.post(
            HF_INFERENCE_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text},
            timeout=15,
        )

        if response.status_code != 200:
            return {"emotion": "neutral", "confidence": 0.0}

        raw_payload: Any = response.json()

        if isinstance(raw_payload, dict) and raw_payload.get("error"):
            return {"emotion": "neutral", "confidence": 0.0}

        predictions = _normalize_inference_response(raw_payload)
        if not predictions:
            return {"emotion": "neutral", "confidence": 0.0}

        best_prediction = max(predictions, key=lambda item: float(item.get("score", 0.0)))
        label = str(best_prediction.get("label", "neutral")).lower().strip()
        confidence = float(best_prediction.get("score", 0.0))

        if label not in SUPPORTED_EMOTIONS:
            label = "neutral"

        return {"emotion": label, "confidence": round(confidence, 4)}
    except requests.RequestException:
        return {"emotion": "neutral", "confidence": 0.0}
    except Exception:
        return {"emotion": "neutral", "confidence": 0.0}
