"""Emotion detection helpers for mapping natural language moods to emotions."""

from functools import lru_cache
from typing import Any, Dict, List

from transformers import pipeline

EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SUPPORTED_EMOTIONS = {
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "neutral",
}


@lru_cache(maxsize=1)
def _load_emotion_pipeline() -> Any:
    """Load and cache the HuggingFace emotion classification pipeline."""

    try:
        return pipeline("text-classification", model=EMOTION_MODEL_NAME, top_k=None)
    except Exception as exc:  # pragma: no cover - runtime dependency/network failure
        raise RuntimeError(f"Failed to load emotion model: {exc}") from exc


def _normalize_predictions(raw_predictions: Any) -> List[Dict[str, Any]]:
    """Normalize HuggingFace pipeline output into a flat list of predictions."""

    try:
        if isinstance(raw_predictions, list) and raw_predictions:
            first_item = raw_predictions[0]
            if isinstance(first_item, dict):
                return raw_predictions
            if isinstance(first_item, list):
                return first_item
    except Exception:
        return []

    return []


def detect_emotion(text: str) -> Dict[str, float | str]:
    """Detect the dominant emotion in text and return label plus confidence."""

    if not isinstance(text, str) or not text.strip():
        return {"emotion": "neutral", "confidence": 0.0}

    try:
        emotion_pipeline = _load_emotion_pipeline()
        raw_predictions = emotion_pipeline(text)
        predictions = _normalize_predictions(raw_predictions)

        if not predictions:
            return {"emotion": "neutral", "confidence": 0.0}

        best_prediction = max(predictions, key=lambda item: float(item.get("score", 0.0)))
        label = str(best_prediction.get("label", "neutral")).lower().strip()
        confidence = float(best_prediction.get("score", 0.0))

        if label not in SUPPORTED_EMOTIONS:
            label = "neutral"

        return {"emotion": label, "confidence": round(confidence, 4)}
    except Exception:
        return {"emotion": "neutral", "confidence": 0.0}
