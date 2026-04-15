"""Recommendation route logic for the Mood2Food API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.emotion_model import detect_emotion
from utils.nutrition_fetch import get_nutrition

router = APIRouter(tags=["Recommendations"])
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "food_mood_map.csv"
EMOTION_TAGS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
TIME_ALIASES = {
    "morning": {"breakfast"},
    "breakfast": {"breakfast"},
    "afternoon": {"lunch"},
    "lunch": {"lunch"},
    "evening": {"snack", "dinner"},
    "snack": {"snack"},
    "night": {"dinner"},
    "dinner": {"dinner"},
    "late night": {"dinner", "snack"},
    "any": {"breakfast", "lunch", "dinner", "snack"},
}


class RecommendRequest(BaseModel):
    """Input payload for the food recommendation endpoint."""

    mood_text: str = Field(..., description="User's natural language mood description.")
    time_of_day: str = Field(..., description="Current time of day or meal window.")
    diet_type: str = Field(..., description="Diet preference such as veg, nonveg, or vegan.")


def _load_food_dataframe() -> pd.DataFrame:
    """Load the food mapping dataset from disk."""

    try:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")
        dataframe = pd.read_csv(DATA_FILE)
        required_columns = {
            "food_name",
            "cuisine",
            "diet_type",
            "mood_tags",
            "nutrients",
            "meal_time",
            "mood_score_weight",
        }
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
        return dataframe.fillna("")
    except Exception as exc:
        raise RuntimeError(f"Failed to load food dataset: {exc}") from exc


@lru_cache(maxsize=1)
def _get_food_index() -> Dict[str, Any]:
    """Build and cache the emotion-tag vectorizer and dataset for matching."""

    dataframe = _load_food_dataframe()
    vectorizer = TfidfVectorizer(
        vocabulary=EMOTION_TAGS,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        binary=True,
        use_idf=False,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(dataframe["mood_tags"].astype(str).str.replace(";", " ", regex=False))
    return {"dataframe": dataframe, "vectorizer": vectorizer, "matrix": matrix}


def _normalize_text(value: str) -> str:
    """Normalize free text to a lowercase trimmed string."""

    try:
        return str(value).strip().lower()
    except Exception:
        return ""


def _diet_matches(food_diet: str, user_diet: str) -> bool:
    """Check whether a food item matches the requested diet preference."""

    normalized_food_diet = _normalize_text(food_diet)
    normalized_user_diet = _normalize_text(user_diet)
    if normalized_user_diet in {"", "any", "all"}:
        return True
    return normalized_food_diet == normalized_user_diet


def _meal_matches(food_meal_time: str, user_time_of_day: str) -> bool:
    """Check whether a food item matches the requested time of day."""

    normalized_food_meal = _normalize_text(food_meal_time)
    normalized_user_time = _normalize_text(user_time_of_day)
    if normalized_user_time in {"", "any", "all"}:
        return True
    allowed_meals = TIME_ALIASES.get(normalized_user_time, {normalized_user_time})
    return normalized_food_meal in allowed_meals


def _select_best_food(
    dataframe: pd.DataFrame,
    matrix,
    vectorizer: TfidfVectorizer,
    detected_emotion: str,
    diet_type: str,
    time_of_day: str,
) -> Dict[str, Any]:
    """Select the best matching food row from the dataset."""

    try:
        if matrix is None or getattr(matrix, "shape", (0,))[0] != len(dataframe):
            matrix = vectorizer.transform(dataframe["mood_tags"].astype(str).str.replace(";", " ", regex=False))

        query_vector = vectorizer.transform([detected_emotion])
        similarity_scores = cosine_similarity(query_vector, matrix).flatten()

        if similarity_scores.size == 0:
            raise ValueError("No similarity scores could be computed.")

        score_df = dataframe.copy().reset_index(drop=True)
        score_df["score"] = [float(score) for score in similarity_scores]
        score_df = score_df.sort_values(by="score", ascending=False, kind="stable").reset_index(drop=True)

        top_debug_rows = score_df.head(3)
        print(f"[Mood2Food] Top 3 scores for emotion='{detected_emotion}':")
        for row in top_debug_rows.itertuples(index=False):
            print(
                "[Mood2Food] "
                f"food={row.food_name} score={round(float(row.score), 4)} "
                f"diet={row.diet_type} meal={row.meal_time} moods={row.mood_tags}"
            )

        filtered_scored_df = score_df[
            score_df["diet_type"].astype(str).apply(lambda value: _diet_matches(value, diet_type))
            & score_df["meal_time"].astype(str).apply(lambda value: _meal_matches(value, time_of_day))
        ]
        positive_filtered_scored_df = filtered_scored_df[filtered_scored_df["score"] > 0.0]

        if not positive_filtered_scored_df.empty:
            best_row_series = positive_filtered_scored_df.iloc[0]
        else:
            best_row_series = score_df.iloc[0]

        best_cosine_score = float(best_row_series["score"])
        best_row = best_row_series.drop(labels=["score"]).to_dict()

        return {
            "food": best_row,
            "score": round(max(0.0, min(1.0, best_cosine_score)), 4),
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to score food recommendations: {exc}") from exc


def _build_why_it_fits(food_row: Dict[str, Any], emotion: str, time_of_day: str, nutrition: Dict[str, Any]) -> str:
    """Create a human-readable explanation for the selected food."""

    try:
        food_name = str(food_row.get("food_name", "this dish")).strip()
        time_phrase = str(food_row.get("meal_time", "anytime")).strip()
        cuisine = str(food_row.get("cuisine", "Indian")).strip()
        nutrients = str(food_row.get("nutrients", "")).strip()
        mood_tags = str(food_row.get("mood_tags", "")).strip().replace(";", ", ")
        nutrition_note = ""
        if nutrition.get("available"):
            calories = nutrition.get("calories")
            protein = nutrition.get("protein")
            nutrition_note = f" USDA data indicates about {calories} kcal and {protein} g protein per 100g."
        return (
            f"Detected emotion is '{emotion}', and '{food_name}' is tagged for moods: {mood_tags}. "
            f"It suits the requested {time_of_day} timing via its {time_phrase} meal slot, follows {cuisine} cuisine, "
            f"and its key nutrients are {nutrients}.{nutrition_note}"
        )
    except Exception:
        return "The selected food is matched to the detected emotion and requested meal context."


@router.post("/recommend")
def recommend_food(payload: RecommendRequest) -> Dict[str, Any]:
    """Return a mood-aware food recommendation with nutrition details."""

    try:
        if not payload.mood_text.strip():
            raise HTTPException(status_code=400, detail="mood_text is required.")
        if not payload.time_of_day.strip():
            raise HTTPException(status_code=400, detail="time_of_day is required.")
        if not payload.diet_type.strip():
            raise HTTPException(status_code=400, detail="diet_type is required.")

        emotion_result = detect_emotion(payload.mood_text)
        emotion = str(emotion_result.get("emotion", "neutral")).lower()
        confidence = float(emotion_result.get("confidence", 0.0))

        index_bundle = _get_food_index()
        dataframe = index_bundle["dataframe"]
        vectorizer = index_bundle["vectorizer"]
        matrix = index_bundle["matrix"]

        selection = _select_best_food(
            dataframe=dataframe,
            matrix=matrix,
            vectorizer=vectorizer,
            detected_emotion=emotion,
            diet_type=payload.diet_type,
            time_of_day=payload.time_of_day,
        )
        food_row = selection["food"]
        nutrition_summary = get_nutrition(str(food_row.get("food_name", "")))
        why_it_fits = _build_why_it_fits(food_row, emotion, payload.time_of_day, nutrition_summary)

        return {
            "detected_emotion": {"emotion": emotion, "confidence": confidence},
            "recommended_food": {
                "food_name": food_row.get("food_name"),
                "cuisine": food_row.get("cuisine"),
                "diet_type": food_row.get("diet_type"),
                "meal_time": food_row.get("meal_time"),
                "nutrients": food_row.get("nutrients"),
            },
            "why_it_fits": why_it_fits,
            "mood_match_score": selection["score"],
            "nutrition_summary": nutrition_summary,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc
