"""USDA FoodData Central nutrition lookup helpers."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

load_dotenv()

USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
IFCT_DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "ifct_nutrition.csv"
DEFAULT_NUTRITION = {
    "calories": None,
    "protein": None,
    "carbs": None,
    "fiber": None,
    "iron": None,
    "calcium": None,
    "magnesium": None,
}


def _find_nutrient_amount(food_nutrients: Iterable[Dict[str, Any]], target_names: set[str]) -> float | None:
    """Extract the numeric amount for a nutrient from USDA response data."""

    try:
        for nutrient in food_nutrients:
            nutrient_name = ""
            if isinstance(nutrient, dict):
                if isinstance(nutrient.get("nutrient"), dict):
                    nutrient_name = str(nutrient["nutrient"].get("name", "")).lower()
                if not nutrient_name:
                    nutrient_name = str(nutrient.get("nutrientName", "")).lower()
                if nutrient_name in target_names:
                    amount = nutrient.get("amount")
                    if amount is not None:
                        return float(amount)
    except Exception:
        return None

    return None


def _normalize_food_name(value: str) -> str:
    normalized = str(value or "").lower().strip()
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@lru_cache(maxsize=1)
def _load_ifct_dataframe():
    if not IFCT_DATA_FILE.exists():
        return None

    try:
        import pandas as pd

        return pd.read_csv(IFCT_DATA_FILE)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_ifct_rows_fallback() -> list[Dict[str, str]]:
    """CSV fallback loader when pandas is unavailable."""

    if not IFCT_DATA_FILE.exists():
        return []

    import csv

    with IFCT_DATA_FILE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row and row.get("food_name")]


def get_ifct_nutrition(food_name: str) -> Dict[str, Any] | None:
    """Lookup nutrition facts for a food name in the IFCT 2017 CSV dataset.

    Returns a best-effort fuzzy match against `data/ifct_nutrition.csv`.
    """

    if not isinstance(food_name, str) or not food_name.strip():
        return None

    # Prefer pandas for loading (as requested), but keep a csv fallback.
    df = _load_ifct_dataframe()
    if df is not None and not df.empty and "food_name" in df.columns:
        food_list = [str(x) for x in df["food_name"].fillna("").tolist() if str(x).strip()]
        match = process.extractOne(food_name, food_list, scorer=fuzz.WRatio)
        if not match:
            return None
        matched_name, score, index = match
        if float(score) <= 70:
            return None
        row = df.iloc[int(index)].to_dict()
        best_row = {str(k): ("" if v is None else v) for k, v in row.items()}
        best_score = float(score) / 100.0
    else:
        rows = _load_ifct_rows_fallback()
        if not rows:
            return None

        food_list = [str(r.get("food_name", "")).strip() for r in rows if str(r.get("food_name", "")).strip()]
        match = process.extractOne(food_name, food_list, scorer=fuzz.WRatio)
        if not match:
            return None
        matched_name, score, _ = match
        if float(score) <= 70:
            return None
        best_row = next((r for r in rows if str(r.get("food_name", "")).strip() == matched_name), None)
        if not best_row:
            return None
        best_score = float(score) / 100.0

    def _to_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    return {
        "calories": _to_float(best_row.get("calories_per_100g")),
        "protein": _to_float(best_row.get("protein_g")),
        "carbs": _to_float(best_row.get("carbs_g")),
        "fiber": _to_float(best_row.get("fiber_g")),
        "iron": _to_float(best_row.get("iron_mg")),
        "calcium": _to_float(best_row.get("calcium_mg")),
        "magnesium": _to_float(best_row.get("magnesium_mg")),
        "available": True,
        "source": "ifct",
        "ifct_food_name": str(best_row.get("food_name", food_name)).strip(),
        "match_score": round(float(best_score), 4),
    }


def get_estimated_nutrition(food_name: str) -> Dict[str, Any]:
    """Return a generic per-100g estimate when no source data is available."""

    name = str(food_name or "").strip()
    normalized = _normalize_food_name(name)

    def base(category: str) -> Dict[str, float]:
        if category == "rice":
            return {
                "calories": 130,
                "protein": 3.0,
                "carbs": 28.0,
                "fiber": 0.4,
                "iron": 0.3,
                "calcium": 10.0,
                "magnesium": 12.0,
            }
        if category == "dal":
            return {
                "calories": 100,
                "protein": 9.0,
                "carbs": 18.0,
                "fiber": 7.0,
                "iron": 3.0,
                "calcium": 35.0,
                "magnesium": 35.0,
            }
        if category == "roti":
            return {
                "calories": 120,
                "protein": 4.0,
                "carbs": 25.0,
                "fiber": 3.0,
                "iron": 1.6,
                "calcium": 20.0,
                "magnesium": 25.0,
            }
        if category == "curry_veg":
            return {
                "calories": 80,
                "protein": 3.0,
                "carbs": 12.0,
                "fiber": 2.5,
                "iron": 1.2,
                "calcium": 35.0,
                "magnesium": 20.0,
            }
        if category == "curry_nonveg":
            return {
                "calories": 150,
                "protein": 15.0,
                "carbs": 8.0,
                "fiber": 0.5,
                "iron": 2.0,
                "calcium": 20.0,
                "magnesium": 20.0,
            }
        if category == "biryani":
            return {
                "calories": 200,
                "protein": 8.0,
                "carbs": 35.0,
                "fiber": 1.5,
                "iron": 1.2,
                "calcium": 25.0,
                "magnesium": 22.0,
            }
        if category == "snack":
            return {
                "calories": 250,
                "protein": 5.0,
                "carbs": 30.0,
                "fiber": 3.0,
                "iron": 1.5,
                "calcium": 25.0,
                "magnesium": 20.0,
            }
        if category == "sweet":
            return {
                "calories": 300,
                "protein": 5.0,
                "carbs": 50.0,
                "fiber": 1.0,
                "iron": 0.8,
                "calcium": 120.0,
                "magnesium": 18.0,
            }

        # mixed fallback
        return {
            "calories": 140,
            "protein": 5.0,
            "carbs": 20.0,
            "fiber": 2.0,
            "iron": 1.0,
            "calcium": 30.0,
            "magnesium": 18.0,
        }

    category = "mixed"
    if any(k in normalized for k in ["biryani", "pulao", "pulav"]):
        category = "biryani"
    elif any(k in normalized for k in ["dal", "lentil", "sambar", "rasam", "khichdi"]):
        category = "dal"
    elif any(k in normalized for k in ["roti", "chapati", "paratha", "naan", "kulcha", "bhatura", "poori", "puri"]):
        category = "roti"
    elif "rice" in normalized or any(k in normalized for k in ["puliyogare", "pulihora", "curd rice", "lemon rice", "coconut rice"]):
        category = "rice"
    elif any(k in normalized for k in ["samosa", "pakora", "bhaji", "chaat", "pani puri", "bhel", "momos", "vada pav", "kachori"]):
        category = "snack"
    elif any(k in normalized for k in ["halwa", "kheer", "payasam", "ladoo", "jalebi", "gulab jamun", "rasgulla", "barfi", "kulfi"]):
        category = "sweet"
    elif any(k in normalized for k in ["curry", "masala", "korma", "jhol", "stew", "roast"]):
        category = "curry_nonveg" if any(k in normalized for k in ["chicken", "mutton", "fish", "prawn", "egg", "beef", "pork", "keema"]) else "curry_veg"

    values = base(category)
    return {
        **values,
        "available": True,
        "source": "estimated",
        "category": category,
        "note": "Estimated nutrition values (generic per-100g).",
    }


def _get_usda_nutrition(food_name: str) -> Dict[str, Any]:
    """Fetch nutrition facts for a food name using USDA FoodData Central."""

    if not isinstance(food_name, str) or not food_name.strip():
        return {
            **DEFAULT_NUTRITION,
            "available": False,
            "source": "none",
            "note": "Food name is required.",
        }

    api_key = os.getenv("USDA_API_KEY", "").strip()
    if not api_key or api_key == "your_usda_api_key_here":
        return {
            **DEFAULT_NUTRITION,
            "available": False,
            "source": "none",
            "note": "USDA_API_KEY is not configured.",
        }

    try:
        search_response = requests.post(
            f"{USDA_BASE_URL}/foods/search",
            params={"api_key": api_key},
            json={
                "query": food_name,
                "pageSize": 1,
                "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)"] ,
            },
            timeout=10,
        )
        search_response.raise_for_status()
        search_payload = search_response.json()
        foods = search_payload.get("foods") or []
        if not foods:
            return {
                **DEFAULT_NUTRITION,
                "available": False,
                "source": "usda",
                "note": "No USDA match found for this food.",
            }

        fdc_id = foods[0].get("fdcId")
        if fdc_id is None:
            return {
                **DEFAULT_NUTRITION,
                "available": False,
                "source": "usda",
                "note": "USDA response did not include a food identifier.",
            }

        detail_response = requests.get(
            f"{USDA_BASE_URL}/food/{fdc_id}",
            params={"api_key": api_key},
            timeout=10,
        )
        detail_response.raise_for_status()
        detail_payload = detail_response.json()
        food_nutrients = detail_payload.get("foodNutrients", [])

        calories = _find_nutrient_amount(food_nutrients, {"energy"})
        protein = _find_nutrient_amount(food_nutrients, {"protein"})
        carbs = _find_nutrient_amount(food_nutrients, {"carbohydrate", "carbohydrate, by difference"})
        iron = _find_nutrient_amount(food_nutrients, {"iron, fe"})
        magnesium = _find_nutrient_amount(food_nutrients, {"magnesium, mg"})

        return {
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fiber": None,
            "iron": iron,
            "calcium": None,
            "magnesium": magnesium,
            "available": True,
            "source": "usda",
            "usda_food_name": detail_payload.get("description", food_name),
            "usda_fdc_id": fdc_id,
        }
    except requests.RequestException as exc:
        return {
            **DEFAULT_NUTRITION,
            "available": False,
            "source": "usda",
            "note": f"USDA lookup failed: {exc}",
        }
    except Exception as exc:
        return {
            **DEFAULT_NUTRITION,
            "available": False,
            "source": "usda",
            "note": f"Unexpected nutrition lookup failure: {exc}",
        }


def get_nutrition(food_name: str) -> Dict[str, Any]:
    """Fetch nutrition facts for a food name.

    Strategy: USDA → IFCT → Estimated.
    Always returns a non-null nutrition payload.
    """

    usda_result = _get_usda_nutrition(food_name)
    if usda_result.get("available"):
        return usda_result

    ifct_result = get_ifct_nutrition(food_name)
    if ifct_result and ifct_result.get("available"):
        return ifct_result

    return get_estimated_nutrition(food_name)
