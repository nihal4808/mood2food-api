"""USDA FoodData Central nutrition lookup helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable

import requests
from dotenv import load_dotenv

load_dotenv()

USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DEFAULT_NUTRITION = {
    "calories": None,
    "protein": None,
    "carbs": None,
    "iron": None,
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


def get_nutrition(food_name: str) -> Dict[str, Any]:
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
            "iron": iron,
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
