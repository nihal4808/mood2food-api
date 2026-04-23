"""FastAPI application entrypoint for the Mood2Food API."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Dict

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes.recommend import router as recommend_router

VALID_KEYS: Dict[str, Dict[str, Any]] = {
    "demo-key-123": {"plan": "free", "daily_limit": 100},
    "startup-key-456": {"plan": "startup", "daily_limit": 10000},
}

_usage_lock = asyncio.Lock()
_usage_state: Dict[str, Dict[str, Any]] = {}

_PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
    "/validate-key",
}


def _today_key() -> str:
    return date.today().isoformat()


def _get_usage_for_today(api_key: str) -> Dict[str, Any]:
    today = _today_key()
    state = _usage_state.get(api_key)
    if not state or state.get("date") != today:
        state = {"date": today, "count": 0}
        _usage_state[api_key] = state
    return state


def _remaining_calls(api_key: str) -> int:
    meta = VALID_KEYS[api_key]
    usage = _get_usage_for_today(api_key)
    limit = int(meta.get("daily_limit", 0))
    used = int(usage.get("count", 0))
    return max(0, limit - used)

app = FastAPI(
    title="Mood2Food AI API",
    description="A mood-aware food recommendation API for food delivery integrations.",
    version="1.0.0",
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    if path in _PUBLIC_PATHS or path.startswith("/docs"):
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(status_code=401, content={"detail": "Missing API key."})

    if api_key not in VALID_KEYS:
        return JSONResponse(status_code=401, content={"detail": "Invalid API key."})

    async with _usage_lock:
        meta = VALID_KEYS[api_key]
        usage = _get_usage_for_today(api_key)
        daily_limit = int(meta.get("daily_limit", 0))
        used_today = int(usage.get("count", 0))

        if used_today >= daily_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Daily API key limit exceeded.",
                    "plan": meta.get("plan"),
                    "daily_limit": daily_limit,
                },
            )

        usage["count"] = used_today + 1

    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend_router)


@app.get("/", tags=["System"])
def root() -> dict:
    """Return a root status payload for platform health checks."""

    return {"status": "ok", "service": "mood2food-api"}


@app.get("/health", tags=["System"])
def health_check() -> dict:
    """Return a simple health status payload."""

    return {"status": "ok"}


@app.get("/validate-key", tags=["System"])
async def validate_key(request: Request) -> dict:
    """Validate an API key and return plan details + remaining calls for today."""

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(status_code=401, content={"detail": "Missing API key."})

    if api_key not in VALID_KEYS:
        return JSONResponse(status_code=401, content={"detail": "Invalid API key."})

    async with _usage_lock:
        meta = VALID_KEYS[api_key]
        usage = _get_usage_for_today(api_key)
        daily_limit = int(meta.get("daily_limit", 0))
        used_today = int(usage.get("count", 0))
        remaining_today = max(0, daily_limit - used_today)

    return {
        "valid": True,
        "plan": meta.get("plan"),
        "daily_limit": daily_limit,
        "used_today": used_today,
        "remaining_today": remaining_today,
        "date": usage.get("date"),
    }
