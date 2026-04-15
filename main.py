"""FastAPI application entrypoint for the Mood2Food API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.recommend import router as recommend_router

app = FastAPI(
    title="Mood2Food AI API",
    description="A mood-aware food recommendation API for food delivery integrations.",
    version="1.0.0",
)

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
