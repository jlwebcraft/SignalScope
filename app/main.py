import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import inference_engine, router as v1_router
from app.inference.artifacts import ModelArtifactManager
from app.utils.logger import logger

APP_VERSION = "0.1.0"
APP_TITLE = "SignalScope API"
APP_DESCRIPTION = (
    "Production-grade multimodal authenticity detection API for generative media. "
    "Features evidence fusion across RGB spatial anomalies, Fourier frequency cues, "
    "provenance metadata, and authenticity stability scoring under degradation."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown event handler."""
    logger.info("==================================================")
    logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
    logger.info("Telling Real From Synthetic in the Age of Generative Media")
    logger.info("SIH 2026 Internal Hackathon")
    logger.info("==================================================")
    yield
    logger.info(f"Shutting down {APP_TITLE}...")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS) configuration
frontend_origins_env = os.environ.get("FRONTEND_ORIGIN", "")
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
if frontend_origins_env:
    for origin in frontend_origins_env.split(","):
        cleaned = origin.strip()
        if cleaned and cleaned not in allowed_origins:
            allowed_origins.append(cleaned)

is_production = os.environ.get("APP_ENV", "").lower() == "production"
cors_origins = allowed_origins if (is_production or frontend_origins_env) else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(v1_router)


@app.get("/", tags=["root"], summary="Root status check")
async def root():
    """Root endpoint providing service metadata and documentation links."""
    return {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "api_v1": "/api/v1",
    }


@app.get("/health", tags=["root"], summary="Root health check")
async def health():
    """Convenience root health endpoint for container liveness probes."""
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION,
    }


@app.get("/ready", tags=["root"], summary="Root readiness check")
async def ready():
    """Root readiness probe distinguishing service boot from model initialization."""
    if not inference_engine.is_ready or not inference_engine.has_trained_weights:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": "Model weights are initializing or unavailable.",
            },
        )
    return {
        "status": "ready",
        "model_version": ModelArtifactManager.get_version_info()["model_version"],
        "device": inference_engine.device,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

