"""SignalScope Application Entry Point.

FastAPI service for AI-generated synthetic media detection and evidence fusion.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import router as v1_router
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

# Cross-Origin Resource Sharing (CORS) for local Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
        "api_v1": "/api/v1",
    }


@app.get("/health", tags=["root"], summary="Root health check")
async def health():
    """Convenience root health endpoint for container health probes."""
    return {"status": "healthy", "service": APP_TITLE, "version": APP_VERSION}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
