"""
main.py
-------
AYU FastAPI application entry point.

Registers:
  - CORS middleware (allows the Next.js frontend to call the API)
  - All routers
  - Startup / shutdown lifecycle events
  - Global exception handler for unhandled errors

Run with:
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env before anything else reads os.getenv
load_dotenv()

try:
    from routers.reports import router as reports_router  # type: ignore  # noqa: E402
    from routers.chat import router as chat_router  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    from backend.routers.reports import router as reports_router  # type: ignore  # noqa: E402
    from backend.routers.chat import router as chat_router  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ayu.main")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AYU backend starting up…")
    logger.info("Groq model: %s", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    yield
    logger.info("AYU backend shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AYU — AI Health Intelligence API",
    description=(
        "AYU provides educational analysis of medical reports. "
        "This is NOT a diagnostic tool. Always consult a qualified healthcare professional."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# CORS — allow local Next.js dev server and production domain
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "http://localhost:3000",   # Next.js dev
    "http://127.0.0.1:3000",
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — never expose stack traces to clients
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected server error occurred. Please try again.",
            "disclaimer": (
                "AYU provides educational insights only. "
                "This is NOT a medical diagnosis."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(reports_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "AYU Health Intelligence API",
        "version": "0.1.0-prototype",
        "docs": "/docs",
        "status": "running",
    }
