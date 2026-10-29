import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, check_database_connection, engine
from app.routes.products import router as products_router

load_dotenv("backend/.env")

APP_NAME = os.getenv("APP_NAME", "EconoCausal API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Importing models registers their tables with Base.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    yield

    engine.dispose()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="EconoCausal backend API",
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(products_router)


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the EconoCausal API",
        "version": APP_VERSION,
        "docs": "/docs",
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "EconoCausal API",
    }


@app.get("/api/health/database", tags=["Health"])
def database_health_check():
    try:
        check_database_connection()

        return {
            "status": "healthy",
            "database": "connected",
        }

    except Exception:
        return {
            "status": "unhealthy",
            "database": "disconnected",
        }