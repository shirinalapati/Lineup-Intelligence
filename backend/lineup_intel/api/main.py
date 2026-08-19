"""FastAPI application entrypoint for MLB Lineup Intelligence."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from .routers import (
    health,
    league,
    lineups,
    optimizer,
    ops,
    players,
    research,
    search,
    synergy,
    teams,
)
from .scheduler import start_daily_refresh_scheduler


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_daily_refresh_scheduler()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLB Lineup Intelligence",
        description="Batting-order research application for the 2026 MLB season.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (
        health.router,
        league.router,
        teams.router,
        lineups.router,
        optimizer.router,
        players.router,
        synergy.router,
        research.router,
        search.router,
        ops.router,
    ):
        app.include_router(router, prefix="/api")

    @app.get("/")
    def root():
        return {
            "name": "MLB Lineup Intelligence",
            "docs": "/docs",
            "api": "/api",
        }

    return app


app = create_app()
