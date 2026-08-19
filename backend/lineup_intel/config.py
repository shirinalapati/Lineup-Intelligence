from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LI_", env_file=".env", extra="ignore")

    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    processed_dir: Path = ROOT / "data" / "processed"
    artifacts_dir: Path = ROOT / "data" / "artifacts"
    models_dir: Path = ROOT / "data" / "models"
    vendor_models_dir: Path = ROOT / "vendor" / "diamondiq_models"

    # Read-only DiamondIQ sources (never write)
    diamondiq_db: Path = Path("/Users/Shirin/Diamond_IQ/data/diamondiq.db")
    gumbo_cache: Path = Path("/Users/Shirin/Diamond_IQ/data/gumbo_cache")
    undervalued_stats_2025: Path = Path(
        "/Users/Shirin/Undervalued_MLBPlayers/data/processed/comprehensive_stats_2025.csv"
    )
    undervalued_stats_2026: Path = Path(
        "/Users/Shirin/Undervalued_MLBPlayers/data/processed/comprehensive_stats_2026.csv"
    )

    database_url: str = f"sqlite:///{ROOT / 'data' / 'lineup_intel.db'}"
    cors_origins: list[str] = ["*"]
    target_season: int = 2026
    train_seasons: list[int] = [2024, 2025]
    equivalence_eps: float = 0.02  # runs/game — operationally equivalent band


settings = Settings()
for p in (settings.data_dir, settings.processed_dir, settings.artifacts_dir, settings.models_dir):
    p.mkdir(parents=True, exist_ok=True)