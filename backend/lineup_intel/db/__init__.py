"""Artifact-backed data access for MLB Lineup Intelligence."""

from .store import ArtifactStore, get_store, unavailable

__all__ = ["ArtifactStore", "get_store", "unavailable"]
