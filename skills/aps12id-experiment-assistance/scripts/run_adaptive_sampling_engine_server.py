"""Serve one persistent spatial SAXS adaptive-sampling engine over HTTP.

Large array inputs should be saved with ``numpy.save`` and supplied to the API as
absolute ``.npy`` path strings. This avoids embedding large numeric arrays in agent tool
calls. Small arrays may still be supplied inline as JSON lists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eaa_aps12id.tools.spatial_saxs_sampling import (
    SpatialSAXSAdaptiveSamplingEngineTool,
)


app = FastAPI(title="Spatial SAXS adaptive-sampling engine")
engine = SpatialSAXSAdaptiveSamplingEngineTool()
engine_id = uuid4().hex
engine_lock = RLock()


class UpdateRequest(BaseModel):
    """Inputs for one engine update."""

    positions: list[Any] | str = Field(
        description=(
            "Small JSON array or, preferably for large data, an absolute .npy path."
        )
    )
    q_values: list[Any] | str = Field(
        description=(
            "Small JSON array or, preferably for large data, an absolute .npy path."
        )
    )
    intensities: list[Any] | str = Field(
        description=(
            "Small JSON array or, preferably for large data, an absolute .npy path."
        )
    )


class SuggestRequest(BaseModel):
    """Inputs for one adaptive suggestion call."""

    n_suggestions: int = Field(default=1, ge=1)


def load_npy_if_path(value: Any) -> Any:
    """Load an array when an API argument is an absolute ``.npy`` path.

    Parameters
    ----------
    value : Any
        A small inline JSON-compatible array or an absolute ``.npy`` path. Prefer the
        path form for large arrays so their values stay out of agent tool calls.

    Returns
    -------
    Any
        The loaded numeric array when ``value`` is a path; otherwise ``value``
        unchanged.
    """
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute() or path.suffix != ".npy":
        raise ValueError("Array paths must be absolute paths to `.npy` files.")
    return np.load(path, allow_pickle=False)


def call_engine(
    function: Callable[..., Any],
    array_arguments: tuple[str, ...] = (),
    **kwargs: Any,
) -> Any:
    """Call one engine operation while preserving state consistency."""
    try:
        for name in array_arguments:
            if name in kwargs:
                kwargs[name] = load_npy_if_path(kwargs[name])
        with engine_lock:
            return function(**kwargs)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def state_payload() -> dict[str, Any]:
    """Return a compact, JSON-safe summary of the current engine state."""
    return {
        "engine_id": engine_id,
        "initialized": engine.candidate_positions is not None,
        "candidate_count": (
            0 if engine.candidate_positions is None else len(engine.candidate_positions)
        ),
        "measurement_count": len(engine.measurements),
        "measured_candidate_indices": list(engine.measured_candidate_indices),
        "excluded_measurement_indices": sorted(engine.excluded_measurement_indices),
        "modeled_peak_ids": list(engine.modeled_peak_ids),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service identity and basic engine state."""
    with engine_lock:
        return {"status": "ok", **state_payload()}


@app.get("/state")
def state() -> dict[str, Any]:
    """Return the current engine state summary."""
    with engine_lock:
        return state_payload()


@app.post("/initialize")
def initialize(payload: dict[str, Any]) -> dict[str, Any]:
    """Initialize the engine, preferably loading large array arguments from ``.npy``."""
    call_engine(
        engine.initialize,
        ("candidate_positions", "known_peak_q_values"),
        **payload,
    )
    with engine_lock:
        return state_payload()


@app.post("/suggest_initial_measurements")
def suggest_initial_measurements() -> dict[str, Any]:
    """Return path-optimized positions for initial measurements."""
    positions = call_engine(engine.suggest_initial_measurements)
    return {"engine_id": engine_id, "positions": positions.tolist()}


@app.post("/update")
def update(payload: UpdateRequest) -> dict[str, Any]:
    """Update the engine, preferably loading large measurement arrays from ``.npy``."""
    call_engine(
        engine.update,
        ("positions", "q_values", "intensities"),
        **payload.model_dump(),
    )
    with engine_lock:
        return state_payload()


@app.post("/suggest")
def suggest(payload: SuggestRequest) -> dict[str, Any]:
    """Return adaptive measurement positions from the current model."""
    positions = call_engine(engine.suggest, n_suggestions=payload.n_suggestions)
    return {"engine_id": engine_id, "positions": positions.tolist()}


def main() -> None:
    """Run the single-process engine service on the loopback interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The engine service must bind only to the loopback interface.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
