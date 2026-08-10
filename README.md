# EAA workflows and tools for APS 12-ID

This package provides workflow controllers, active-learning engines, and SAXS
acquisition adapters for automated experiments at APS 12-ID. The spatial SAXS
implementation separates experiment orchestration from the Bayesian decision logic so
the two parts can be used together or integrated independently.

## Organization

```text
src/eaa_aps12id/
├── task_managers/
│   └── spatial_saxs_sampling.py   # acquisition workflow and run control
└── tools/
    ├── spatial_saxs_sampling.py   # Bayesian active-learning engine
    ├── aps12id_saxs.py            # APS 12-ID SAXS acquisition adapter
    └── spatial_saxs.py            # simulated spatial SAXS acquisition
```

## Spatial SAXS task manager

### `SpatialSAXSAdaptiveSamplingTaskManager`

The task manager is the high-level experiment controller. It owns the
`acquisition_tool` and is responsible for:

- enforcing the total measurement budget and per-call iteration limit;
- requesting initial and adaptive positions from the engine;
- acquiring SAXS data at the returned positions, in the returned order, using the `acquisition_tool`;
- batching the measured positions and q-intensity arrays into one engine update;
- forwarding non-position acquisition arguments; and
- publishing progress and posterior visualizations to the EAA WebUI.

The manager does not implement any logic like Gaussian process to suggest new
measurements or accept new data. These logics are implemented in the `engine_tool`
and the task manager only queries the tool for suggestions and updates. It creates a
`SpatialSAXSAdaptiveSamplingEngineTool` by default, or accepts a supplied engine through
the `engine_tool` constructor argument.

Calling `run(...)` performs the following loop:

1. Initialize the engine with the candidate positions and active-learning configuration.
2. Ask the engine for path-optimized initial positions.
3. Acquire the initial SAXS spectra and update the engine once with the complete batch.
4. Ask the engine for the next path-optimized batch, acquire it, and update the engine.
5. Repeat until the manager-owned measurement budget or iteration limit is reached.

## Spatial SAXS active-learning engine

### `SpatialSAXSAdaptiveSamplingEngineTool`

The engine is a stateful EAA `BaseTool` that owns the complete measurement-suggestion
logic. It never collects data and has no acquisition backend.

Its exposed interface is:

- `initialize(...)`: configure the candidate positions, preprocessing, peak-detection, GP,
  and acquisition parameters.
- `suggest_initial_measurements()`: select initial candidate positions with scrambled Sobol
  sampling and return them in a travel-optimized order.
- `update(positions, q_values, intensities)`: preprocess one or more new spectra,
  update the peak dictionary, and refit the GP models once for the batch.
- `suggest(n_suggestions=1)`: select the next eligible position or batch and return it
  in a travel-optimized order.

### Peak detection and GP model

For every supplied SAXS measurement, the engine:

1. interpolates the raw spectrum onto a common log-spaced q grid;
2. estimates and subtracts a smooth spectral background;
3. detects peaks using configurable height, prominence, and log-q width criteria; and
4. records either integrated peak area or peak height as the modeled observable.

The engine can use authoritative `known_peak_q_values`, or it can build and update a
dynamic peak dictionary from measured spectra. It fits an independent Gaussian process
for each active peak using normalized spatial coordinates and transformed peak
observables.

### Acquisition function

For each eligible, unmeasured candidate, the engine computes

```text
acquisition = normalized_uncertainty × (
    epsilon_acquisition
    + w_peak × normalized_peak_observable
    + w_g × normalized_spatial_gradient
)
```

The peak-observable term favors positions where an eligible peak is predicted to be
strong, while the gradient term favors spatial boundaries and rapid changes. Peak maps
can be concentration-gated and spatially blurred before scoring. Configurable exclusion
radii prevent suggestions near measured points or near other members of the same batch.
Periodic farthest-point exploration can be enabled with `exploration_interval`.

After selecting a batch, the engine orders it with a low-travel path beginning at the
latest measured position. Initial suggestions are also path optimized.

## Standalone engine usage

A custom workflow can use the engine with any data-collection backend:

```python
from eaa_aps12id.tools import SpatialSAXSAdaptiveSamplingEngineTool

engine = SpatialSAXSAdaptiveSamplingEngineTool()
engine.initialize(
    candidate_positions=[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
)

positions = engine.suggest_initial_measurements()
q_values, intensities = acquire_with_custom_backend(positions)
engine.update(positions, q_values, intensities)

while more_measurements_are_needed():
    positions = engine.suggest(n_suggestions=2)
    q_values, intensities = acquire_with_custom_backend(positions)
    engine.update(positions, q_values, intensities)
```

`q_values` and `intensities` may contain arrays of different lengths for different
measurements. Each update must also include the corresponding spatial positions from
the configured candidate set. Candidate-position columns are ordered ``(y, x)``;
coordinates otherwise use the same unit and frame as the acquisition tool.

## Acquisition tools

### `APS12IDSAXSAcquisitionTool`

Connects the logic-driven workflow to the APS 12-ID SAXS data-acquisition MCP server.
It requests a reduced data file, loads the q and intensity arrays, and validates the
requested q coverage.

### `SimulatedSpatialSAXS`

Loads measured SAXS data and metadata from disk, constructs a spatial interpolation
backend, and returns simulated q-intensity arrays through the same `acquire_saxs`
interface used by the task manager.
