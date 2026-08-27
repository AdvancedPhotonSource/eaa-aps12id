# Spatial SAXS adaptive sampling

`SpatialSAXSAdaptiveSamplingTaskManager` is an end-to-end workflow. It initializes a
stateful active-learning engine, selects path-optimized initial positions, acquires SAXS,
updates peak Gaussian-process models, suggests new positions, and repeats until its
measurement or iteration budget is reached. It also publishes progress, measured spectra,
and posterior/acquisition visualizations to the EAA WebUI when a runtime is attached.

## Launching the registered workflow

The workflow may be registered with EAA's sub-task manager tool.

1. Call `subagent_tool.list_registered_task_managers` and find the entry whose class is
   `SpatialSAXSAdaptiveSamplingTaskManager`. A common registered name is
   `spatial_saxs_adaptive_sampling`, but use the returned name rather than assuming it.
2. Prepare and validate all `run` arguments before starting. This tool launches live,
   end-to-end acquisition; listing the registry is the read-only discovery step.
3. Call `subagent_tool.launch_subtask_manager` with the discovered name and a
   `task_manager_kwargs` mapping containing the `run` arguments.

```json
{
  "task_manager_name": "spatial_saxs_adaptive_sampling",
  "task_manager_kwargs": {
    "candidate_positions": "/absolute/shared/path/candidate_positions.npy",
    "q_min": 0.01,
    "q_max": 0.8,
    "num_initial_samples": 10,
    "max_measurements": 50,
    "non_position_kwargs_for_acquisition_tool": {"exposure": 0.5}
  }
}
```

### Candidate positions

`candidate_positions` must be a nonempty finite numeric array of shape `(N, 2)`, with
unique rows. Each row is `(y, x)`, not `(x, y)`. Coordinates must use the same physical
frame and units as the acquisition tool; supplied row order is retained.

When the candidate set contains many points, **always write the numeric array to a
`.npy` file first and pass its path**. Do not put the full array into the tool call. The
path must be readable by the EAA/task-manager process, preferably an absolute path on a
filesystem shared with that process.

```python
from pathlib import Path
import numpy as np

positions = np.column_stack((y_grid.ravel(), x_grid.ravel()))  # columns: y, x
path = Path("/shared/beamtime/candidate_positions.npy")
np.save(path, positions)
```

## `run` arguments

### Candidate set, q preprocessing, and budget

| Argument | Default | Meaning |
| --- | --- | --- |
| `candidate_positions` | required | `(N, 2)` array/list in `(y, x)` order, or path to a `.npy` containing it. |
| `q_min` | `0.001` | Lower endpoint of the engine's common log-spaced q grid; must be positive and below `q_max`. |
| `q_max` | `1.0` | Upper endpoint of the common q grid. Each raw spectrum must cover the full `[q_min, q_max]` interval. |
| `num_q_points` | `256` | Number of points in the common log-q grid; at least 2. |
| `epsilon_intensity` | `1e-12` | Positive numerical offset used before taking log intensity. Raw values must remain greater than its negative. |
| `num_initial_samples` | `5` | Number of unique path-optimized Sobol candidates acquired before adaptive suggestions. |
| `max_measurements` | `20` | Total budget including initial measurements. It must satisfy `num_initial_samples <= max_measurements <= N` and cannot change after measurements exist in this manager. |
| `n_iterations` | `null` | Maximum adaptive measurements added by this call after initial acquisition. Omission consumes the remaining total budget. This counts measurements, not GP update batches. |
| `random_seed` | `null` | Seed for reproducible randomized selection, including scrambled Sobol initialization. |

### Spatial eligibility and batching

| Argument | Default | Meaning |
| --- | --- | --- |
| `exclusion_radius` | `null` | Nonnegative distance in physical coordinate units; candidates at or inside this radius of any measured point are ineligible. |
| `suggestion_exclusion_radius` | `null` | Nonnegative physical distance enforced between members of the same suggestion batch. |
| `num_candidates_per_suggestion` | `1` | Positive batch size requested from the engine. Positions are acquired sequentially in travel-optimized order, then the engine is updated once with the batch. |
| `exploration_interval` | `5` | Positive adaptive-measurement interval at which a farthest-point exploration candidate is included. `null` disables scheduled farthest-point exploration. |

### Background and peak extraction

| Argument | Default | Meaning |
| --- | --- | --- |
| `background_valley_smoothing_sigma` | `3.0` | Positive Gaussian smoothing width, in log-q grid samples, used to identify spectral valleys and for the fallback background. |
| `background_valley_min_prominence` | `0.05` | Nonnegative minimum valley prominence in log-intensity. |
| `background_smoothness` | `1e6` | Positive arPLS smoothness setting retained by the API. The current measurement preprocessing path uses valley/PCHIP background fitting and does not call arPLS, so changing this currently has no effect on `run`. |
| `background_max_iterations` | `50` | Positive arPLS iteration cap retained by the API; currently not used by the `run` preprocessing path. |
| `background_tolerance` | `1e-3` | Positive arPLS convergence tolerance retained by the API; currently not used by the `run` preprocessing path. |
| `peak_smoothing_sigma` | `1.0` | Nonnegative Gaussian smoothing width in common-grid samples before peak detection; zero disables smoothing. |
| `peak_min_height` | `1.0` | Nonnegative minimum natural-log robust signal-to-noise height. |
| `peak_min_prominence` | `1.0` | Nonnegative minimum natural-log peak-to-local-base prominence. |
| `peak_min_width_log_q` | `0.03` | Minimum detected half-height peak width in log-q. |
| `peak_max_width_log_q` | `null` | Optional maximum peak width in log-q; when set, it must exceed the minimum. |
| `peak_window_width_factor` | `2.0` | Positive multiplier that converts detected width into the frozen peak integration window. |
| `peak_observable` | `"area"` | GP target: integrated positive background-subtracted `area` or peak `height`. |

### Peak dictionary and GP model

| Argument | Default | Meaning |
| --- | --- | --- |
| `num_initial_peaks` | `5` | Positive maximum number of strongest peaks seeded into a dynamically discovered peak dictionary. |
| `max_peaks_in_dict` | `10` | Maximum dynamic dictionary size; at least `num_initial_peaks`. |
| `known_peak_q_values` | `null` | Optional authoritative positive, unique q positions within `[q_min, q_max]`. When supplied, use these instead of relying on dynamic discovery. |
| `new_peak_min_relative_area` | `0.001` | Nonnegative threshold for admitting a new detected peak relative to the strongest peak area. |
| `peak_area_scale` | `1.0` | Positive scale in the GP target transform `log1p(observable / scale)`; the legacy name applies even when modeling height. |
| `max_fit_gp_mll_iterations` | `null` | Optional positive maximum optimizer iterations for GP marginal-likelihood fitting; omission leaves it unlimited by this workflow. |
| `epsilon_z` | `1e-12` | Positive floor in target standardization and inverse transformation. |

### Acquisition scoring

The adaptive score is normalized uncertainty multiplied by an exploration floor plus
weighted peak-observable and spatial-gradient terms.

| Argument | Default | Meaning |
| --- | --- | --- |
| `peak_map_min_concentration` | `0.2` | `[0, 1]` concentration gate for including a modeled peak in the peak-observable score. |
| `peak_observale_map_blur` | `null` | Optional nonnegative Gaussian spatial blur width in physical coordinate units. The misspelling `observale` is part of the current public API and must be passed exactly. |
| `w_peak` | `1.0` | Nonnegative weight for normalized predicted peak observable. |
| `w_g` | `1.0` | Nonnegative weight for normalized predicted spatial gradient. |
| `epsilon_acquisition` | `1e-3` | Positive acquisition floor that preserves uncertainty-only exploration. |
| `epsilon_normalization` | `1e-12` | Positive denominator floor used by acquisition-term and concentration normalization. |
| `normalization_lower_percentile` | `5.0` | Lower percentile for robust score normalization. |
| `normalization_upper_percentile` | `95.0` | Upper percentile; must be above the lower percentile and no greater than 100. |

### Acquisition adapter arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `non_position_kwargs_for_acquisition_tool` | `null` | Extra keyword arguments forwarded to `acquire_saxs(x=..., y=...)`, commonly `exposure`. With `null`, the manager passes engine `q_min` and `q_max` so the APS adapter validates returned q coverage. If a mapping is supplied, it replaces that default mapping rather than merging with it; include `q_min`/`q_max` explicitly when coverage validation is still wanted. |

## Partial engine operations

Do not launch the registered task manager if the user asks only to suggest, update,
analyze existing spectra, or perform another isolated phase. The task manager owns the
entire acquisition loop and may collect live data.

Before viewing or calling the underlying algorithm, locate the installed package in the
same Python environment as EAA:

```bash
python -c "import inspect, eaa_aps12id; print(inspect.getfile(eaa_aps12id))"
python -c "import inspect; from eaa_aps12id.task_managers.spatial_saxs_sampling import SpatialSAXSAdaptiveSamplingTaskManager; from eaa_aps12id.tools.spatial_saxs_sampling import SpatialSAXSAdaptiveSamplingEngineTool; print(inspect.getfile(SpatialSAXSAdaptiveSamplingTaskManager)); print(inspect.getfile(SpatialSAXSAdaptiveSamplingEngineTool))"
```

The relevant modules are:

- `eaa_aps12id.task_managers.spatial_saxs_sampling`
- `eaa_aps12id.tools.spatial_saxs_sampling`

The engine's stateful sequence is `initialize(...)`,
`suggest_initial_measurements()`, `update(positions, q_values, intensities)`, then
`suggest(n_suggestions=...)` and further updates. Custom code must preserve the exact
candidate set, configuration, measured positions, and corresponding q/intensity spectra.
If a partial request lacks the engine state or enough data to reconstruct it reliably,
tell the user what is missing instead of fabricating measurements, inferred state, or
recommendations.
