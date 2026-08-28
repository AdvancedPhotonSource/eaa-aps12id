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

## Special case 1: launch the full workflow from Bash

Normally, launch the registered `SpatialSAXSAdaptiveSamplingTaskManager` as described
above. Use this fallback only when that class is not registered with the subagent tool,
or when the registered `run` API does not expose everything required for the requested
full adaptive-sampling workflow. This fallback still launches live, end-to-end
acquisition; it is not the procedure for isolated engine operations.

1. Copy
   [the backend-only driver template](../scripts/run_adaptive_sampling.py) to a working
   location and adapt the copy. Set the MCP endpoint, checkpoint and transcript paths,
   candidate positions, acquisition arguments, and task-manager arguments needed for
   the experiment. Preserve the candidate validation and coordinate conventions above.
2. Keep the copied script backend-only. Do not enable `use_webui`, start a WebUI runtime,
   call `launch_html_webui_subprocess`, or launch any other frontend process. The current
   WebUI will display the Bash process output.
3. Launch the adapted script with the Bash tool in its provided environment, for example
   with `uv run python /absolute/path/to/run_adaptive_sampling.py`. Set the Bash tool's
   `create_ui_tab` argument to `True` so the running script's output appears in a WebUI
   tab. Use a timeout suitable for the full acquisition workflow.
4. Treat a nonzero exit, timeout, MCP error, or task-manager error as a failed workflow;
   do not report completion merely because the UI tab was created.

## Special case 2: partial engine operations

Do not use the task manager interface if the user asks only to suggest, update,
analyze existing spectra, or perform another isolated phase. The task manager owns the
entire acquisition loop and may collect live data.

The Bayesian optimization-based suggestion and state update routines used in
`SpatialSAXSAdaptiveSamplingTaskManager` can also be used separately without the task
manager and the data collection logic. These routines are implemented in the
`SpatialSAXSAdaptiveSamplingEngineTool` class. This engine tool exposes the interfaces
of `initialize`, `suggest_initial_measurements`, `update`, and `suggest` for completing
individual steps in the adaptive sampling process.

Before viewing or calling the underlying algorithm, locate the installed package in the
same Python environment as EAA:

```bash
python -c "import inspect, eaa_aps12id; print(inspect.getfile(eaa_aps12id))"
python -c "import inspect; from eaa_aps12id.task_managers.spatial_saxs_sampling import SpatialSAXSAdaptiveSamplingTaskManager; from eaa_aps12id.tools.spatial_saxs_sampling import SpatialSAXSAdaptiveSamplingEngineTool; print(inspect.getfile(SpatialSAXSAdaptiveSamplingTaskManager)); print(inspect.getfile(SpatialSAXSAdaptiveSamplingEngineTool))"
```

The relevant modules are:

- `eaa_aps12id.task_managers.spatial_saxs_sampling`: the task manager that calls the
  engine tool in the defined workflow
- `eaa_aps12id.tools.spatial_saxs_sampling`: the engine tool itself

### Persistent engine service

The engine is stateful but is not expected to be registered as an MCP server. Keep one
engine instance alive in a local FastAPI process so later agent turns can call the same
instance:

**For large arrays, always save the numeric data to `.npy` files and pass only their
absolute paths in the JSON request.** Do not serialize a large candidate grid, q array,
intensity array, or measurement batch into a tool call. Inline JSON arrays are suitable
only for small inputs. Passing paths keeps agent requests compact and lets the server
load the arrays without the agent reproducing their values.

1. Copy
   [the engine server template](../scripts/run_adaptive_sampling_engine_server.py) to a
   working location. Inspect the installed engine source first and adapt the copy if its
   public method signatures differ from the template. Bind only to `127.0.0.1`, choose an
   unused port, and do not configure multiple Uvicorn workers or auto-reload; either
   would create or replace engine instances.
2. Launch the copied server in the foreground with the Bash tool, for example:

   ```bash
   uv run python /absolute/path/to/run_adaptive_sampling_engine_server.py --port 8765
   ```

   The Bash tool is preconfigured with a `release_timeout`: when it releases the agent
   loop, the server keeps running. Do not append `&`, use `nohup`, or launch a second
   copy. Set `create_ui_tab` to `True` so server output remains visible in the current
   WebUI.
3. Call `GET /health` after launch. Record both the base URL and returned `engine_id`.
   Before every later state-changing call, check `/health` or `/state` and require the
   same `engine_id`. A different ID means the process restarted and the prior in-memory
   engine state is gone.
4. Call the HTTP endpoints in the engine's required stateful order:

   | Endpoint | Purpose |
   | --- | --- |
   | `POST /initialize` | Configure a new engine instance. Send the engine's `initialize` keyword arguments as one JSON object. |
   | `POST /suggest_initial_measurements` | Return the initial path-optimized positions. Call only before any update. |
   | `POST /update` | Add measured `positions`, `q_values`, and `intensities`, then refit the model. |
   | `POST /suggest` | Return adaptive positions. Send `{"n_suggestions": 1}` or another positive batch size. |
   | `GET /state` | Check the engine ID, configuration status, and recorded measurement indices/count. |

   Each array argument accepts either a small inline JSON array or an absolute `.npy`
   path string. The path form is supported for `candidate_positions`,
   `known_peak_q_values`, `positions`, `q_values`, and `intensities`. Prefer it whenever
   an array would make the request large. The server process must be able to read the
   file, and `.npy` contents must be regular numeric arrays loadable with
   `allow_pickle=False`.

Save large inputs before making the HTTP request:

```python
import numpy as np

np.save("/shared/beamtime/candidates.npy", candidate_positions)
np.save("/shared/beamtime/measured_positions.npy", measured_positions)
np.save("/shared/beamtime/q_values.npy", q_values)
np.save("/shared/beamtime/intensities.npy", intensities)
```

The corresponding request contains path strings, not the array values:

```bash
curl -sS -X POST http://127.0.0.1:8765/initialize \
  -H 'Content-Type: application/json' \
  -d '{"candidate_positions":"/shared/beamtime/candidates.npy","q_min":0.01,"q_max":0.8,"num_initial_samples":5}'

curl -sS -X POST http://127.0.0.1:8765/update \
  -H 'Content-Type: application/json' \
  -d '{"positions":"/shared/beamtime/measured_positions.npy","q_values":"/shared/beamtime/q_values.npy","intensities":"/shared/beamtime/intensities.npy"}'

curl -sS -X POST http://127.0.0.1:8765/suggest \
  -H 'Content-Type: application/json' \
  -d '{"n_suggestions":1}'
```

The engine's stateful sequence is `initialize(...)`,
`suggest_initial_measurements()`, `update(positions, q_values, intensities)`, then
`suggest(n_suggestions=...)` and further updates. Do not reinitialize or stop the server
while later updates are expected. The template serializes engine operations, but the
agent should still avoid concurrent mutations.

The service does not acquire data and does not persist engine state across process exits.
If it stops, reconstruct state only by initializing with the exact original configuration
and replaying every measured position with its corresponding q/intensity spectrum in the
original order. If those inputs are unavailable, tell the user what is missing instead
of fabricating measurements, inferred state, or recommendations.
