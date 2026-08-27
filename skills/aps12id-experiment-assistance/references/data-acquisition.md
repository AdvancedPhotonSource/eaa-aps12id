# APS 12-ID data acquisition

The server is a thin adapter. It does not access EPICS directly. Main-beamline commands
and multi-heater (MH) plate commands use separate ZMQ agent ports and separate control
leases. The beamline processes enforce access policy, live motor limits, hardware safety,
rate limits, and auditing. A tool call is a request, not proof that hardware acted.

## Before actuation

1. Use `get_state` to inspect beam, beamstop, detector, motor, warning, and lease state.
2. Use `list_devices` when a logical device name or whether it is settable is uncertain.
3. Confirm requested coordinates, exposure, plate/sample identifiers, and magazine slot.
4. After an actuation, check its synchronous result or poll the documented lifecycle tool.
5. If a result begins with `ERROR`, stop that operation and report the complete error.
   Do not bypass a rejection with another tool or modified limits.

## Main beamline tools

| Tool | Arguments | Behavior and result |
| --- | --- | --- |
| `list_devices` | none | Returns JSON `{name: {kind, pv, settable}}` from the beamline device configuration. |
| `read_device` | `name: str`, `field: str = ""` | Reads a logical device. Motor fields include `rbv` (default), `val`, `setpoint`, `llm`, `hlm`, and `dmov`. Returns a value string or `ERROR: ...`. |
| `set_device` | `name: str`, `value: float` | Acquires the main control lease and requests a safety-gated set/move. Success is `ACCEPTED`; rejection is returned verbatim. Use only for a confirmed settable device. |
| `get_state` | none | Returns the machine-readable state snapshot, including beam, beamstop match, detector thresholds, motor readbacks and limits, counters, lease holder, and warnings. |
| `acquire_saxs_data_file` | `x: float`, `y: float`, `exposure: float \| null = null` | Serializes acquisition calls, acquires the main lease, requires the DAQ to be idle, moves the configured SAXS motors, acquires and waits for reduction, validates the HDF5 result, and returns the file reference. Exposure must be positive; omission uses the server default (currently 0.5 s unless deployment overrides it). |

`acquire_saxs_data_file` returns this shape:

```json
{
  "data_file": {
    "name": "saxs_spectrum",
    "type": "hdf5",
    "path": "/server-visible/path/reduced.h5",
    "hdf5_dataset_path": "/entry/result/I",
    "q_hdf5_dataset_path": "/entry/result/q"
  }
}
```

The file path is on the MCP server host and must also be visible to any process that
opens it. The acquisition tool moves the configured spatial x motor first and y motor
second, waits for both readbacks and `dmov`, then starts one exposure. A timeout during
an active exposure requests an abort. Do not issue parallel acquisitions.

## Multi-heater and plate tools

Read-only status calls do not acquire the MH lease. Actuating calls acquire and renew it.

| Tool | Arguments | Behavior and result |
| --- | --- | --- |
| `mh_status` | none | Returns a concise MH sample/plate status line. |
| `mh_get_acq_status` | none | Returns acquisition-status JSON containing `isRunning`, `currentRow`, `MaxSampleN`, `plate_id`, `sample_id`, and `scan_id`. Poll after `run_plate` or `run_sample`. |
| `mh_get_last_action` | none | Returns lifecycle JSON for the most recent asynchronous MH action: accepted/running/done/error plus detail. |
| `load_plate` | `code: str` | Synchronously populates the MH sample table from PVapp. Expected success resembles `OK loaded plate=<code> samples=<n>`. It does not require a lease. |
| `mount_plate` | `magazine_pos: int` | Asynchronously robot-loads the plate from the specified magazine slot. Poll `mh_get_last_action` to completion. |
| `unmount_plate` | `magazine_pos: int = -1` | Asynchronously returns the staged plate. A negative or omitted slot uses its current slot. Poll `mh_get_last_action`. |
| `autoreference` | `tol: float = 0.0` | Asynchronously scans H and V from the saved reference, centers, and establishes first-sample zero. Positive `tol` overrides alignment tolerance. Poll `mh_get_last_action`. |
| `align_plate` | `dry_run: bool = false` | Asynchronously performs three-step photodiode roll calibration. Dry-run computes and stores the fit without rewriting the table. Poll `mh_get_last_action`. |
| `apply_roll` | none | Asynchronously rewrites the current sample table using the saved/latest roll fit without rescanning. Poll `mh_get_last_action`. |
| `goto_sample` | `plate_id: str`, `sample_id: str` | Moves to one sample without acquiring. This is an actuation and uses the MH lease. |
| `run_sample` | `plate_id: str`, `sample_id: str` | Starts one sample asynchronously. Follow progress with `mh_get_acq_status`; the stage must have a valid first-sample reference. |
| `run_plate` | `plate_id: str` | Starts all enabled samples asynchronously. Follow progress with `mh_get_acq_status`; the stage must have a valid first-sample reference. |

For an asynchronous action, poll at a reasonable interval until the corresponding
status reports done or error. Stop polling once terminal, and never automatically rerun
a failed robot, alignment, or acquisition action.

## Typical plate workflow

Use only the steps the operator requests and preserve facility procedures:

1. `load_plate(code)` and verify the returned sample count.
2. `mount_plate(magazine_pos)` and poll `mh_get_last_action`.
3. Establish or verify the first-sample reference with `autoreference`; use
   `align_plate`/`apply_roll` only when requested or required by the operator's setup.
4. Start `run_sample` or `run_plate` and poll `mh_get_acq_status`.
5. Unmount only when the run is no longer active and the operator requested it.

Never infer a magazine position, plate code, sample ID, or whether an existing roll fit
is valid from unrelated state.
