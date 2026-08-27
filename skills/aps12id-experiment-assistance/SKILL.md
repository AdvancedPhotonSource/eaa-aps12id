---
name: aps12id-experiment-assistance
description: Use whenever a user asks for data acquisition, data analysis, or experiment steering that may be specific to APS 12-ID, including beamline DAQ, spatial SAXS adaptive sampling, or AFL composition-space steering.
---

# APS 12-ID experiment assistance

Route the request to the relevant reference below and read only that reference before
acting. If a request spans multiple modes, read each applicable reference. Confirm
experiment-specific identifiers, coordinates, exposure, budgets, and input paths rather
than inventing them. Treat tool results that report policy, lease, safety, transport, or
data-validation errors as failures; do not silently continue or retry an actuation with
changed parameters.

## Data acquisition

Use the APS 12-ID SAXSDaq MCP tools to inspect beamline state, read or move configured
devices, acquire reduced SAXS files, and operate the multi-heater sample/plate workflow.
The server is a safety-gated adapter to the beamline control processes, so live state and
returned error messages are authoritative.

Read [references/data-acquisition.md](references/data-acquisition.md).

## Spatial SAXS adaptive sampling

Use the registered spatial SAXS task manager for an end-to-end spatial
measure-update-suggest loop over a finite candidate set. Candidate coordinates are
ordered `(y, x)`, and large candidate sets must be passed through a server-visible
`.npy` file. Do not launch the full workflow when the user requests only an isolated
engine operation.

Read
[references/spatial-saxs-adaptive-sampling.md](references/spatial-saxs-adaptive-sampling.md).

## AFL composition-space steering

Use the AFL steering MCP server as a single-iteration recommendation engine for mapping
SAXS-derived properties or phases in chemical-composition space. It reads an externally
updated Zarr snapshot, returns compact recommendations, and writes a separate full result
Zarr. It does not acquire data or maintain the experiment loop.

Read [references/afl-steering.md](references/afl-steering.md).
