# AFL composition-space steering

The AFL steering server is a single-iteration recommendation engine for
exploring SAXS-derived sample properties or phases in chemical-composition space. It is
not the spatial `(x, y)` adaptive sampler and it does not implement data acquisition,
sample preparation, experiment-history updates, or a suggest-measure-update loop.

## `recommend_next`

Call with one server-local path:

```json
{
  "input_zarr_path": "/absolute/path/on/server/experiment_snapshot.zarr"
}
```

The input must be an immutable, consolidated `.zarr` directory under a configured
`server.allowed_data_roots`. It must contain:

| Variable/coordinate | Required shape or role |
| --- | --- |
| `I` | Data variable with dimensions `(sample, q)`. |
| `composition` | Data variable with dimensions `(sample, component)`. |
| `composition_grid` | Candidate data variable with dimensions `(grid, component)`. |
| `q` | Positive, finite, strictly increasing coordinate. |
| `component` | Coordinate whose names and order exactly match the startup profile. |

The external experiment workflow must finish publishing the snapshot before this call.
Never update a directory store in place while the server is reading it. The server reads
the snapshot once, runs preprocessing, similarity, spectral clustering, GP phase
classification, and maximum-variance acquisition, then writes a newly named consolidated
result Zarr. It never modifies the input.

The compact JSON response includes:

- `config_revision` and algorithm profile/version/seed;
- input path, counts, and component order;
- a short `recommendations` list with rank, candidate grid index, component values and
  units, predicted cluster label, predictive variance, and acquisition value;
- `result_zarr_path`, the server-local path to the full result dataset; and
- `effective_settings`, the settings snapshot used for this calculation.

Cluster labels are algorithmic integer labels, not intrinsically named physical phases.
Use an experiment-owned mapping before describing them as specific phases. The returned
result path must be visible to any later analysis process.

## `update_setting`

Update one existing dotted algorithm key for subsequent calls:

```json
{
  "key": "clustering.n_phases",
  "value": 7
}
```

The `algorithm.` prefix is optional. Useful examples include
`clustering.n_phases`, `acquisition.count`, `acquisition.exclusion_radius`,
`savgol.q_max`, and existing component bounds such as
`composition.maximum.fingolimod`. The server validates the complete proposed settings
atomically. A successful change increments `config_revision`; a rejected change leaves
the active settings unchanged.

Runtime updates:

- affect only recommendations that start after the update;
- are kept in memory and do not rewrite YAML;
- disappear when the server restarts; and
- cannot change host, port, MCP path, allowed data roots, or result directory.

Record requested runtime changes and check the `effective_settings` returned by
`recommend_next`. For a durable scientific change, ask the workflow owner to review and
update the startup YAML rather than relying on an undocumented runtime mutation.

## Operational boundary

A normal composition-space iteration is external orchestration:

1. Another system acquires data and publishes a new immutable Zarr snapshot.
2. Call `recommend_next` with that snapshot.
3. Use the compact recommendation to plan the next composition measurement.
4. Preserve `result_zarr_path` for analysis/provenance.
5. After measurement, the external system publishes the next snapshot.

Do not claim the AFL server measured a sample or updated experiment history. If the user
requests a multi-iteration autonomous loop, sample preparation, or state update, identify
the missing external components and obtain an explicit plan for integrating them rather
than simulating those actions inside `recommend_next`.
