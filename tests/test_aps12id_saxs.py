import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from eaa_core.tool.base import ExposedToolSpec
from eaa_aps12id.task_managers.spatial_saxs_sampling import (
    SpatialSAXSAdaptiveSamplingTaskManager,
)
from eaa_aps12id.tools.aps12id_saxs import APS12IDSAXSAcquisitionTool


class FakeMCPTool:
    def __init__(self, remote_name, function):
        self.calls = []

        def record_call(**kwargs):
            self.calls.append(kwargs)
            return function(**kwargs)

        self.exposed_tools = [ExposedToolSpec(name=remote_name, function=record_call)]


def write_reduced_hdf5(
    path: Path,
    q=(0.01, 0.1, 1.0),
    intensity=(10.0, 20.0, 30.0),
) -> None:
    with h5py.File(path, "w") as h5_file:
        result = h5_file.create_group("entry").create_group("result")
        result.create_dataset("q", data=q)
        result.create_dataset("I", data=intensity)


def make_payload(path: Path) -> dict:
    return {
        "data_file": {
            "name": "saxs_spectrum",
            "type": "hdf5",
            "path": str(path),
            "hdf5_dataset_path": "/entry/result/I",
            "q_hdf5_dataset_path": "/entry/result/q",
        }
    }


def make_tool(path: Path, *, payload=None, remote_name="acquire_saxs_data_file"):
    result = make_payload(path) if payload is None else payload
    remote = FakeMCPTool(remote_name, lambda **kwargs: result)
    return APS12IDSAXSAcquisitionTool(remote), remote


def test_acquire_saxs_calls_remote_rpc_and_loads_hdf5(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, remote = make_tool(path)

    q, intensity = tool.acquire_saxs(
        x=1.25,
        y=-0.5,
        q_min=0.01,
        q_max=1.0,
        exposure=0.25,
    )

    assert remote.calls == [{"x": 1.25, "y": -0.5, "exposure": 0.25}]
    np.testing.assert_allclose(q, [0.01, 0.1, 1.0])
    np.testing.assert_allclose(intensity, [10.0, 20.0, 30.0])


def test_acquire_saxs_uses_server_default_exposure(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, remote = make_tool(path)

    tool.acquire_saxs(x=1.0, y=2.0)

    assert remote.calls == [{"x": 1.0, "y": 2.0}]


def test_remote_tool_name_can_be_resolved_by_suffix(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, remote = make_tool(path, remote_name="aps12.acquire_saxs_data_file")

    tool.acquire_saxs(x=0.0, y=0.0)

    assert remote.calls == [{"x": 0.0, "y": 0.0}]


def test_acquire_saxs_is_model_hidden(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, _remote = make_tool(path)

    visible = {spec.name for spec in tool.exposed_tools if spec.model_visible}
    hidden = {spec.name for spec in tool.exposed_tools if not spec.model_visible}

    assert visible == set()
    assert "aps12id_saxs.acquire_saxs" in hidden


def test_json_string_payload_is_supported(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, _remote = make_tool(path, payload=json.dumps(make_payload(path)))

    q, intensity = tool.acquire_saxs(0.0, 0.0)

    np.testing.assert_allclose(q, [0.01, 0.1, 1.0])
    np.testing.assert_allclose(intensity, [10.0, 20.0, 30.0])


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "data_file"),
        ({"data_file": {}}, "'path'"),
        (
            {
                "data_file": {
                    "type": "text",
                    "path": "data.h5",
                    "hdf5_dataset_path": "/entry/result/I",
                    "q_hdf5_dataset_path": "/entry/result/q",
                }
            },
            "type 'hdf5'",
        ),
    ],
)
def test_invalid_remote_payload_is_rejected(tmp_path, payload, message):
    tool, _remote = make_tool(tmp_path / "unused.h5", payload=payload)

    with pytest.raises(ValueError, match=message):
        tool.acquire_saxs(0.0, 0.0)


def test_missing_hdf5_dataset_is_rejected(tmp_path):
    path = tmp_path / "reduced.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("/entry/result/q", data=[0.01, 0.1])
    tool, _remote = make_tool(path)

    with pytest.raises(ValueError, match="/entry/result/I"):
        tool.acquire_saxs(0.0, 0.0)


def test_mismatched_array_shapes_are_rejected(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path, q=(0.01, 0.1), intensity=(10.0, 20.0, 30.0))
    tool, _remote = make_tool(path)

    with pytest.raises(ValueError, match="matching shapes"):
        tool.acquire_saxs(0.0, 0.0)


def test_q_coverage_is_checked_after_remote_acquisition(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, remote = make_tool(path)

    with pytest.raises(ValueError, match="q_min"):
        tool.acquire_saxs(0.0, 0.0, q_min=0.001)
    with pytest.raises(ValueError, match="q_max"):
        tool.acquire_saxs(0.0, 0.0, q_max=2.0)

    assert len(remote.calls) == 2


def test_invalid_q_range_is_rejected_before_remote_acquisition(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, remote = make_tool(path)

    with pytest.raises(ValueError, match="q_min < q_max"):
        tool.acquire_saxs(0.0, 0.0, q_min=1.0, q_max=0.1)

    assert remote.calls == []


def test_tool_is_accepted_by_spatial_sampling_task_manager(tmp_path):
    path = tmp_path / "reduced.h5"
    write_reduced_hdf5(path)
    tool, _remote = make_tool(path)

    manager = SpatialSAXSAdaptiveSamplingTaskManager(
        acquisition_tool=tool,
        checkpoint_db_path=None,
        build=False,
    )

    assert manager.acquisition_tool is tool
