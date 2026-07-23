from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np

from eaa_core.tool.base import BaseTool, check, tool
from eaa_core.tool.mcp_adapter import MCPRPCWrapper
from eaa_core.tool.mcp_client import MCPTool


class APS12IDSAXSAcquisitionTool(BaseTool):
    """Load SAXS arrays returned by the APS12 SAXSDaq MCP server."""

    name: str = "aps12id_saxs"

    @check
    def __init__(
        self,
        mcp_tool_client: MCPTool,
        remote_tool_name: str = "acquire_saxs_data_file",
        *args,
        **kwargs,
    ) -> None:
        """Initialize the logic-driven SAXS acquisition adapter.

        Parameters
        ----------
        mcp_tool_client : MCPTool
            Connected EAA client for the APS12 SAXSDaq MCP server.
        remote_tool_name : str, optional
            Exact or uniquely suffixed remote acquisition tool name.
        """
        if not remote_tool_name:
            raise ValueError("`remote_tool_name` must be non-empty.")
        self.mcp_tool_client = mcp_tool_client
        self.remote_tool_name = remote_tool_name
        self._rpc = MCPRPCWrapper(
            mcp_tool_client=mcp_tool_client,
            mappings={
                "acquire_saxs_data_file": {
                    "remote": remote_tool_name,
                }
            },
        )
        super().__init__(*args, **kwargs)

    @tool(name="aps12id_saxs.acquire_saxs", model_visible=False)
    def acquire_saxs(
        self,
        x: Annotated[float, "Spatial x coordinate."],
        y: Annotated[float, "Spatial y coordinate."],
        q_min: Annotated[float | None, "Required minimum q coverage."] = None,
        q_max: Annotated[float | None, "Required maximum q coverage."] = None,
        exposure: Annotated[float | None, "Exposure time in seconds."] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Acquire and return q and intensity arrays.

        Parameters
        ----------
        x, y : float
            Spatial positions forwarded to the remote acquisition tool.
        q_min, q_max : float, optional
            Required coverage of the returned q axis. These values are checked
            locally and are not forwarded to the acquisition server.
        exposure : float, optional
            Exposure time in seconds. When omitted, the server uses its
            configured default.

        Returns
        -------
        tuple of numpy.ndarray
            Native reduced q values and measured intensities.
        """
        q_min = self._optional_finite_float(q_min, "q_min")
        q_max = self._optional_finite_float(q_max, "q_max")
        if q_min is not None and q_max is not None and q_min >= q_max:
            raise ValueError("Expected `q_min < q_max`.")

        arguments: dict[str, Any] = {"x": x, "y": y}
        if exposure is not None:
            arguments["exposure"] = exposure
        payload = self._rpc.acquire_saxs_data_file(**arguments)
        data_file = self._parse_data_file_payload(payload)
        q, intensity = self._load_hdf5_arrays(data_file)

        if q_min is not None and np.min(q) > q_min:
            raise ValueError(
                f"Reduced spectrum does not cover requested q_min={q_min}."
            )
        if q_max is not None and np.max(q) < q_max:
            raise ValueError(
                f"Reduced spectrum does not cover requested q_max={q_max}."
            )
        return q, intensity

    @staticmethod
    def _parse_data_file_payload(payload: Any) -> dict[str, str]:
        """Validate and return the remote ``data_file`` payload."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError("Remote SAXS result was not valid JSON.") from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("data_file"), dict
        ):
            raise ValueError("Remote SAXS result must contain a `data_file` object.")

        data_file = payload["data_file"]
        required_fields = (
            "path",
            "hdf5_dataset_path",
            "q_hdf5_dataset_path",
        )
        for field in required_fields:
            if not isinstance(data_file.get(field), str) or not data_file[field]:
                raise ValueError(
                    f"Remote SAXS `data_file` must contain a non-empty {field!r}."
                )
        if data_file.get("type") not in (None, "hdf5"):
            raise ValueError("Remote SAXS `data_file` must have type 'hdf5'.")
        return data_file

    @staticmethod
    def _load_hdf5_arrays(
        data_file: dict[str, str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load and validate q and intensity arrays from a remote result file."""
        path = Path(data_file["path"])
        intensity_path = data_file["hdf5_dataset_path"]
        q_path = data_file["q_hdf5_dataset_path"]
        with h5py.File(path, "r") as h5_file:
            for dataset_path in (intensity_path, q_path):
                if dataset_path not in h5_file:
                    raise ValueError(
                        f"Reduced HDF5 file has no dataset {dataset_path!r}."
                    )
                if not isinstance(h5_file[dataset_path], h5py.Dataset):
                    raise ValueError(f"HDF5 path {dataset_path!r} is not a dataset.")
            q = np.asarray(h5_file[q_path], dtype=float).reshape(-1)
            intensity = np.asarray(h5_file[intensity_path], dtype=float).reshape(-1)

        if q.shape != intensity.shape:
            raise ValueError(
                "Reduced q and intensity arrays must have matching shapes."
            )
        if q.size < 2:
            raise ValueError("Reduced SAXS data must contain at least two q points.")
        if not np.all(np.isfinite(q)):
            raise ValueError("Reduced SAXS q values must be finite.")
        return q, intensity

    @staticmethod
    def _optional_finite_float(value: float | None, name: str) -> float | None:
        """Return an optional finite float."""
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"`{name}` must be finite.")
        return value
