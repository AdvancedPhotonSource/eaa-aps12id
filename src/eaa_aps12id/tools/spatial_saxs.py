from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from eaa_core.tool.base import BaseTool, check, tool
from eaa_aps12id.data_parser import SAXSDataParser


class SimulatedSpatialSAXS(BaseTool):
    """Simulated spatially resolved SAXS acquisition tool."""

    name: str = "simulated_spatial_saxs"

    _FILENAME_PATTERN = re.compile(
        r"^(?P<sample_prefix>.+)_(?P<scan_id>\d{5})_(?P<spectrum_id>\d{5})\.dat$"
    )

    @check
    def __init__(
        self,
        data_root: str | PathLike[str],
        file_name_pattern: str,
        *args,
        require_approval: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the simulated spatial SAXS tool.

        Parameters
        ----------
        data_root : str or PathLike
            Root directory containing ``SAXS/Averaged`` and ``Metadata``.
        file_name_pattern : str
            Glob pattern for averaged ``.dat`` files.
        require_approval : bool, optional
            Whether tool execution requires approval.
        """
        self.data_root = Path(data_root)
        self.file_name_pattern = file_name_pattern
        self.dat_files: list[Path] = []
        self.scan_identifier: str | None = None
        self.x_values: np.ndarray | None = None
        self.y_values: np.ndarray | None = None
        self.q_values: np.ndarray | None = None
        self.saxs_data: np.ndarray | None = None
        self.intensity_grid: np.ndarray | None = None
        self.interpolator: RegularGridInterpolator | None = None
        self.parser = SAXSDataParser()
        super().__init__(*args, require_approval=require_approval, **kwargs)

    def build(self) -> None:
        """Load SAXS spectra and build the spatial interpolation backend."""
        averaged_dir = self.data_root / "SAXS" / "Averaged"
        self.dat_files = sorted(averaged_dir.glob(self.file_name_pattern))
        if not self.dat_files:
            raise FileNotFoundError(
                f"No SAXS .dat files matched {self.file_name_pattern!r} in {averaged_dir}."
            )

        parsed_files: list[tuple[str, str, int, Path]] = []
        for dat_file in self.dat_files:
            metadata_identifier, scan_identifier, spectrum_id = self._parse_dat_filename(
                dat_file
            )
            if spectrum_id < 1:
                raise ValueError("Spectrum IDs are one-based; `00000` is invalid.")
            parsed_files.append(
                (metadata_identifier, scan_identifier, spectrum_id, dat_file)
            )

        scan_ids = {scan_identifier for _, scan_identifier, _, _ in parsed_files}
        if len(scan_ids) != 1:
            raise ValueError(
                "All SAXS files must belong to one scan identifier, got "
                f"{sorted(scan_ids)}."
            )
        self.scan_identifier = next(iter(scan_ids))

        metadata_ids = {
            metadata_identifier for metadata_identifier, _, _, _ in parsed_files
        }
        if len(metadata_ids) != 1:
            raise ValueError(
                "All SAXS files must belong to one metadata identifier, got "
                f"{sorted(metadata_ids)}."
            )
        metadata_identifier = next(iter(metadata_ids))

        spectra_by_id: dict[int, np.ndarray] = {}
        for _, _, spectrum_id, dat_file in parsed_files:
            if spectrum_id in spectra_by_id:
                raise ValueError(f"Duplicate spectrum ID {spectrum_id:05d}.")
            spectra_by_id[spectrum_id] = self.parser.parse(dat_file)

        x_positions, y_positions = self._load_positions(metadata_identifier)
        self._build_grid(spectra_by_id, x_positions, y_positions)

    def _parse_dat_filename(self, dat_file: Path) -> tuple[str, str, int]:
        match = self._FILENAME_PATTERN.match(dat_file.name)
        if match is None:
            raise ValueError(
                "Expected SAXS filename shaped like "
                "`Sample_Name_<5-digit-scan-id>_<5-digit-spectrum-id>.dat`, "
                f"got {dat_file.name!r}."
            )
        sample_prefix = match.group("sample_prefix")
        if not sample_prefix.startswith("S"):
            raise ValueError(
                f"Expected sample prefix to start with `S`, got {sample_prefix!r}."
            )
        metadata_identifier = sample_prefix[1:]
        scan_identifier = f"{metadata_identifier}_{match.group('scan_id')}"
        return metadata_identifier, scan_identifier, int(match.group("spectrum_id"))

    def _load_positions(self, scan_identifier: str) -> tuple[np.ndarray, np.ndarray]:
        metadata_path = self.data_root / "Metadata" / f"{scan_identifier}.h5"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

        with h5py.File(metadata_path, "r") as h5_file:
            measurement = h5_file["entry"]["measurement"]
            x_positions = np.asarray(measurement["motor_sth"], dtype=float).reshape(-1)
            y_positions = np.asarray(measurement["motor_sav"], dtype=float).reshape(-1)

        if x_positions.shape != y_positions.shape:
            raise ValueError("`motor_sth` and `motor_sav` must have matching lengths.")
        return x_positions, y_positions

    def _build_grid(
        self,
        spectra_by_id: dict[int, np.ndarray],
        x_positions: np.ndarray,
        y_positions: np.ndarray,
    ) -> None:
        first_q: np.ndarray | None = None
        records: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for spectrum_id, spectrum in sorted(spectra_by_id.items()):
            row_index = spectrum_id - 1
            if row_index >= x_positions.size:
                raise ValueError(
                    f"Spectrum ID {spectrum_id:05d} maps to row {row_index}, "
                    f"but metadata has only {x_positions.size} rows."
                )
            q = np.asarray(spectrum[:, 0], dtype=float)
            intensity = np.asarray(spectrum[:, 1], dtype=float)
            if first_q is None:
                first_q = q
            elif q.shape != first_q.shape or not np.allclose(q, first_q):
                raise ValueError("All SAXS spectra must share the same q grid.")
            records.append(
                (
                    float(x_positions[row_index]),
                    float(y_positions[row_index]),
                    q,
                    intensity,
                )
            )

        if first_q is None:
            raise ValueError("At least one SAXS spectrum is required.")
        if first_q.size < 2:
            raise ValueError("A SAXS spectrum must contain at least two q points.")
        if np.any(np.diff(first_q) <= 0):
            raise ValueError("SAXS q values must be strictly increasing.")

        self.x_values = np.array(sorted({record[0] for record in records}), dtype=float)
        self.y_values = np.array(sorted({record[1] for record in records}), dtype=float)
        expected_count = self.x_values.size * self.y_values.size
        if len(records) != expected_count:
            raise ValueError(
                "SAXS positions must form a complete rectangular grid; "
                f"expected {expected_count} spectra, got {len(records)}."
            )

        x_index = {value: idx for idx, value in enumerate(self.x_values)}
        y_index = {value: idx for idx, value in enumerate(self.y_values)}
        n_q = first_q.size
        self.saxs_data = np.empty((self.y_values.size, self.x_values.size, n_q, 2))
        self.intensity_grid = np.empty((self.y_values.size, self.x_values.size, n_q))
        populated: set[tuple[int, int]] = set()

        for x, y, q, intensity in records:
            grid_index = (y_index[y], x_index[x])
            if grid_index in populated:
                raise ValueError(f"Duplicate SAXS position at x={x}, y={y}.")
            populated.add(grid_index)
            self.saxs_data[grid_index[0], grid_index[1], :, 0] = q
            self.saxs_data[grid_index[0], grid_index[1], :, 1] = intensity
            self.intensity_grid[grid_index[0], grid_index[1], :] = intensity

        if len(populated) != expected_count:
            raise ValueError("SAXS positions must cover every rectangular grid point.")

        self.q_values = first_q.copy()
        self.interpolator = RegularGridInterpolator(
            (self.y_values, self.x_values),
            self.intensity_grid,
            bounds_error=True,
        )

    @tool(name="simulated_spatial_saxs.acquire_saxs")
    def acquire_saxs(
        self,
        x: Annotated[float, "Spatial x coordinate."],
        y: Annotated[float, "Spatial y coordinate."],
        q_min: Annotated[float, "Minimum q value for the output grid."],
        q_max: Annotated[float, "Maximum q value for the output grid."],
        q_step: Annotated[float, "Constant q interval, as used by numpy.arange."],
    ) -> Annotated[tuple[np.ndarray, np.ndarray], "Tuple of q values and intensities."]:
        """Acquire a simulated SAXS spectrum at a spatial position.

        Parameters
        ----------
        x : float
            Spatial x coordinate.
        y : float
            Spatial y coordinate.
        q_min : float
            Minimum q value for the output grid.
        q_max : float
            Maximum q value for the output grid.
        q_step : float
            Constant q interval. The output grid is ``np.arange(q_min, q_max,
            q_step)``.

        Returns
        -------
        tuple of numpy.ndarray
            Output q values and interpolated intensities.
        """
        if self.interpolator is None or self.q_values is None:
            raise RuntimeError("SimulatedSpatialSAXS has not been built.")
        q_min = float(q_min)
        q_max = float(q_max)
        q_step = float(q_step)
        if not (q_min < q_max):
            raise ValueError("Expected `q_min < q_max`.")
        if q_step <= 0:
            raise ValueError("`q_step` must be positive.")

        q_out = np.arange(q_min, q_max, q_step, dtype=float)
        if q_out.size == 0:
            raise ValueError("Requested q grid is empty.")
        if q_out[0] < self.q_values[0] or q_out[-1] > self.q_values[-1]:
            raise ValueError("Requested q grid is outside the native SAXS q range.")

        intensity = np.asarray(self.interpolator([[float(y), float(x)]])[0], dtype=float)
        intensity_out = np.interp(q_out, self.q_values, intensity)
        return q_out, intensity_out
