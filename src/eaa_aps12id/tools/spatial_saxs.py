from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator

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
        self.measured_positions: np.ndarray | None = None
        self.q_values: np.ndarray | None = None
        self.saxs_data: np.ndarray | None = None
        self.interpolator: LinearNDInterpolator | None = None
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
        self._build_interpolator(spectra_by_id, x_positions, y_positions)

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

    def _build_interpolator(
        self,
        spectra_by_id: dict[int, np.ndarray],
        x_positions: np.ndarray,
        y_positions: np.ndarray,
    ) -> None:
        if len(spectra_by_id) > x_positions.size:
            raise ValueError(
                f"Found {len(spectra_by_id)} spectra but metadata has only "
                f"{x_positions.size} positions."
            )

        sorted_spectra = sorted(spectra_by_id.items())
        if len(sorted_spectra) < x_positions.size:
            positioned_spectra = [
                (row_index, spectrum)
                for row_index, (_, spectrum) in enumerate(sorted_spectra)
            ]
        else:
            positioned_spectra = [
                (spectrum_id - 1, spectrum)
                for spectrum_id, spectrum in sorted_spectra
            ]

        first_q: np.ndarray | None = None
        records: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for row_index, spectrum in positioned_spectra:
            if row_index >= x_positions.size:
                raise ValueError(
                    f"Spectrum maps to row {row_index}, but metadata has only "
                    f"{x_positions.size} rows."
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
        self.measured_positions = np.array(
            [(record[0], record[1]) for record in records], dtype=float
        )
        if np.unique(self.measured_positions, axis=0).shape[0] != len(records):
            raise ValueError("SAXS positions must not contain duplicates.")

        self.q_values = first_q.copy()
        q = np.stack([record[2] for record in records])
        intensity = np.stack([record[3] for record in records])
        self.saxs_data = np.stack((q, intensity), axis=-1)
        self.interpolator = LinearNDInterpolator(
            self.measured_positions[:, [1, 0]],
            intensity,
            fill_value=np.nan,
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
