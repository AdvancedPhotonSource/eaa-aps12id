from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import networkx as nx
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from eaa_core.tool.base import BaseTool, tool
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class SAXSMeasurement:
    """Recorded SAXS measurement at one spatial position."""

    position: np.ndarray
    q: np.ndarray
    intensity: np.ndarray
    spectrum: np.ndarray
    background: np.ndarray


@dataclass
class SAXSPeak:
    """Peak definition shared by all measured SAXS spectra."""

    peak_id: int
    q_position: float
    log_q_position: float
    q_left: float
    q_right: float
    width_log_q: float
    max_integrated_area: float
    discovery_measurement_index: int


@dataclass
class DetectedSAXSPeak:
    """Peak detected in one background-subtracted SAXS spectrum."""

    q_position: float
    log_q_position: float
    q_left: float
    q_right: float
    width_log_q: float
    height: float
    prominence: float
    integrated_area: float


@dataclass
class AcquisitionScores:
    """Acquisition terms for unsampled candidate positions."""

    positions: np.ndarray
    acquisition: np.ndarray
    sigma_tilde: np.ndarray
    peak_area_tilde: np.ndarray
    gradient_tilde: np.ndarray

    @property
    def peak_observable_tilde(self) -> np.ndarray:
        """Return the normalized peak observable."""
        return self.peak_area_tilde


class SpatialSAXSAdaptiveSamplingEngineTool(BaseTool):
    """Bayesian active-learning engine for spatially resolved SAXS.

    The engine models spectra measured at finite spatial candidates, preprocesses each
    ``(q, I)`` SAXS spectrum onto a common log-spaced q grid, subtracts a
    smooth background, and uses independent GP models for detected peak areas
    or heights to choose additional measurement positions.
    """

    name: str = "spatial_saxs_adaptive_sampling_engine"

    def __init__(self, *args, require_approval: bool = False, **kwargs) -> None:
        """Initialize an unconfigured spatial SAXS active-learning engine.

        Parameters
        ----------
        require_approval : bool, optional
            Whether exposed engine operations require approval.
        """
        if torch is None:
            raise ImportError(
                "SpatialSAXSAdaptiveSamplingEngineTool requires torch, botorch, "
                "and gpytorch. Install this package with its declared dependencies."
            )
        self._sampling_config: dict[str, Any] | None = None
        self.candidate_positions: np.ndarray | None = None
        self.q_min: float | None = None
        self.q_max: float | None = None
        self.num_q_points: int | None = None
        self.epsilon_intensity: float | None = None
        self.num_initial_samples: int | None = None
        self.exclusion_radius: float | None = None
        self.suggestion_exclusion_radius: float | None = None
        self.background_smoothness: float | None = None
        self.background_max_iterations: int | None = None
        self.background_tolerance: float | None = None
        self.background_valley_smoothing_sigma: float | None = None
        self.background_valley_min_prominence: float | None = None
        self.peak_smoothing_sigma: float | None = None
        self.peak_min_height: float | None = None
        self.peak_min_prominence: float | None = None
        self.peak_min_width_log_q: float | None = None
        self.peak_max_width_log_q: float | None = None
        self.peak_window_width_factor: float | None = None
        self.num_initial_peaks: int | None = None
        self.max_peaks_in_dict: int | None = None
        self.known_peak_q_values: np.ndarray | None = None
        self.new_peak_min_relative_area: float | None = None
        self.peak_map_min_concentration: float | None = None
        self.peak_observable: str | None = None
        self.peak_observale_map_blur: float | None = None
        self.peak_area_scale: float | None = None
        self.exploration_interval: int | None = None
        self.max_fit_gp_mll_iterations: int | None = None
        self.w_peak: float | None = None
        self.w_g: float | None = None
        self.epsilon_acquisition: float | None = None
        self.epsilon_normalization: float | None = None
        self.epsilon_z: float | None = None
        self.normalization_lower_percentile: float | None = None
        self.normalization_upper_percentile: float | None = None
        self.random_seed: int | None = None
        self.q_grid: np.ndarray | None = None
        self.position_min: np.ndarray | None = None
        self.position_max: np.ndarray | None = None
        self.position_span: np.ndarray | None = None
        self.candidate_positions_normalized: np.ndarray | None = None
        self.measurements: list[SAXSMeasurement] = []
        self.measured_candidate_indices: list[int] = []
        self.excluded_measurement_indices: set[int] = set()
        self.peak_dict: dict[int, SAXSPeak] = {}
        self._next_peak_id = 0
        self._peak_detection_measurement_count = 0
        self.standardized_peak_scores: torch.Tensor | None = None
        self.peak_score_mean: torch.Tensor | None = None
        self.peak_score_std: torch.Tensor | None = None
        self.modeled_peak_ids: list[int] = []
        self.gp_model = None
        self.latest_scores: AcquisitionScores | None = None
        self._peak_blur_indices: np.ndarray | None = None
        self._peak_blur_weights: np.ndarray | None = None
        self._peak_blur_weight_matrix: torch.Tensor | None = None
        super().__init__(
            *args,
            build=False,
            require_approval=require_approval,
            **kwargs,
        )

    @staticmethod
    def _validate_candidate_positions(values: Any) -> np.ndarray:
        if values is None:
            raise ValueError("`candidate_positions` must be provided.")
        positions = np.asarray(values, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2 or not positions.shape[0]:
            raise ValueError(
                "`candidate_positions` must be a non-empty array with shape (N, 2)."
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("`candidate_positions` must contain only finite values.")
        if np.unique(positions, axis=0).shape[0] != positions.shape[0]:
            raise ValueError("`candidate_positions` must not contain duplicates.")
        return positions

    @tool(name="spatial_saxs_adaptive_sampling_engine.initialize")
    def initialize(
        self,
        candidate_positions: (
            np.ndarray
            | list[list[float]]
            | list[tuple[float, float]]
            | tuple[tuple[float, float], ...]
        ),
        q_min: float = 0.001,
        q_max: float = 1.0,
        num_q_points: int = 256,
        epsilon_intensity: float = 1e-12,
        num_initial_samples: int = 5,
        exclusion_radius: float | None = None,
        suggestion_exclusion_radius: float | None = None,
        background_smoothness: float = 1e6,
        background_max_iterations: int = 50,
        background_tolerance: float = 1e-3,
        background_valley_smoothing_sigma: float = 3.0,
        background_valley_min_prominence: float = 0.05,
        peak_smoothing_sigma: float = 1.0,
        peak_min_height: float = 1.0,
        peak_min_prominence: float = 1.0,
        peak_min_width_log_q: float = 0.03,
        peak_max_width_log_q: float | None = None,
        peak_window_width_factor: float = 2.0,
        num_initial_peaks: int = 5,
        max_peaks_in_dict: int = 10,
        known_peak_q_values: (
            np.ndarray | list[float] | tuple[float, ...] | None
        ) = None,
        new_peak_min_relative_area: float = 0.001,
        peak_map_min_concentration: float = 0.2,
        peak_observable: Literal["height", "area"] = "area",
        peak_observale_map_blur: float | None = None,
        peak_area_scale: float = 1.0,
        exploration_interval: int | None = 5,
        max_fit_gp_mll_iterations: int | None = None,
        w_peak: float = 1.0,
        w_g: float = 1.0,
        epsilon_acquisition: float = 1e-3,
        epsilon_normalization: float = 1e-12,
        epsilon_z: float = 1e-12,
        normalization_lower_percentile: float = 5.0,
        normalization_upper_percentile: float = 95.0,
        random_seed: int | None = None,
    ) -> None:
        """Configure the active-learning engine for one sampling run.

        Parameters
        ----------
        candidate_positions : array-like
            Candidate spatial coordinates with shape ``(N, 2)``. Columns are
            ordered ``(y, x)`` and rows retain the supplied order.
        """
        candidates = self._validate_candidate_positions(candidate_positions)
        known_peaks = (
            None
            if known_peak_q_values is None
            else np.asarray(known_peak_q_values, dtype=float).reshape(-1)
        )
        config = {
            "candidate_positions": tuple(map(tuple, candidates.tolist())),
            "q_min": float(q_min),
            "q_max": float(q_max),
            "num_q_points": int(num_q_points),
            "epsilon_intensity": float(epsilon_intensity),
            "num_initial_samples": int(num_initial_samples),
            "exclusion_radius": (
                None if exclusion_radius is None else float(exclusion_radius)
            ),
            "suggestion_exclusion_radius": (
                None
                if suggestion_exclusion_radius is None
                else float(suggestion_exclusion_radius)
            ),
            "background_smoothness": float(background_smoothness),
            "background_max_iterations": int(background_max_iterations),
            "background_tolerance": float(background_tolerance),
            "background_valley_smoothing_sigma": float(
                background_valley_smoothing_sigma
            ),
            "background_valley_min_prominence": float(background_valley_min_prominence),
            "peak_smoothing_sigma": float(peak_smoothing_sigma),
            "peak_min_height": float(peak_min_height),
            "peak_min_prominence": float(peak_min_prominence),
            "peak_min_width_log_q": float(peak_min_width_log_q),
            "peak_max_width_log_q": (
                None if peak_max_width_log_q is None else float(peak_max_width_log_q)
            ),
            "peak_window_width_factor": float(peak_window_width_factor),
            "num_initial_peaks": int(num_initial_peaks),
            "max_peaks_in_dict": int(max_peaks_in_dict),
            "known_peak_q_values": (
                None if known_peaks is None else tuple(known_peaks.tolist())
            ),
            "new_peak_min_relative_area": float(new_peak_min_relative_area),
            "peak_map_min_concentration": float(peak_map_min_concentration),
            "peak_observable": str(peak_observable),
            "peak_observale_map_blur": (
                None
                if peak_observale_map_blur is None
                else float(peak_observale_map_blur)
            ),
            "peak_area_scale": float(peak_area_scale),
            "exploration_interval": (
                None if exploration_interval is None else int(exploration_interval)
            ),
            "max_fit_gp_mll_iterations": (
                None
                if max_fit_gp_mll_iterations is None
                else int(max_fit_gp_mll_iterations)
            ),
            "w_peak": float(w_peak),
            "w_g": float(w_g),
            "epsilon_acquisition": float(epsilon_acquisition),
            "epsilon_normalization": float(epsilon_normalization),
            "epsilon_z": float(epsilon_z),
            "normalization_lower_percentile": float(normalization_lower_percentile),
            "normalization_upper_percentile": float(normalization_upper_percentile),
            "random_seed": random_seed,
        }
        if self.measurements and self._sampling_config != config:
            raise ValueError(
                "Cannot change adaptive sampling parameters after measurements "
                "have been added. Create a new engine for a new run."
            )
        if self._sampling_config == config:
            return

        self._sampling_config = config
        self.candidate_positions = candidates.copy()
        self.q_min = config["q_min"]
        self.q_max = config["q_max"]
        self.num_q_points = config["num_q_points"]
        self.epsilon_intensity = config["epsilon_intensity"]
        self.num_initial_samples = config["num_initial_samples"]
        self.exclusion_radius = config["exclusion_radius"]
        self.suggestion_exclusion_radius = config["suggestion_exclusion_radius"]
        self.background_smoothness = config["background_smoothness"]
        self.background_max_iterations = config["background_max_iterations"]
        self.background_tolerance = config["background_tolerance"]
        self.background_valley_smoothing_sigma = config[
            "background_valley_smoothing_sigma"
        ]
        self.background_valley_min_prominence = config[
            "background_valley_min_prominence"
        ]
        self.peak_smoothing_sigma = config["peak_smoothing_sigma"]
        self.peak_min_height = config["peak_min_height"]
        self.peak_min_prominence = config["peak_min_prominence"]
        self.peak_min_width_log_q = config["peak_min_width_log_q"]
        self.peak_max_width_log_q = config["peak_max_width_log_q"]
        self.peak_window_width_factor = config["peak_window_width_factor"]
        self.num_initial_peaks = config["num_initial_peaks"]
        self.max_peaks_in_dict = config["max_peaks_in_dict"]
        self.known_peak_q_values = None if known_peaks is None else known_peaks.copy()
        self.new_peak_min_relative_area = config["new_peak_min_relative_area"]
        self.peak_map_min_concentration = config["peak_map_min_concentration"]
        self.peak_observable = config["peak_observable"]
        self.peak_observale_map_blur = config["peak_observale_map_blur"]
        self.peak_area_scale = config["peak_area_scale"]
        self.exploration_interval = config["exploration_interval"]
        self.max_fit_gp_mll_iterations = config["max_fit_gp_mll_iterations"]
        self.w_peak = config["w_peak"]
        self.w_g = config["w_g"]
        self.epsilon_acquisition = config["epsilon_acquisition"]
        self.epsilon_normalization = config["epsilon_normalization"]
        self.epsilon_z = config["epsilon_z"]
        self.normalization_lower_percentile = config["normalization_lower_percentile"]
        self.normalization_upper_percentile = config["normalization_upper_percentile"]
        self.random_seed = random_seed
        self._validate_parameters()

        self.q_grid = self.create_log_q_grid()
        self.position_min = self.candidate_positions.min(axis=0)
        self.position_max = self.candidate_positions.max(axis=0)
        self.position_span = self.position_max - self.position_min
        self.position_span[self.position_span == 0] = 1.0
        self.candidate_positions_normalized = self.normalize_positions(
            self.candidate_positions
        )
        self._build_peak_blur_weights()

    def _require_sampling_configured(self) -> None:
        """Raise if adaptive sampling parameters have not been configured."""
        if self._sampling_config is None:
            raise ValueError(
                "Adaptive sampling parameters must be provided to `run` first."
            )

    def _validate_parameters(self) -> None:
        self._require_sampling_configured()
        if not (0 < self.q_min < self.q_max):
            raise ValueError("Expected `0 < q_min < q_max`.")
        if self.num_q_points < 2:
            raise ValueError("`num_q_points` must be at least 2.")
        if self.epsilon_intensity <= 0:
            raise ValueError("`epsilon_intensity` must be positive.")
        if not (1 <= self.num_initial_samples <= self.candidate_positions.shape[0]):
            raise ValueError(
                "`num_initial_samples` cannot exceed the number of candidate positions."
            )
        if self.exclusion_radius is not None and (
            not np.isfinite(self.exclusion_radius) or self.exclusion_radius < 0
        ):
            raise ValueError(
                "`exclusion_radius` must be finite and nonnegative or None."
            )
        if self.suggestion_exclusion_radius is not None and (
            not np.isfinite(self.suggestion_exclusion_radius)
            or self.suggestion_exclusion_radius < 0
        ):
            raise ValueError(
                "`suggestion_exclusion_radius` must be finite and nonnegative or None."
            )
        if self.background_smoothness <= 0:
            raise ValueError("`background_smoothness` must be positive.")
        if self.background_max_iterations < 1:
            raise ValueError("`background_max_iterations` must be positive.")
        if self.background_tolerance <= 0:
            raise ValueError("`background_tolerance` must be positive.")
        if self.background_valley_smoothing_sigma <= 0:
            raise ValueError("`background_valley_smoothing_sigma` must be positive.")
        if self.background_valley_min_prominence < 0:
            raise ValueError("`background_valley_min_prominence` must be nonnegative.")
        if self.peak_smoothing_sigma < 0:
            raise ValueError("`peak_smoothing_sigma` must be nonnegative.")
        if self.peak_min_height < 0 or self.peak_min_prominence < 0:
            raise ValueError(
                "`peak_min_height` and `peak_min_prominence` must be nonnegative."
            )
        if self.peak_min_width_log_q < 0:
            raise ValueError("`peak_min_width_log_q` must be nonnegative.")
        if (
            self.peak_max_width_log_q is not None
            and self.peak_max_width_log_q <= self.peak_min_width_log_q
        ):
            raise ValueError(
                "`peak_max_width_log_q` must exceed `peak_min_width_log_q`."
            )
        if self.peak_window_width_factor <= 0:
            raise ValueError("`peak_window_width_factor` must be positive.")
        if self.num_initial_peaks < 1:
            raise ValueError("`num_initial_peaks` must be positive.")
        if self.max_peaks_in_dict < self.num_initial_peaks:
            raise ValueError(
                "`max_peaks_in_dict` must be at least `num_initial_peaks`."
            )
        if self.known_peak_q_values is not None:
            if self.known_peak_q_values.size == 0:
                raise ValueError(
                    "`known_peak_q_values` must contain at least one value."
                )
            if not np.all(np.isfinite(self.known_peak_q_values)) or np.any(
                self.known_peak_q_values <= 0
            ):
                raise ValueError(
                    "`known_peak_q_values` must contain finite positive values."
                )
            if np.unique(self.known_peak_q_values).size != (
                self.known_peak_q_values.size
            ):
                raise ValueError("`known_peak_q_values` must not contain duplicates.")
            if np.any(
                (self.known_peak_q_values < self.q_min)
                | (self.known_peak_q_values > self.q_max)
            ):
                raise ValueError(
                    "`known_peak_q_values` must lie within `[q_min, q_max]`."
                )
        if self.new_peak_min_relative_area < 0:
            raise ValueError("`new_peak_min_relative_area` must be nonnegative.")
        if not (0 <= self.peak_map_min_concentration <= 1):
            raise ValueError(
                "`peak_map_min_concentration` must be between zero and one."
            )
        if self.peak_observable not in {"area", "height"}:
            raise ValueError("`peak_observable` must be either 'area' or 'height'.")
        if self.peak_observale_map_blur is not None and (
            not np.isfinite(self.peak_observale_map_blur)
            or self.peak_observale_map_blur < 0
        ):
            raise ValueError(
                "`peak_observale_map_blur` must be finite and nonnegative or None."
            )
        if self.peak_area_scale <= 0:
            raise ValueError("`peak_area_scale` must be positive.")
        if self.exploration_interval is not None and self.exploration_interval < 1:
            raise ValueError("`exploration_interval` must be positive or None.")
        if (
            self.max_fit_gp_mll_iterations is not None
            and self.max_fit_gp_mll_iterations < 1
        ):
            raise ValueError("`max_fit_gp_mll_iterations` must be positive or None.")
        if self.w_peak < 0 or self.w_g < 0:
            raise ValueError("`w_peak` and `w_g` must be nonnegative.")
        if self.epsilon_acquisition <= 0:
            raise ValueError("`epsilon_acquisition` must be positive.")
        if self.epsilon_normalization <= 0 or self.epsilon_z <= 0:
            raise ValueError(
                "`epsilon_normalization` and `epsilon_z` must be positive."
            )
        if not (
            0
            <= self.normalization_lower_percentile
            < self.normalization_upper_percentile
            <= 100
        ):
            raise ValueError(
                "Expected `0 <= normalization_lower_percentile < "
                "normalization_upper_percentile <= 100`."
            )

    def create_log_q_grid(self) -> np.ndarray:
        """Return the common log-spaced q grid."""
        self._require_sampling_configured()
        return np.exp(
            np.linspace(np.log(self.q_min), np.log(self.q_max), self.num_q_points)
        )

    def normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to the full candidate-coordinate bounds."""
        self._require_sampling_configured()
        positions = np.asarray(positions, dtype=float)
        return (positions - self.position_min) / self.position_span

    @tool(name="spatial_saxs_adaptive_sampling_engine.suggest_initial_measurements")
    def suggest_initial_measurements(self) -> np.ndarray:
        """Return path-optimized Sobol positions for initial measurements.

        Returns
        -------
        numpy.ndarray
            Initial positions with shape ``(num_initial_samples, 2)``.
        """
        self._require_sampling_configured()
        if self.measurements:
            raise ValueError(
                "Initial measurements can only be suggested before updates."
            )
        candidate_indices = self.get_initial_candidate_indices()
        return self._optimize_suggestion_path(
            self.candidate_positions[candidate_indices]
        )

    def get_initial_candidate_indices(self) -> list[int]:
        """Return unique initial candidate indices selected by Sobol sampling."""
        self._require_sampling_configured()
        engine = torch.quasirandom.SobolEngine(
            dimension=2, scramble=True, seed=self.random_seed
        )
        selected: list[int] = []
        selected_set: set[int] = set()
        max_draws = max(16, self.num_initial_samples * 8)
        while (
            len(selected) < self.num_initial_samples
            and max_draws <= self.candidate_positions.shape[0] * 16
        ):
            sobol = engine.draw(max_draws).double()
            candidates = torch.as_tensor(
                self.candidate_positions_normalized,
                dtype=torch.double,
                device=sobol.device,
            )
            distances = torch.cdist(sobol, candidates)
            nearest = torch.argmin(distances, dim=1).detach().cpu().numpy()
            for idx in nearest:
                idx_int = int(idx)
                eligible = self._filter_candidate_indices_by_radius(
                    np.asarray([idx_int], dtype=int),
                    self.measured_candidate_indices,
                    self.exclusion_radius,
                )
                eligible = self._filter_candidate_indices_by_radius(
                    eligible,
                    selected,
                    self.suggestion_exclusion_radius,
                )
                if idx_int not in selected_set and eligible.size:
                    selected.append(idx_int)
                    selected_set.add(idx_int)
                    if len(selected) == self.num_initial_samples:
                        return selected
            max_draws *= 2

        for idx in range(self.candidate_positions.shape[0]):
            eligible = self._filter_candidate_indices_by_radius(
                np.asarray([idx], dtype=int),
                self.measured_candidate_indices,
                self.exclusion_radius,
            )
            eligible = self._filter_candidate_indices_by_radius(
                eligible,
                selected,
                self.suggestion_exclusion_radius,
            )
            if idx not in selected_set and eligible.size:
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == self.num_initial_samples:
                    break
        if len(selected) < self.num_initial_samples:
            raise ValueError(
                "The configured exclusion radii leave fewer than "
                "`num_initial_samples` eligible candidate positions."
            )
        return selected

    @staticmethod
    def _normalize_spectrum_batch(
        values: np.ndarray | list[np.ndarray] | list[list[float]] | list[float],
        batch_size: int,
        name: str,
    ) -> list[np.ndarray]:
        """Normalize one or more spectrum arrays to a list."""
        if batch_size == 1:
            single = np.asarray(values, dtype=float)
            if single.ndim == 1:
                return [single]
        try:
            arrays = [np.asarray(value, dtype=float) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"`{name}` must contain one one-dimensional array per position."
            ) from exc
        if len(arrays) != batch_size or any(array.ndim != 1 for array in arrays):
            raise ValueError(
                f"`{name}` must contain one one-dimensional array per position."
            )
        return arrays

    @tool(name="spatial_saxs_adaptive_sampling_engine.update")
    def update(
        self,
        positions: Annotated[
            np.ndarray | list[list[float]] | list[float],
            "Measured spatial positions.",
        ],
        q_values: Annotated[
            np.ndarray | list[np.ndarray] | list[list[float]] | list[float],
            "Measured SAXS q arrays.",
        ],
        intensities: Annotated[
            np.ndarray | list[np.ndarray] | list[list[float]] | list[float],
            "Measured SAXS intensity arrays.",
        ],
    ) -> None:
        """Update the Gaussian-process model with measured SAXS spectra.

        Parameters
        ----------
        positions : array-like
            One position with shape ``(2,)`` or a batch with shape
            ``(n_measurements, 2)``.
        q_values : array-like or sequence of array-like
            One q array per measured position.
        intensities : array-like or sequence of array-like
            One intensity array per measured position.
        """
        self._require_sampling_configured()
        positions_array = np.asarray(positions, dtype=float)
        if positions_array.shape == (2,):
            positions_array = positions_array.reshape(1, 2)
        if (
            positions_array.ndim != 2
            or positions_array.shape[1] != 2
            or positions_array.shape[0] == 0
            or not np.all(np.isfinite(positions_array))
        ):
            raise ValueError(
                "`positions` must be a non-empty finite array with shape (n, 2)."
            )

        q_batch = self._normalize_spectrum_batch(
            q_values, positions_array.shape[0], "q_values"
        )
        intensity_batch = self._normalize_spectrum_batch(
            intensities, positions_array.shape[0], "intensities"
        )
        candidate_indices = [
            self.get_candidate_index(position) for position in positions_array
        ]
        if len(set(candidate_indices)) != len(candidate_indices):
            raise ValueError("An update batch cannot contain duplicate positions.")
        measured = set(self.measured_candidate_indices)
        duplicate = next(
            (index for index in candidate_indices if index in measured),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"Candidate index {duplicate} has already been measured.")

        measurements = []
        for position, q, intensity in zip(
            positions_array,
            q_batch,
            intensity_batch,
            strict=True,
        ):
            spectrum, background = self._preprocess_spectrum_with_background(
                q, intensity
            )
            measurements.append(
                SAXSMeasurement(
                    position=position.copy(),
                    q=np.asarray(q, dtype=float).copy(),
                    intensity=np.asarray(intensity, dtype=float).copy(),
                    spectrum=spectrum,
                    background=background,
                )
            )

        self.measurements.extend(measurements)
        self.measured_candidate_indices.extend(candidate_indices)
        self.refit_model()

    def preprocess_spectrum(self, q: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Interpolate a spectrum and subtract its fitted background."""
        spectrum, _ = self._preprocess_spectrum_with_background(q, intensity)
        return spectrum

    def _preprocess_spectrum_with_background(
        self,
        q: np.ndarray,
        intensity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return background-subtracted intensity and the fitted background."""
        self._require_sampling_configured()
        q = np.asarray(q, dtype=float).reshape(-1)
        intensity = np.asarray(intensity, dtype=float).reshape(-1)
        if q.shape != intensity.shape:
            raise ValueError("`q` and intensity arrays must have the same shape.")
        if q.size < 2:
            raise ValueError("A spectrum must contain at least two q points.")
        finite_mask = np.isfinite(q) & np.isfinite(intensity)
        q = q[finite_mask]
        intensity = intensity[finite_mask]
        order = np.argsort(q)
        q = q[order]
        intensity = intensity[order]
        unique_q, unique_indices = np.unique(q, return_index=True)
        q = unique_q
        intensity = intensity[unique_indices]
        if q.size < 2 or q[0] > self.q_min or q[-1] < self.q_max:
            raise ValueError(
                "Raw spectrum does not cover the full [q_min, q_max] range."
            )
        if np.any(intensity + self.epsilon_intensity <= 0):
            raise ValueError(
                "Intensity values must be greater than `-epsilon_intensity`."
            )
        interpolated = np.interp(self.q_grid, q, intensity)
        log_intensity = np.log(interpolated + self.epsilon_intensity)
        log_background, _ = self.fit_valley_background(
            log_intensity,
            smoothing_sigma=self.background_valley_smoothing_sigma,
            min_prominence=self.background_valley_min_prominence,
        )
        if log_background is None:
            log_background = gaussian_filter1d(
                log_intensity,
                sigma=self.background_valley_smoothing_sigma,
                mode="nearest",
            )
        background = np.maximum(
            np.exp(log_background) - self.epsilon_intensity,
            0.0,
        )
        return interpolated - background, background

    @staticmethod
    def fit_valley_background(
        values: np.ndarray,
        smoothing_sigma: float,
        min_prominence: float,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Fit a shape-preserving background through spectral valleys.

        The input is assumed to be sampled uniformly in log-q. It is smoothed
        before valleys are found as peaks of its negative. When at least two
        valleys are present, PCHIP interpolates their smoothed values together
        with five-point median endpoint anchors. Otherwise, no background is
        returned so the caller can use the smoothed spectrum.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional log-intensity spectrum.
        smoothing_sigma : float
            Gaussian smoothing width in grid samples.
        min_prominence : float
            Minimum valley prominence in log-intensity.

        Returns
        -------
        numpy.ndarray or None
            Interpolated log-background, or ``None`` when fewer than two
            valleys are found.
        numpy.ndarray
            Indices of the detected valleys.
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        smoothed = gaussian_filter1d(
            values,
            sigma=smoothing_sigma,
            mode="nearest",
        )
        valley_indices, _ = find_peaks(
            -smoothed,
            prominence=min_prominence,
        )
        if valley_indices.size < 2:
            return None, valley_indices

        endpoint_points = min(5, values.size)
        anchor_indices = np.concatenate(([0], valley_indices, [values.size - 1]))
        anchor_values = np.concatenate(
            (
                [float(np.median(values[:endpoint_points]))],
                smoothed[valley_indices],
                [float(np.median(values[-endpoint_points:]))],
            )
        )
        background = PchipInterpolator(
            anchor_indices,
            anchor_values,
        )(np.arange(values.size))
        return np.asarray(background, dtype=float), valley_indices

    @staticmethod
    def fit_arpls_background(
        values: np.ndarray,
        smoothness: float,
        max_iterations: int,
        tolerance: float,
    ) -> np.ndarray:
        """Fit an asymmetrically reweighted penalized least-squares baseline.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional spectrum values on a uniformly spaced grid.
        smoothness : float
            Positive second-difference smoothness penalty.
        max_iterations : int
            Maximum number of asymmetric reweighting iterations.
        tolerance : float
            Relative convergence tolerance for the weights.

        Returns
        -------
        numpy.ndarray
            Smooth fitted baseline with the same shape as ``values``.
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size < 3:
            raise ValueError("arPLS background fitting requires at least three points.")
        difference = diags(
            [
                np.ones(values.size - 2),
                -2.0 * np.ones(values.size - 2),
                np.ones(values.size - 2),
            ],
            [0, 1, 2],
            shape=(values.size - 2, values.size),
            format="csc",
        )
        penalty = smoothness * (difference.T @ difference)
        weights = np.ones(values.size, dtype=float)
        baseline = values.copy()
        for _ in range(max_iterations):
            weight_matrix = diags(weights, offsets=0, format="csc")
            baseline = np.asarray(
                spsolve(weight_matrix + penalty, weights * values),
                dtype=float,
            )
            residual = values - baseline
            negative = residual[residual < 0]
            if negative.size < 2:
                break
            negative_mean = float(negative.mean())
            negative_std = float(negative.std())
            if negative_std <= np.finfo(float).eps:
                break
            exponent = np.clip(
                2.0 * (residual - (2.0 * negative_std - negative_mean)) / negative_std,
                -100.0,
                100.0,
            )
            new_weights = 1.0 / (1.0 + np.exp(exponent))
            relative_change = np.linalg.norm(new_weights - weights) / max(
                np.linalg.norm(weights),
                np.finfo(float).eps,
            )
            weights = new_weights
            if relative_change < tolerance:
                break
        return baseline

    def detect_peaks(self, spectrum: np.ndarray) -> list[DetectedSAXSPeak]:
        """Detect significant peaks in one background-subtracted spectrum.

        Height is evaluated as the natural log of robust signal-to-noise, and
        prominence is therefore a natural-log peak-to-local-base ratio.
        Peak widths are measured at half height on the linear corrected
        intensity, then represented in log-q for the frozen integration
        intervals.
        """
        self._require_sampling_configured()
        spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
        if spectrum.shape != self.q_grid.shape:
            raise ValueError("Peak detection requires a spectrum on the common q grid.")
        smoothed = (
            spectrum
            if self.peak_smoothing_sigma == 0
            else gaussian_filter1d(
                spectrum,
                sigma=self.peak_smoothing_sigma,
                mode="nearest",
            )
        )
        differences = np.diff(spectrum)
        difference_median = np.median(differences)
        noise_scale = (
            1.4826 * np.median(np.abs(differences - difference_median)) / np.sqrt(2.0)
        )
        noise_scale = max(
            float(noise_scale),
            np.finfo(float).eps * max(float(np.max(np.abs(spectrum))), 1.0),
        )
        detection_signal = np.log(np.maximum(smoothed / noise_scale, 1.0))
        log_q = np.log(self.q_grid)
        log_q_step = float(log_q[1] - log_q[0])
        peak_indices, properties = find_peaks(
            detection_signal,
            height=self.peak_min_height,
            prominence=self.peak_min_prominence,
        )
        detected_widths = peak_widths(
            np.maximum(smoothed, 0.0),
            peak_indices,
            rel_height=0.5,
        )[0]

        detected = []
        for result_index, peak_index in enumerate(peak_indices):
            width_log_q = float(detected_widths[result_index]) * log_q_step
            if width_log_q < self.peak_min_width_log_q:
                continue
            if (
                self.peak_max_width_log_q is not None
                and width_log_q > self.peak_max_width_log_q
            ):
                continue
            width_log_q = max(width_log_q, log_q_step)
            half_window = 0.5 * self.peak_window_width_factor * width_log_q
            log_q_position = float(log_q[peak_index])
            log_q_left = max(log_q_position - half_window, float(log_q[0]))
            log_q_right = min(log_q_position + half_window, float(log_q[-1]))
            q_left = float(np.exp(log_q_left))
            q_right = float(np.exp(log_q_right))
            detected.append(
                DetectedSAXSPeak(
                    q_position=float(self.q_grid[peak_index]),
                    log_q_position=log_q_position,
                    q_left=q_left,
                    q_right=q_right,
                    width_log_q=width_log_q,
                    height=float(properties["peak_heights"][result_index]),
                    prominence=float(properties["prominences"][result_index]),
                    integrated_area=self.integrate_peak_area(
                        spectrum,
                        q_left,
                        q_right,
                    ),
                )
            )
        return detected

    def integrate_peak_area(
        self,
        spectrum: np.ndarray,
        q_left: float,
        q_right: float,
    ) -> float:
        """Integrate positive background-subtracted intensity over a q interval."""
        spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
        mask = (self.q_grid >= q_left) & (self.q_grid <= q_right)
        if np.count_nonzero(mask) < 2:
            return 0.0
        return float(
            np.trapezoid(
                np.maximum(spectrum[mask], 0.0),
                self.q_grid[mask],
            )
        )

    def get_peak_height(
        self,
        spectrum: np.ndarray,
        q_left: float,
        q_right: float,
    ) -> float:
        """Return the maximum positive intensity in a q interval."""
        spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
        mask = (self.q_grid >= q_left) & (self.q_grid <= q_right)
        if not np.any(mask):
            return 0.0
        return max(float(np.max(spectrum[mask])), 0.0)

    @staticmethod
    def _peak_intervals_overlap(
        q_left: float,
        q_right: float,
        other_q_left: float,
        other_q_right: float,
    ) -> bool:
        """Return whether two width-derived q intervals overlap."""
        return q_left <= other_q_right and other_q_left <= q_right

    def _detected_peak_matches_dictionary(
        self,
        detected_peak: DetectedSAXSPeak,
    ) -> bool:
        """Return whether a detected peak overlaps an active dictionary entry."""
        return any(
            self._peak_intervals_overlap(
                detected_peak.q_left,
                detected_peak.q_right,
                peak.q_left,
                peak.q_right,
            )
            for peak in self.peak_dict.values()
        )

    def _add_detected_peak(
        self,
        detected_peak: DetectedSAXSPeak,
        discovery_measurement_index: int,
    ) -> None:
        """Add one detected peak with a frozen center and integration interval."""
        peak_id = self._next_peak_id
        self._next_peak_id += 1
        self.peak_dict[peak_id] = SAXSPeak(
            peak_id=peak_id,
            q_position=detected_peak.q_position,
            log_q_position=detected_peak.log_q_position,
            q_left=detected_peak.q_left,
            q_right=detected_peak.q_right,
            width_log_q=detected_peak.width_log_q,
            max_integrated_area=max(
                self.integrate_peak_area(
                    measurement.spectrum,
                    detected_peak.q_left,
                    detected_peak.q_right,
                )
                for measurement in self.measurements
            ),
            discovery_measurement_index=discovery_measurement_index,
        )

    @staticmethod
    def _match_known_peak(
        q_position: float,
        detected_peaks: list[DetectedSAXSPeak],
    ) -> DetectedSAXSPeak | None:
        """Return the closest detected peak whose FWHM contains a known q."""
        log_q_position = float(np.log(q_position))
        matches = [
            peak
            for peak in detected_peaks
            if abs(log_q_position - peak.log_q_position) <= 0.5 * peak.width_log_q
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda peak: (
                abs(log_q_position - peak.log_q_position),
                -peak.integrated_area,
            ),
        )

    def _initialize_known_peak_dictionary(self) -> None:
        """Create exactly one dictionary entry for each known peak q."""
        detections = [
            self.detect_peaks(measurement.spectrum) for measurement in self.measurements
        ]
        self.peak_dict = {}
        for peak_id, q_position in enumerate(self.known_peak_q_values):
            candidates = [
                (matched_peak, measurement_index)
                for measurement_index, detected_peaks in enumerate(detections)
                if (
                    matched_peak := self._match_known_peak(
                        float(q_position),
                        detected_peaks,
                    )
                )
                is not None
            ]
            if candidates:
                matched_peak, discovery_index = min(
                    candidates,
                    key=lambda item: (
                        abs(np.log(q_position) - item[0].log_q_position),
                        -item[0].integrated_area,
                    ),
                )
                q_left = matched_peak.q_left
                q_right = matched_peak.q_right
                width_log_q = matched_peak.width_log_q
                max_integrated_area = max(
                    self.integrate_peak_area(
                        measurement.spectrum,
                        q_left,
                        q_right,
                    )
                    for measurement in self.measurements
                )
            else:
                discovery_index = -1
                q_left = float(q_position)
                q_right = float(q_position)
                width_log_q = 0.0
                max_integrated_area = 0.0
            self.peak_dict[peak_id] = SAXSPeak(
                peak_id=peak_id,
                q_position=float(q_position),
                log_q_position=float(np.log(q_position)),
                q_left=q_left,
                q_right=q_right,
                width_log_q=width_log_q,
                max_integrated_area=max_integrated_area,
                discovery_measurement_index=discovery_index,
            )
        self._next_peak_id = len(self.peak_dict)
        self._peak_detection_measurement_count = len(self.measurements)

    def _initialize_peak_dictionary(self) -> None:
        """Build the initial dictionary from all initial spectra."""
        if self.known_peak_q_values is not None:
            self._initialize_known_peak_dictionary()
            return
        candidates = []
        for measurement_index, measurement in enumerate(self.measurements):
            candidates.extend(
                (peak, measurement_index)
                for peak in self.detect_peaks(measurement.spectrum)
            )
        candidates.sort(
            key=lambda item: item[0].integrated_area,
            reverse=True,
        )
        for detected_peak, measurement_index in candidates:
            if len(self.peak_dict) >= self.num_initial_peaks:
                break
            if not self._detected_peak_matches_dictionary(detected_peak):
                self._add_detected_peak(detected_peak, measurement_index)
        self._peak_detection_measurement_count = len(self.measurements)
        if not self.peak_dict:
            raise ValueError(
                "No peaks met the configured height, prominence, and width "
                "criteria in the initial SAXS measurements."
            )

    def update_peak_dictionary(self) -> None:
        """Update peak areas and admit new peaks from unprocessed measurements."""
        if not self.peak_dict:
            self._initialize_peak_dictionary()
            return
        if self.known_peak_q_values is not None:
            for measurement_index in range(
                self._peak_detection_measurement_count,
                len(self.measurements),
            ):
                measurement = self.measurements[measurement_index]
                detected_peaks = self.detect_peaks(measurement.spectrum)
                for peak in self.peak_dict.values():
                    if peak.width_log_q == 0:
                        matched_peak = self._match_known_peak(
                            peak.q_position,
                            detected_peaks,
                        )
                        if matched_peak is not None:
                            peak.q_left = matched_peak.q_left
                            peak.q_right = matched_peak.q_right
                            peak.width_log_q = matched_peak.width_log_q
                            peak.discovery_measurement_index = measurement_index
                            peak.max_integrated_area = max(
                                self.integrate_peak_area(
                                    previous_measurement.spectrum,
                                    peak.q_left,
                                    peak.q_right,
                                )
                                for previous_measurement in self.measurements
                            )
                            continue
                    peak.max_integrated_area = max(
                        peak.max_integrated_area,
                        self.integrate_peak_area(
                            measurement.spectrum,
                            peak.q_left,
                            peak.q_right,
                        ),
                    )
            self._peak_detection_measurement_count = len(self.measurements)
            return
        for measurement_index in range(
            self._peak_detection_measurement_count,
            len(self.measurements),
        ):
            measurement = self.measurements[measurement_index]
            for peak in self.peak_dict.values():
                peak.max_integrated_area = max(
                    peak.max_integrated_area,
                    self.integrate_peak_area(
                        measurement.spectrum,
                        peak.q_left,
                        peak.q_right,
                    ),
                )
            strongest_peak_area = max(
                peak.max_integrated_area for peak in self.peak_dict.values()
            )
            minimum_new_peak_area = (
                self.new_peak_min_relative_area * strongest_peak_area
            )
            new_candidates = sorted(
                self.detect_peaks(measurement.spectrum),
                key=lambda peak: peak.integrated_area,
                reverse=True,
            )
            for detected_peak in new_candidates:
                if self._detected_peak_matches_dictionary(detected_peak):
                    continue
                if detected_peak.integrated_area < minimum_new_peak_area:
                    continue
                self._add_detected_peak(detected_peak, measurement_index)
                if len(self.peak_dict) > self.max_peaks_in_dict:
                    weakest_peak_id = min(
                        self.peak_dict,
                        key=lambda peak_id: self.peak_dict[peak_id].max_integrated_area,
                    )
                    del self.peak_dict[weakest_peak_id]
        self._peak_detection_measurement_count = len(self.measurements)

    def get_peak_integrated_areas(
        self,
        measurement_indices: list[int] | None = None,
        peak_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Return integrated areas with measurements in rows and peaks in columns."""
        if measurement_indices is None:
            measurement_indices = list(range(len(self.measurements)))
        if peak_ids is None:
            peak_ids = sorted(self.peak_dict)
        return np.asarray(
            [
                [
                    self.integrate_peak_area(
                        self.measurements[measurement_index].spectrum,
                        self.peak_dict[peak_id].q_left,
                        self.peak_dict[peak_id].q_right,
                    )
                    for peak_id in peak_ids
                ]
                for measurement_index in measurement_indices
            ],
            dtype=float,
        )

    def get_peak_heights(
        self,
        measurement_indices: list[int] | None = None,
        peak_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Return peak heights with measurements in rows and peaks in columns."""
        if measurement_indices is None:
            measurement_indices = list(range(len(self.measurements)))
        if peak_ids is None:
            peak_ids = sorted(self.peak_dict)
        return np.asarray(
            [
                [
                    (
                        max(
                            float(
                                np.interp(
                                    self.peak_dict[peak_id].q_position,
                                    self.q_grid,
                                    self.measurements[measurement_index].spectrum,
                                )
                            ),
                            0.0,
                        )
                        if self.peak_dict[peak_id].width_log_q == 0
                        else self.get_peak_height(
                            self.measurements[measurement_index].spectrum,
                            self.peak_dict[peak_id].q_left,
                            self.peak_dict[peak_id].q_right,
                        )
                    )
                    for peak_id in peak_ids
                ]
                for measurement_index in measurement_indices
            ],
            dtype=float,
        )

    def get_peak_observables(
        self,
        measurement_indices: list[int] | None = None,
        peak_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Return the configured peak observables."""
        if self.peak_observable == "height":
            return self.get_peak_heights(measurement_indices, peak_ids)
        return self.get_peak_integrated_areas(measurement_indices, peak_ids)

    def refit_model(self) -> None:
        """Update the peak dictionary and refit the independent peak GPs."""
        self._require_sampling_configured()
        self.update_peak_dictionary()
        fitting_indices = [
            index
            for index in range(len(self.measurements))
            if index not in self.excluded_measurement_indices
        ]
        try:
            model_state = self._fit_model(fitting_indices, fit_mll=True)
        except RuntimeError as exc:
            if "Must provide inverse transform" not in str(exc):
                raise
            excluded_index = fitting_indices[-1]
            self.excluded_measurement_indices.add(excluded_index)
            fitting_indices.remove(excluded_index)
            message = (
                "MLL fitting failed while adding measurement "
                f"{excluded_index + 1}; excluding it from current and future "
                "model fits."
            )
            logger.warning("%s Error: %s", message, exc)
            model_state = self._fit_model(fitting_indices, fit_mll=False)
        (
            self.standardized_peak_scores,
            self.peak_score_mean,
            self.peak_score_std,
            self.modeled_peak_ids,
            self.gp_model,
        ) = model_state

    def _fit_model(
        self,
        measurement_indices: list[int],
        fit_mll: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], Any]:
        """Fit independent GPs to transformed peak observables."""
        peak_ids = sorted(self.peak_dict)
        peak_observables = torch.as_tensor(
            self.get_peak_observables(measurement_indices, peak_ids),
            dtype=torch.double,
        )
        peak_scores = torch.log1p(peak_observables / self.peak_area_scale)
        peak_score_mean = peak_scores.mean(dim=0)
        peak_score_std = peak_scores.std(dim=0, unbiased=False)
        standardized_peak_scores = (peak_scores - peak_score_mean) / (
            peak_score_std + self.epsilon_z
        )
        train_x = torch.as_tensor(
            self.normalize_positions(
                np.vstack(
                    [self.measurements[index].position for index in measurement_indices]
                )
            ),
            dtype=torch.double,
        )
        gp_model = self.fit_gp(
            train_x,
            standardized_peak_scores,
            max_fit_gp_mll_iterations=self.max_fit_gp_mll_iterations,
            fit_mll=fit_mll,
        )
        return (
            standardized_peak_scores,
            peak_score_mean,
            peak_score_std,
            peak_ids,
            gp_model,
        )

    @staticmethod
    def fit_gp(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        max_fit_gp_mll_iterations: int | None = None,
        fit_mll: bool = True,
    ) -> Any:
        """Fit independent GPs for peak targets in normalized position space.

        Parameters
        ----------
        train_x : torch.Tensor
            Normalized measured positions with shape ``(N, 2)``.
        train_y : torch.Tensor
            Standardized peak targets with shape ``(N, N_peaks)``.
        max_fit_gp_mll_iterations : int, optional
            Maximum marginal likelihood optimizer iterations. When omitted,
            do not impose an iteration limit.
        fit_mll : bool
            Whether to optimize the marginal log likelihood.

        Returns
        -------
        ModelListGP
            Fitted collection of independent Gaussian process models.
        """
        peak_models = []
        for peak_index in range(train_y.shape[-1]):
            peak_model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y[:, peak_index : peak_index + 1].double(),
                covar_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=2),
                ),
                outcome_transform=None,
            )
            if fit_mll:
                mll = ExactMarginalLogLikelihood(
                    peak_model.likelihood,
                    peak_model,
                )
                if max_fit_gp_mll_iterations is None:
                    fit_gpytorch_mll(mll)
                else:
                    fit_gpytorch_mll(
                        mll,
                        optimizer_kwargs={
                            "options": {"maxiter": max_fit_gp_mll_iterations}
                        },
                    )
            peak_models.append(peak_model)

        gp_model = ModelListGP(*peak_models)
        gp_model.eval()
        return gp_model

    @property
    def measured_positions(self) -> np.ndarray:
        """Return measured spatial positions as an ``(N_past, 2)`` array."""
        if not self.measurements:
            return np.empty((0, 2), dtype=float)
        return np.vstack([measurement.position for measurement in self.measurements])

    def get_unsampled_candidate_indices(self) -> np.ndarray:
        """Return candidate indices that have not yet been measured."""
        self._require_sampling_configured()
        measured = set(self.measured_candidate_indices)
        return np.array(
            [
                idx
                for idx in range(self.candidate_positions.shape[0])
                if idx not in measured
            ],
            dtype=int,
        )

    def _filter_candidate_indices_by_exclusion(
        self,
        candidate_indices: np.ndarray,
        reference_indices: list[int],
    ) -> np.ndarray:
        """Remove candidates inside the exclusion radius of references."""
        return self._filter_candidate_indices_by_radius(
            candidate_indices,
            reference_indices,
            self.exclusion_radius,
        )

    def _filter_candidate_indices_by_radius(
        self,
        candidate_indices: np.ndarray,
        reference_indices: list[int],
        exclusion_radius: float | None,
    ) -> np.ndarray:
        """Remove candidates within a physical radius of references."""
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        if exclusion_radius is None or not reference_indices:
            return candidate_indices
        candidate_positions = self.candidate_positions[candidate_indices]
        reference_positions = self.candidate_positions[reference_indices]
        minimum_distances = np.linalg.norm(
            candidate_positions[:, None, :] - reference_positions[None, :, :],
            axis=-1,
        ).min(axis=1)
        return candidate_indices[minimum_distances > exclusion_radius]

    def get_eligible_candidate_indices(self) -> np.ndarray:
        """Return unsampled candidates outside measured-point exclusions."""
        return self._filter_candidate_indices_by_exclusion(
            self.get_unsampled_candidate_indices(),
            self.measured_candidate_indices,
        )

    @tool(name="spatial_saxs_adaptive_sampling_engine.suggest")
    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """Return spaced, travel-optimized top-k eligible positions."""
        self._require_sampling_configured()
        k = n_suggestions
        if k < 1:
            raise ValueError("`n_suggestions` must be positive.")
        if self.gp_model is None:
            raise ValueError(
                "Gaussian process model must be fit before adaptive "
                "suggestions are requested. Call "
                "`suggest_initial_measurements` and `update` first."
            )
        include_farthest = self._batch_includes_farthest_exploration(k)
        if k == 1 and include_farthest:
            return self.get_farthest_unsampled_position()[None, :]
        scores = self.compute_acquisition_scores()
        if k > scores.positions.shape[0]:
            raise ValueError(
                "`n_suggestions` cannot exceed the number of eligible candidates."
            )
        order = np.argsort(-scores.acquisition, kind="stable")
        selected: list[np.ndarray] = []
        if include_farthest:
            selected.append(self.get_farthest_unsampled_position())
        for index in order:
            position = scores.positions[index]
            if selected and any(
                np.all(np.isclose(position, selected_position))
                for selected_position in selected
            ):
                continue
            if selected and self.suggestion_exclusion_radius is not None:
                distances = np.linalg.norm(
                    np.asarray(selected) - position,
                    axis=1,
                )
                if np.any(distances <= self.suggestion_exclusion_radius):
                    continue
            selected.append(position)
            if len(selected) == k:
                break
        if len(selected) < k:
            raise ValueError(
                "`suggestion_exclusion_radius` leaves fewer than "
                "`n_suggestions` "
                "eligible suggestions."
            )
        return self._optimize_suggestion_path(np.asarray(selected))

    def _optimize_suggestion_path(self, positions: np.ndarray) -> np.ndarray:
        """Order positions with a low-overhead greedy TSP approximation."""
        positions = np.asarray(positions, dtype=float)
        if positions.shape[0] < 2:
            return positions.copy()

        graph = nx.complete_graph(positions.shape[0])
        for left, right in graph.edges:
            graph[left][right]["weight"] = float(
                np.linalg.norm(positions[left] - positions[right])
            )

        if self.measured_candidate_indices:
            start_node = positions.shape[0]
            graph.add_node(start_node)
            start_position = self.candidate_positions[
                self.measured_candidate_indices[-1]
            ]
            for index, position in enumerate(positions):
                graph.add_edge(
                    start_node,
                    index,
                    weight=float(np.linalg.norm(start_position - position)),
                )
            route = nx.approximation.greedy_tsp(graph, source=start_node)[1:-1]
        else:
            route = nx.approximation.greedy_tsp(graph, source=0)[:-1]
        return positions[route].copy()

    def _is_farthest_exploration_step(self) -> bool:
        """Return whether the next adaptive measurement uses exploration."""
        return self._batch_includes_farthest_exploration(1)

    def _batch_includes_farthest_exploration(self, k: int) -> bool:
        """Return whether a batch spans a scheduled exploration measurement."""
        if self.exploration_interval is None:
            return False
        adaptive_measurements = max(
            len(self.measurements) - self.num_initial_samples,
            0,
        )
        completed_intervals = adaptive_measurements // self.exploration_interval
        completed_intervals_after_batch = (
            adaptive_measurements + k
        ) // self.exploration_interval
        return completed_intervals_after_batch > completed_intervals

    def get_farthest_unsampled_position(self) -> np.ndarray:
        """Return the eligible position farthest from all measured positions."""
        eligible_indices = self.get_eligible_candidate_indices()
        if eligible_indices.size == 0:
            raise ValueError("No eligible candidate positions remain.")
        if not self.measured_candidate_indices:
            return self.candidate_positions[eligible_indices[0]].copy()
        eligible = self.candidate_positions_normalized[eligible_indices]
        measured = self.candidate_positions_normalized[self.measured_candidate_indices]
        minimum_distances = np.linalg.norm(
            eligible[:, None, :] - measured[None, :, :],
            axis=-1,
        ).min(axis=1)
        return self.candidate_positions[
            eligible_indices[int(np.argmax(minimum_distances))]
        ].copy()

    def compute_acquisition_scores(self) -> AcquisitionScores:
        """Compute acquisition scores over all unsampled candidates."""
        self._require_sampling_configured()
        eligible_indices = self.get_eligible_candidate_indices()
        if eligible_indices.size == 0:
            raise ValueError("No eligible candidate positions remain.")
        candidate_x = torch.as_tensor(
            self.candidate_positions_normalized[eligible_indices],
            dtype=torch.double,
        )
        if self.w_peak == 0:
            posterior = self.gp_model.posterior(candidate_x)
            mean = posterior.mean
            variance = posterior.variance.clamp_min(0.0)
            full_peak_observable_tilde = None
        else:
            full_candidate_x = torch.as_tensor(
                self.candidate_positions_normalized,
                dtype=torch.double,
            )
            posterior = self.gp_model.posterior(full_candidate_x)
            full_mean = posterior.mean
            eligible_tensor_indices = torch.as_tensor(
                eligible_indices,
                dtype=torch.long,
                device=full_mean.device,
            )
            mean = full_mean[eligible_tensor_indices]
            variance = posterior.variance[eligible_tensor_indices].clamp_min(0.0)
            full_peak_observable_tilde = self.compute_concentration_gated_peak_score(
                full_mean
            )
        sigma = torch.sqrt(variance.mean(dim=-1))

        sigma_tilde = self.normalize_tensor(sigma)
        gradient_tilde = (
            torch.zeros_like(sigma)
            if self.w_g == 0
            else self.normalize_tensor(
                self.compute_peak_gradient_magnitude(candidate_x)
            )
        )
        peak_observable_tilde = (
            torch.zeros_like(sigma)
            if self.w_peak == 0
            else full_peak_observable_tilde[eligible_tensor_indices]
        )
        acquisition = sigma_tilde * (
            self.w_peak * peak_observable_tilde
            + self.w_g * gradient_tilde
            + self.epsilon_acquisition
        )
        scores = AcquisitionScores(
            positions=self.candidate_positions[eligible_indices].copy(),
            acquisition=acquisition.detach().cpu().numpy(),
            sigma_tilde=sigma_tilde.detach().cpu().numpy(),
            peak_area_tilde=peak_observable_tilde.detach().cpu().numpy(),
            gradient_tilde=gradient_tilde.detach().cpu().numpy(),
        )
        self.latest_scores = scores
        del mean
        return scores

    def inverse_transform_peak_scores(
        self,
        standardized_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Transform standardized GP scores back to physical peak observables."""
        score_mean = self.peak_score_mean.to(
            dtype=standardized_scores.dtype,
            device=standardized_scores.device,
        )
        score_std = self.peak_score_std.to(
            dtype=standardized_scores.dtype,
            device=standardized_scores.device,
        )
        transformed_scores = (
            standardized_scores * (score_std + self.epsilon_z) + score_mean
        )
        return (self.peak_area_scale * torch.expm1(transformed_scores)).clamp_min(0.0)

    def compute_predicted_peak_areas(
        self,
        standardized_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted observables under the legacy area-specific name."""
        return self.inverse_transform_peak_scores(standardized_mean)

    def compute_predicted_max_peak_observable(
        self,
        standardized_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Return the maximum predicted peak observable at each position."""
        return self.inverse_transform_peak_scores(standardized_mean).max(dim=-1).values

    def compute_predicted_max_peak_area(
        self,
        standardized_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Return the maximum observable under the legacy area-specific name."""
        return self.compute_predicted_max_peak_observable(standardized_mean)

    def compute_concentration_gated_peak_score(
        self,
        standardized_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Return the maximum robustly normalized eligible peak map."""
        observables = self.inverse_transform_peak_scores(standardized_mean)
        observables = self.blur_peak_observable_maps(observables)
        upper_caps = torch.quantile(observables, 0.99, dim=0)
        clipped_observables = torch.minimum(observables, upper_caps.unsqueeze(0))
        top_count = max(1, int(np.ceil(0.1 * observables.shape[0])))
        top_observable = torch.topk(
            clipped_observables,
            k=top_count,
            dim=0,
        ).values.sum(dim=0)
        total_observable = clipped_observables.sum(dim=0)
        concentration = top_observable / (total_observable + self.epsilon_normalization)
        eligible = (concentration >= self.peak_map_min_concentration) & (
            total_observable > self.epsilon_normalization
        )

        lower = torch.quantile(observables, 0.50, dim=0)
        upper = torch.quantile(observables, 0.95, dim=0)
        normalized = (
            (observables - lower.unsqueeze(0))
            / (upper.unsqueeze(0) - lower.unsqueeze(0) + self.epsilon_normalization)
        ).clamp(0.0, 1.0)
        if not torch.any(eligible):
            return torch.zeros(
                observables.shape[0],
                dtype=observables.dtype,
                device=observables.device,
            )
        return normalized[:, eligible].max(dim=-1).values

    def blur_peak_observable_maps(
        self,
        observables: torch.Tensor,
    ) -> torch.Tensor:
        """Blur peak-observable maps using a truncated physical spatial width.

        Candidate contributions beyond four blur widths are omitted. The
        normalized sparse weight matrix is cached for repeated calls.

        Parameters
        ----------
        observables : torch.Tensor
            Peak observables with one row per candidate position and one
            column per modeled peak.

        Returns
        -------
        torch.Tensor
            Peak observables after spatial Gaussian smoothing.
        """
        if not self.peak_observale_map_blur:
            return observables

        weight_matrix = self._peak_blur_weight_matrix
        if (
            weight_matrix is None
            or weight_matrix.dtype != observables.dtype
            or weight_matrix.device != observables.device
        ):
            weight_matrix = torch.sparse_coo_tensor(
                torch.as_tensor(
                    self._peak_blur_indices,
                    dtype=torch.long,
                    device=observables.device,
                ),
                torch.as_tensor(
                    self._peak_blur_weights,
                    dtype=observables.dtype,
                    device=observables.device,
                ),
                size=(self.candidate_positions.shape[0],) * 2,
                dtype=observables.dtype,
                device=observables.device,
                is_coalesced=True,
                check_invariants=False,
            )
            self._peak_blur_weight_matrix = weight_matrix
        return torch.sparse.mm(weight_matrix, observables)

    def _build_peak_blur_weights(self) -> None:
        """Build a normalized sparse Gaussian kernel for candidate positions."""
        self._peak_blur_indices = None
        self._peak_blur_weights = None
        self._peak_blur_weight_matrix = None
        if not self.peak_observale_map_blur:
            return

        positions = self.candidate_positions
        blur_width = self.peak_observale_map_blur
        position_tree = cKDTree(positions)
        distances = position_tree.sparse_distance_matrix(
            position_tree,
            max_distance=4.0 * blur_width,
            output_type="coo_matrix",
        )
        weights = np.exp(-0.5 * (distances.data / blur_width) ** 2)
        row_sums = np.bincount(
            distances.row,
            weights=weights,
            minlength=positions.shape[0],
        )
        weights /= row_sums[distances.row]
        order = np.lexsort((distances.col, distances.row))
        self._peak_blur_indices = np.stack(
            (distances.row[order], distances.col[order])
        )
        self._peak_blur_weights = weights[order]

    def compute_peak_gradient_magnitude(
        self,
        candidate_x: torch.Tensor,
    ) -> torch.Tensor:
        """Return aggregate standardized peak-response gradient magnitude."""
        self._require_sampling_configured()
        x = candidate_x.clone().detach().requires_grad_(True)
        mean = self.gp_model.posterior(x).mean
        gradients = []
        for peak_index in range(len(self.modeled_peak_ids)):
            grad = torch.autograd.grad(
                mean[:, peak_index].sum(),
                x,
                retain_graph=True,
                create_graph=False,
            )[0]
            gradients.append(grad)
        stacked = torch.stack(gradients, dim=0)
        return torch.sqrt(torch.sum(stacked.pow(2), dim=(0, 2)).clamp_min(0.0))

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Robustly percentile-normalize a tensor to the interval [0, 1]."""
        self._require_sampling_configured()
        lower = torch.quantile(
            values,
            self.normalization_lower_percentile / 100.0,
        )
        upper = torch.quantile(
            values,
            self.normalization_upper_percentile / 100.0,
        )
        clipped = values.clamp(min=lower, max=upper)
        return (clipped - lower) / (upper - lower + self.epsilon_normalization)

    def get_candidate_index(self, position: np.ndarray) -> int:
        """Return the exact candidate index for a configured position."""
        self._require_sampling_configured()
        position = np.asarray(position, dtype=float).reshape(1, 2)
        matches = np.where(
            np.all(np.isclose(self.candidate_positions, position), axis=1)
        )[0]
        if matches.size != 1:
            raise ValueError(f"Position is not a unique candidate: {position.ravel()}.")
        return int(matches[0])
