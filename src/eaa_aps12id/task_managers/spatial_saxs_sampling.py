from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import eaa_core.matplotlib_setup  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import KroneckerMultiTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.tool.base import BaseTool
from eaa_core.tool.mcp_adapter import MCPRPCWrapper

logger = logging.getLogger(__name__)


@dataclass
class SAXSMeasurement:
    """Recorded SAXS measurement at one spatial position."""

    position: np.ndarray
    q: np.ndarray
    intensity: np.ndarray
    spectrum: np.ndarray


@dataclass
class AcquisitionScores:
    """Acquisition terms for unsampled candidate positions."""

    positions: np.ndarray
    acquisition: np.ndarray
    sigma_tilde: np.ndarray
    q_div_tilde: np.ndarray
    gradient_tilde: np.ndarray


class SpatialSAXSAdaptiveSamplingTaskManager(BaseTaskManager):
    """Adaptive sampler for spatially resolved SAXS.

    The workflow measures spectra on a finite spatial mesh, preprocesses each
    ``(q, I)`` SAXS spectrum onto a common log-spaced q grid, learns a PCA latent
    representation, and uses a joint multi-output GP acquisition function to
    choose additional measurement positions.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        acquisition_tool: BaseTool | MCPRPCWrapper | None = None,
        checkpoint_db_path: Optional[str] = "checkpoint.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the adaptive SAXS sampler.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration forwarded to ``BaseTaskManager``.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to ``BaseTaskManager``.
        acquisition_tool : BaseTool or MCPRPCWrapper
            Object exposing ``acquire_saxs`` with ``x`` and ``y`` position
            arguments.
        checkpoint_db_path : str, optional
            Checkpoint database path forwarded to ``BaseTaskManager``.
        build : bool, optional
            Whether to build the base task manager.
        """
        if torch is None:
            raise ImportError(
                "SpatialSAXSAdaptiveSamplingTaskManager requires torch, botorch, "
                "and gpytorch. Install this package with its declared dependencies."
            )
        if acquisition_tool is None:
            raise ValueError("`acquisition_tool` must be provided.")
        if not isinstance(acquisition_tool, (BaseTool, MCPRPCWrapper)):
            raise TypeError(
                "`acquisition_tool` must be an instance of BaseTool or MCPRPCWrapper."
            )
        if not hasattr(acquisition_tool, "acquire_saxs"):
            raise ValueError("`acquisition_tool` must expose `acquire_saxs`.")

        self.acquisition_tool = acquisition_tool
        self._sampling_config: dict[str, Any] | None = None
        self.x_values: np.ndarray | None = None
        self.y_values: np.ndarray | None = None
        self.candidate_positions: np.ndarray | None = None
        self.q_min: float | None = None
        self.q_max: float | None = None
        self.num_q_points: int | None = None
        self.epsilon_intensity: float | None = None
        self.num_initial_samples: int | None = None
        self.max_measurements: int | None = None
        self.num_pca_components: int | None = None
        self.lambda_logdet: float | None = None
        self.num_mc_samples: int | None = None
        self.w_d: float | None = None
        self.w_g: float | None = None
        self.epsilon: float | None = None
        self.epsilon_z: float | None = None
        self.random_seed: int | None = None
        self.q_grid: np.ndarray | None = None
        self.position_min: np.ndarray | None = None
        self.position_max: np.ndarray | None = None
        self.position_span: np.ndarray | None = None
        self.candidate_positions_normalized: np.ndarray | None = None
        self.measurements: list[SAXSMeasurement] = []
        self.measured_candidate_indices: list[int] = []
        self.standardized_latent_scores: torch.Tensor | None = None
        self.gp_model = None
        self.latest_scores: AcquisitionScores | None = None
        self.posterior_visualization_tile_id: str | None = None
        self.saxs_spectra_visualization_tile_id: str | None = None

        tools = [acquisition_tool] if isinstance(acquisition_tool, BaseTool) else []
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            tools=tools,
            checkpoint_db_path=checkpoint_db_path,
            build=build,
            *args,
            **kwargs,
        )

    @staticmethod
    def _validate_axis_values(values: Any, name: str) -> np.ndarray:
        if values is None:
            raise ValueError(f"`{name}` must be provided.")
        axis = np.asarray(values, dtype=float).reshape(-1)
        if axis.size < 2:
            raise ValueError(f"`{name}` must contain at least two values.")
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"`{name}` must contain only finite values.")
        if np.unique(axis).size != axis.size:
            raise ValueError(f"`{name}` must not contain duplicates.")
        return axis

    @staticmethod
    def _build_candidate_positions(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
        return np.column_stack([x_grid.ravel(), y_grid.ravel()])

    def _configure_sampling_run(
        self,
        x_values: np.ndarray | list[float] | tuple[float, ...],
        y_values: np.ndarray | list[float] | tuple[float, ...],
        q_min: float = 0.001,
        q_max: float = 1.0,
        num_q_points: int = 256,
        epsilon_intensity: float = 1e-12,
        num_initial_samples: int = 5,
        max_measurements: int = 20,
        num_pca_components: int = 2,
        lambda_logdet: float = 1e-6,
        num_mc_samples: int = 64,
        w_d: float = 1.0,
        w_g: float = 1.0,
        epsilon: float = 1e-12,
        epsilon_z: float = 1e-12,
        random_seed: int | None = None,
    ) -> None:
        """Configure sampling parameters for one adaptive run."""
        x_axis = self._validate_axis_values(x_values, "x_values")
        y_axis = self._validate_axis_values(y_values, "y_values")
        config = {
            "x_values": tuple(x_axis.tolist()),
            "y_values": tuple(y_axis.tolist()),
            "q_min": float(q_min),
            "q_max": float(q_max),
            "num_q_points": int(num_q_points),
            "epsilon_intensity": float(epsilon_intensity),
            "num_initial_samples": int(num_initial_samples),
            "max_measurements": int(max_measurements),
            "num_pca_components": int(num_pca_components),
            "lambda_logdet": float(lambda_logdet),
            "num_mc_samples": int(num_mc_samples),
            "w_d": float(w_d),
            "w_g": float(w_g),
            "epsilon": float(epsilon),
            "epsilon_z": float(epsilon_z),
            "random_seed": random_seed,
        }
        if self.measurements and self._sampling_config != config:
            raise ValueError(
                "Cannot change adaptive sampling parameters after measurements "
                "have been collected. Create a new task manager for a new run."
            )
        if self._sampling_config == config:
            return

        self._sampling_config = config
        self.x_values = x_axis
        self.y_values = y_axis
        self.candidate_positions = self._build_candidate_positions(
            self.x_values, self.y_values
        )
        self.q_min = config["q_min"]
        self.q_max = config["q_max"]
        self.num_q_points = config["num_q_points"]
        self.epsilon_intensity = config["epsilon_intensity"]
        self.num_initial_samples = config["num_initial_samples"]
        self.max_measurements = config["max_measurements"]
        self.num_pca_components = config["num_pca_components"]
        self.lambda_logdet = config["lambda_logdet"]
        self.num_mc_samples = config["num_mc_samples"]
        self.w_d = config["w_d"]
        self.w_g = config["w_g"]
        self.epsilon = config["epsilon"]
        self.epsilon_z = config["epsilon_z"]
        self.random_seed = random_seed
        self._validate_parameters()

        self.q_grid = self.create_log_q_grid()
        self.position_min = self.candidate_positions.min(axis=0)
        self.position_max = self.candidate_positions.max(axis=0)
        self.position_span = self.position_max - self.position_min
        self.candidate_positions_normalized = self.normalize_positions(
            self.candidate_positions
        )

    def _require_sampling_configured(self) -> None:
        """Raise if adaptive sampling parameters have not been configured."""
        if self._sampling_config is None:
            raise ValueError("Adaptive sampling parameters must be provided to `run` first.")

    def _validate_parameters(self) -> None:
        self._require_sampling_configured()
        if not (0 < self.q_min < self.q_max):
            raise ValueError("Expected `0 < q_min < q_max`.")
        if self.num_q_points < 2:
            raise ValueError("`num_q_points` must be at least 2.")
        if self.epsilon_intensity <= 0:
            raise ValueError("`epsilon_intensity` must be positive.")
        if not (1 <= self.num_initial_samples <= self.max_measurements):
            raise ValueError(
                "Expected `1 <= num_initial_samples <= max_measurements`."
            )
        if self.max_measurements > self.candidate_positions.shape[0]:
            raise ValueError(
                "`max_measurements` cannot exceed the number of candidate positions."
            )
        if self.num_pca_components < 1:
            raise ValueError("`num_pca_components` must be positive.")
        if self.num_pca_components >= self.num_initial_samples:
            raise ValueError(
                "`num_pca_components` must be smaller than `num_initial_samples` "
                "for initial PCA fitting."
            )
        if self.lambda_logdet <= 0:
            raise ValueError("`lambda_logdet` must be positive.")
        if self.num_mc_samples < 1:
            raise ValueError("`num_mc_samples` must be positive.")
        if self.epsilon <= 0 or self.epsilon_z <= 0:
            raise ValueError("`epsilon` and `epsilon_z` must be positive.")

    def create_log_q_grid(self) -> np.ndarray:
        """Return the common log-spaced q grid."""
        self._require_sampling_configured()
        return np.exp(
            np.linspace(np.log(self.q_min), np.log(self.q_max), self.num_q_points)
        )

    def normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to the full candidate-grid bounds."""
        self._require_sampling_configured()
        positions = np.asarray(positions, dtype=float)
        return (positions - self.position_min) / self.position_span

    def run(
        self,
        x_values: np.ndarray | list[float] | tuple[float, ...],
        y_values: np.ndarray | list[float] | tuple[float, ...],
        q_min: float = 0.001,
        q_max: float = 1.0,
        num_q_points: int = 256,
        epsilon_intensity: float = 1e-12,
        num_initial_samples: int = 5,
        max_measurements: int = 20,
        num_pca_components: int = 2,
        lambda_logdet: float = 1e-6,
        num_mc_samples: int = 64,
        w_d: float = 1.0,
        w_g: float = 1.0,
        epsilon: float = 1e-12,
        epsilon_z: float = 1e-12,
        random_seed: int | None = None,
        n_iterations: int | None = None,
        non_position_kwargs_for_acquisition_tool: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Run adaptive SAXS sampling.

        Parameters
        ----------
        x_values : array-like
            X coordinates used to build the finite Cartesian candidate grid.
        y_values : array-like
            Y coordinates used to build the finite Cartesian candidate grid.
        q_min : float
            Minimum q value for the common grid.
        q_max : float
            Maximum q value for the common grid.
        num_q_points : int
            Number of common q-grid points, denoted ``N_q``.
        epsilon_intensity : float
            Intensity floor used by the log transform, denoted ``epsilon_I``.
        num_initial_samples : int
            Number of initial Sobol measurements, denoted ``N_init``.
        max_measurements : int
            Total measurement budget, including initial measurements, denoted
            ``N_max``.
        num_pca_components : int
            Number of retained PCA latent components, denoted ``N_pcs``.
        lambda_logdet : float
            Regularization added to the latent scatter matrix.
        num_mc_samples : int
            Number of Monte Carlo samples for expected diversity gain, denoted
            ``N_mc``.
        w_d : float
            Diversity acquisition weight.
        w_g : float
            Spatial-gradient acquisition weight.
        epsilon : float
            Numerical constant for normalization denominators.
        epsilon_z : float
            Numerical constant for latent standardization denominators.
        random_seed : int, optional
            Seed for Sobol scrambling and posterior sampling.
        n_iterations : int, optional
            Maximum number of new measurements to acquire in this call. When
            omitted, run until the total measurement count reaches
            ``max_measurements``.
        non_position_kwargs_for_acquisition_tool : dict, optional
            Keyword arguments other than ``x`` and ``y`` to pass through to the
            acquisition tool's ``acquire_saxs`` method. When omitted, ``q_min``
            and ``q_max`` from this manager are used.
        """
        del args, kwargs
        self._configure_sampling_run(
            x_values=x_values,
            y_values=y_values,
            q_min=q_min,
            q_max=q_max,
            num_q_points=num_q_points,
            epsilon_intensity=epsilon_intensity,
            num_initial_samples=num_initial_samples,
            max_measurements=max_measurements,
            num_pca_components=num_pca_components,
            lambda_logdet=lambda_logdet,
            num_mc_samples=num_mc_samples,
            w_d=w_d,
            w_g=w_g,
            epsilon=epsilon,
            epsilon_z=epsilon_z,
            random_seed=random_seed,
        )
        acquisition_kwargs = self._resolve_acquisition_kwargs(
            non_position_kwargs_for_acquisition_tool
        )
        self._record_progress_message(
            "Starting adaptive SAXS sampling with "
            f"{len(self.measurements)}/{self.max_measurements} measurements."
        )
        if not self.measurements:
            self.collect_initial_measurements(acquisition_kwargs)
        if len(self.measurements) >= self.max_measurements:
            self._record_progress_message(
                f"Measurement budget reached: {len(self.measurements)}/"
                f"{self.max_measurements}."
            )
            return

        remaining_budget = self.max_measurements - len(self.measurements)
        n_steps = remaining_budget if n_iterations is None else min(
            int(n_iterations), remaining_budget
        )
        for _ in range(n_steps):
            position = self.suggest_next_positions(k=1)[0]
            candidate_index = self.get_candidate_index(position)
            self._record_progress_message(
                "Selected next SAXS position "
                f"x={position[0]:.6g}, y={position[1]:.6g}."
            )
            self.measure_candidate(candidate_index, acquisition_kwargs)
            self.refit_model()
            if len(self.measurements) >= self.max_measurements:
                break
        self._record_progress_message(
            "Adaptive SAXS sampling complete for this run with "
            f"{len(self.measurements)}/{self.max_measurements} measurements."
        )

    def _resolve_acquisition_kwargs(
        self, non_position_kwargs_for_acquisition_tool: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Return non-position keyword arguments for ``acquire_saxs``."""
        self._require_sampling_configured()
        if non_position_kwargs_for_acquisition_tool is None:
            return {"q_min": self.q_min, "q_max": self.q_max}
        return dict(non_position_kwargs_for_acquisition_tool)

    def collect_initial_measurements(
        self, non_position_kwargs_for_acquisition_tool: dict[str, Any] | None = None
    ) -> None:
        """Acquire the initial Sobol measurements and fit the first model."""
        acquisition_kwargs = self._resolve_acquisition_kwargs(
            non_position_kwargs_for_acquisition_tool
        )
        self._record_progress_message(
            f"Collecting {self.num_initial_samples} initial SAXS measurements."
        )
        for candidate_index in self.get_initial_candidate_indices():
            self.measure_candidate(candidate_index, acquisition_kwargs)
        self.refit_model()

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
                self.candidate_positions_normalized, dtype=torch.double
            )
            distances = torch.cdist(sobol, candidates)
            nearest = torch.argmin(distances, dim=1).detach().cpu().numpy()
            for idx in nearest:
                idx_int = int(idx)
                if idx_int not in selected_set:
                    selected.append(idx_int)
                    selected_set.add(idx_int)
                    if len(selected) == self.num_initial_samples:
                        return selected
            max_draws *= 2

        for idx in range(self.candidate_positions.shape[0]):
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == self.num_initial_samples:
                    break
        return selected

    def measure_candidate(
        self,
        candidate_index: int,
        non_position_kwargs_for_acquisition_tool: dict[str, Any] | None = None,
    ) -> SAXSMeasurement:
        """Acquire and store one SAXS measurement at a candidate index."""
        self._require_sampling_configured()
        if candidate_index in self.measured_candidate_indices:
            raise ValueError(f"Candidate index {candidate_index} has already been measured.")
        position = self.candidate_positions[candidate_index]
        acquisition_kwargs = self._resolve_acquisition_kwargs(
            non_position_kwargs_for_acquisition_tool
        )
        q, intensity = self.acquire_saxs(
            float(position[0]), float(position[1]), acquisition_kwargs
        )
        spectrum = self.preprocess_spectrum(q, intensity)
        measurement = SAXSMeasurement(
            position=position.copy(),
            q=np.asarray(q, dtype=float),
            intensity=np.asarray(intensity, dtype=float),
            spectrum=spectrum,
        )
        self.measurements.append(measurement)
        self.measured_candidate_indices.append(candidate_index)
        message = (
            f"Measured SAXS at x={position[0]:.6g}, y={position[1]:.6g} "
            f"({len(self.measurements)}/{self.max_measurements})."
        )
        logger.info(message)
        self._record_progress_message(message)
        self.publish_saxs_spectra_visualization()
        return measurement

    def acquire_saxs(
        self,
        x: float,
        y: float,
        non_position_kwargs_for_acquisition_tool: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Call the acquisition tool and normalize its return payload."""
        acquisition_kwargs = self._resolve_acquisition_kwargs(
            non_position_kwargs_for_acquisition_tool
        )
        result = self.acquisition_tool.acquire_saxs(x=x, y=y, **acquisition_kwargs)
        if isinstance(result, dict):
            q = result.get("q")
            intensity = result.get("I", result.get("intensity"))
        elif isinstance(result, (tuple, list)) and len(result) == 2:
            q, intensity = result
        else:
            raise ValueError(
                "`acquire_saxs` must return `(q, I)` or a dict containing `q` "
                "and `I`/`intensity`."
            )
        return np.asarray(q, dtype=float), np.asarray(intensity, dtype=float)

    def preprocess_spectrum(self, q: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Interpolate one raw spectrum onto the common grid and log-transform it."""
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
            raise ValueError("Raw spectrum does not cover the full [q_min, q_max] range.")
        if np.any(intensity + self.epsilon_intensity <= 0):
            raise ValueError(
                "Intensity values must be greater than `-epsilon_intensity`."
            )
        interpolated = np.interp(self.q_grid, q, intensity)
        return np.log(interpolated + self.epsilon_intensity)

    def refit_model(self) -> None:
        """Refit PCA, latent standardization, and the joint multi-output GP."""
        self._require_sampling_configured()
        self._record_progress_message(
            f"Updating PCA and Gaussian process model with {len(self.measurements)} "
            "measurements."
        )
        spectra = torch.as_tensor(
            np.vstack([measurement.spectrum for measurement in self.measurements]),
            dtype=torch.double,
        )
        latent_scores = self.fit_pca(spectra, self.num_pca_components)
        standardized_latent_scores = self.standardize_latent_scores(
            latent_scores, self.epsilon_z
        )
        train_x = torch.as_tensor(
            self.normalize_positions(self.measured_positions),
            dtype=torch.double,
        )
        gp_model = self.fit_gp(
            train_x,
            standardized_latent_scores,
            self.num_pca_components,
        )
        self.standardized_latent_scores = standardized_latent_scores
        self.gp_model = gp_model
        self._record_progress_message("Gaussian process model update complete.")
        self.publish_posterior_visualization()

    def publish_posterior_visualization(self) -> None:
        """Publish the posterior status figure to the WebUI visualization tile."""
        runtime_controller = getattr(self, "runtime_controller", None)
        if runtime_controller is None:
            return
        fig = None
        try:
            fig = self.create_posterior_visualization()
            conversation_id = getattr(self, "runtime_conversation_id", "primary")
            if self.posterior_visualization_tile_id is None:
                self.posterior_visualization_tile_id = self._add_visualization_tile(
                    runtime_controller,
                    width=900,
                    height=680,
                    conversation_id=conversation_id,
                )
            self._update_visualization_tile(
                runtime_controller,
                tile_id=self.posterior_visualization_tile_id,
                figure=fig,
                conversation_id=conversation_id,
            )
        except Exception as exc:
            logger.warning("Failed to publish SAXS posterior visualization: %s", exc)
        finally:
            if fig is not None:
                plt.close(fig)

    def publish_saxs_spectra_visualization(self) -> None:
        """Publish collected SAXS spectra to the WebUI visualization tile."""
        runtime_controller = getattr(self, "runtime_controller", None)
        if runtime_controller is None or not self.measurements:
            return
        fig = None
        try:
            fig = self.create_saxs_spectra_visualization()
            conversation_id = getattr(self, "runtime_conversation_id", "primary")
            if self.saxs_spectra_visualization_tile_id is None:
                self.saxs_spectra_visualization_tile_id = self._add_visualization_tile(
                    runtime_controller,
                    width=576,
                    height=384,
                    conversation_id=conversation_id,
                )
            self._update_visualization_tile(
                runtime_controller,
                tile_id=self.saxs_spectra_visualization_tile_id,
                figure=fig,
                conversation_id=conversation_id,
            )
        except Exception as exc:
            logger.warning("Failed to publish SAXS spectra visualization: %s", exc)
        finally:
            if fig is not None:
                plt.close(fig)

    def create_saxs_spectra_visualization(self) -> "plt.Figure":
        """Return a log-log plot of all SAXS spectra collected so far."""
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4), constrained_layout=True)
        for measurement in self.measurements:
            q = np.asarray(measurement.q, dtype=float)
            intensity = np.asarray(measurement.intensity, dtype=float)
            positive = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
            if not np.any(positive):
                continue
            ax.loglog(q[positive], intensity[positive], linewidth=1.0, alpha=0.75)
        ax.set_title(f"Collected SAXS spectra ({len(self.measurements)} sampled points)")
        ax.set_xlabel("q")
        ax.set_ylabel("Intensity")
        ax.grid(True, which="both", alpha=0.25)
        return fig

    def create_posterior_visualization(self) -> "plt.Figure":
        """Return a figure with posterior and acquisition-term maps."""
        self._require_sampling_configured()
        if self.gp_model is None:
            raise ValueError("Gaussian process model must be fit before plotting.")
        candidate_x = torch.as_tensor(
            self.candidate_positions_normalized,
            dtype=torch.double,
        )
        with torch.no_grad():
            posterior = self.gp_model.posterior(candidate_x)
            mean = posterior.mean.detach().cpu().numpy()
            variance = posterior.variance.clamp_min(0.0)
            uncertainty = torch.sqrt(variance.mean(dim=-1)).detach().cpu().numpy()

        scores = self._get_visualization_acquisition_scores()
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        fig.suptitle(f"Posterior status after {len(self.measurements)} sampled points")
        panels: list[tuple[str, np.ndarray, np.ndarray | None]] = [
            ("Posterior uncertainty", self.candidate_positions, uncertainty),
            (
                "Log-det diversity",
                scores.positions if scores is not None else self.candidate_positions,
                scores.q_div_tilde if scores is not None else None,
            ),
            (
                "Gradient",
                scores.positions if scores is not None else self.candidate_positions,
                scores.gradient_tilde if scores is not None else None,
            ),
            (
                "PC1 posterior mean",
                self.candidate_positions,
                mean[:, 0] if mean.shape[1] >= 1 else None,
            ),
            (
                "PC2 posterior mean",
                self.candidate_positions,
                mean[:, 1] if mean.shape[1] >= 2 else None,
            ),
            (
                "PC3 posterior mean",
                self.candidate_positions,
                mean[:, 2] if mean.shape[1] >= 3 else None,
            ),
        ]
        sampled_positions = self.measured_positions
        latest_position = sampled_positions[-1] if sampled_positions.size else None
        for ax, (title, positions, values) in zip(axes.ravel(), panels):
            self._draw_spatial_panel(
                ax,
                positions=positions,
                values=values,
                title=title,
                sampled_positions=sampled_positions,
                latest_position=latest_position,
            )
        return fig

    def _get_visualization_acquisition_scores(self) -> AcquisitionScores | None:
        """Return current acquisition terms for visualization, if available."""
        try:
            return self.compute_acquisition_scores()
        except ValueError as exc:
            logger.debug("Skipping acquisition-term visualization: %s", exc)
            return None

    def _draw_spatial_panel(
        self,
        ax: Any,
        positions: np.ndarray,
        values: np.ndarray | None,
        title: str,
        sampled_positions: np.ndarray,
        latest_position: np.ndarray | None,
    ) -> None:
        """Draw one spatial scatter panel for the posterior status figure."""
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if values is None:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
            ax.set_xlim(self.position_min[0], self.position_max[0])
            ax.set_ylim(self.position_min[1], self.position_max[1])
        else:
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=values,
                cmap="viridis",
                s=70,
            )
            plt.colorbar(scatter, ax=ax)
        if sampled_positions.size:
            ax.scatter(
                sampled_positions[:, 0],
                sampled_positions[:, 1],
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                s=110,
                label="sampled",
            )
            ax.scatter(
                sampled_positions[:, 0],
                sampled_positions[:, 1],
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                s=118,
            )
        if latest_position is not None:
            ax.scatter(
                [latest_position[0]],
                [latest_position[1]],
                marker="*",
                c="red",
                edgecolors="black",
                linewidths=0.8,
                s=220,
                label="latest",
            )
        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def _add_visualization_tile(
        runtime_controller: Any,
        width: int,
        height: int,
        conversation_id: str,
    ) -> str:
        """Add a visualization tile using the runtime controller API."""
        add_tile = getattr(runtime_controller, "add_tile", None)
        if callable(add_tile):
            return add_tile(width, height, conversation_id=conversation_id)
        return runtime_controller.add_visualization_tile(
            width,
            height,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _update_visualization_tile(
        runtime_controller: Any,
        tile_id: str,
        figure: "plt.Figure",
        conversation_id: str,
    ) -> None:
        """Update a visualization tile using the runtime controller API."""
        update_tile = getattr(runtime_controller, "update_tile", None)
        if callable(update_tile):
            update_tile(tile_id, figure=figure, conversation_id=conversation_id)
            return
        runtime_controller.update_visualization_tile(
            tile_id,
            figure=figure,
            conversation_id=conversation_id,
        )

    def _record_progress_message(self, message: str) -> None:
        """Record a progress message locally and in the WebUI."""
        self.record_system_message(message, update_context=False)

    @staticmethod
    def fit_pca(
        spectra: "torch.Tensor", num_pca_components: int
    ) -> "torch.Tensor":
        """Fit PCA and project the spectra into its latent space.

        Parameters
        ----------
        spectra : torch.Tensor
            Preprocessed spectra with shape ``(N, N_q)``.
        num_pca_components : int
            Number of principal components to retain.

        Returns
        -------
        torch.Tensor
            PCA projections with shape ``(N, num_pca_components)``.
        """
        if spectra.shape[0] <= num_pca_components:
            raise ValueError("Need more measured spectra than PCA components.")
        pca_mean = spectra.mean(dim=0)
        centered = spectra - pca_mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        pca_components = vh[:num_pca_components]
        return centered @ pca_components.T

    @staticmethod
    def standardize_latent_scores(
        latent_scores: "torch.Tensor", epsilon_z: float
    ) -> "torch.Tensor":
        """Standardize PCA scores dimension-wise for GP training.

        Parameters
        ----------
        latent_scores : torch.Tensor
            PCA projections with shape ``(N, N_pcs)``.
        epsilon_z : float
            Stabilizing value added to each latent standard deviation.

        Returns
        -------
        torch.Tensor
            Standardized PCA projections with the same shape as ``latent_scores``.
        """
        latent_mean = latent_scores.mean(dim=0)
        latent_std = latent_scores.std(dim=0, unbiased=True)
        return (latent_scores - latent_mean) / (latent_std + epsilon_z)

    @staticmethod
    def fit_gp(
        train_x: "torch.Tensor",
        train_y: "torch.Tensor",
        num_pca_components: int,
    ) -> Any:
        """Fit the joint multi-output GP in normalized position space.

        Parameters
        ----------
        train_x : torch.Tensor
            Normalized measured positions with shape ``(N, 2)``.
        train_y : torch.Tensor
            Standardized PCA projections with shape ``(N, N_pcs)``.
        num_pca_components : int
            Number of modeled PCA components.

        Returns
        -------
        KroneckerMultiTaskGP
            Fitted multi-output Gaussian process model.
        """
        data_kernel = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=2),
        )
        gp_model = KroneckerMultiTaskGP(
            train_X=train_x,
            train_Y=train_y.double(),
            data_covar_module=data_kernel,
            rank=num_pca_components,
            outcome_transform=None,
        )
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_mll(mll)
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

    def suggest_next_positions(self, k: int = 1) -> np.ndarray:
        """Return the current top-k unsampled candidate positions."""
        self._require_sampling_configured()
        if k < 1:
            raise ValueError("`k` must be positive.")
        if self.gp_model is None:
            if not self.measurements:
                indices = self.get_initial_candidate_indices()[:k]
                return self.candidate_positions[indices].copy()
            self.refit_model()
        scores = self.compute_acquisition_scores()
        if k > scores.positions.shape[0]:
            raise ValueError("`k` cannot exceed the number of unsampled candidates.")
        order = np.lexsort((np.arange(scores.acquisition.size), -scores.acquisition))
        return scores.positions[order[:k]].copy()

    def compute_acquisition_scores(self) -> AcquisitionScores:
        """Compute acquisition scores over all unsampled candidates."""
        self._require_sampling_configured()
        unsampled_indices = self.get_unsampled_candidate_indices()
        if unsampled_indices.size == 0:
            raise ValueError("No unsampled candidate positions remain.")
        candidate_x = torch.as_tensor(
            self.candidate_positions_normalized[unsampled_indices],
            dtype=torch.double,
        )
        posterior = self.gp_model.posterior(candidate_x)
        mean = posterior.mean
        variance = posterior.variance.clamp_min(0.0)
        sigma = torch.sqrt(variance.mean(dim=-1))
        gradient = self.compute_latent_gradient_magnitude(candidate_x)
        q_div = self.compute_expected_logdet_diversity(posterior)

        sigma_tilde = self.normalize_tensor(sigma)
        gradient_tilde = self.normalize_tensor(gradient)
        q_div_tilde = self.normalize_tensor(q_div)
        acquisition = sigma_tilde * (
            self.w_d * q_div_tilde + self.w_g * gradient_tilde
        )
        scores = AcquisitionScores(
            positions=self.candidate_positions[unsampled_indices].copy(),
            acquisition=acquisition.detach().cpu().numpy(),
            sigma_tilde=sigma_tilde.detach().cpu().numpy(),
            q_div_tilde=q_div_tilde.detach().cpu().numpy(),
            gradient_tilde=gradient_tilde.detach().cpu().numpy(),
        )
        self.latest_scores = scores
        del mean
        return scores

    def compute_latent_gradient_magnitude(self, candidate_x: "torch.Tensor") -> "torch.Tensor":
        """Return aggregate GP posterior-mean gradient magnitude."""
        self._require_sampling_configured()
        x = candidate_x.clone().detach().requires_grad_(True)
        mean = self.gp_model.posterior(x).mean
        gradients = []
        for latent_idx in range(self.num_pca_components):
            grad = torch.autograd.grad(
                mean[:, latent_idx].sum(),
                x,
                retain_graph=True,
                create_graph=False,
            )[0]
            gradients.append(grad)
        stacked = torch.stack(gradients, dim=0)
        return torch.sqrt(torch.sum(stacked.pow(2), dim=(0, 2)).clamp_min(0.0))

    def compute_expected_logdet_diversity(self, posterior: Any) -> "torch.Tensor":
        """Estimate expected log-det latent diversity gain by Monte Carlo."""
        self._require_sampling_configured()
        samples = posterior.rsample(torch.Size([self.num_mc_samples]))
        current = self.standardized_latent_scores.double()
        return self.logdet_diversity_gain(current, samples).mean(dim=0)

    def logdet_diversity_gain(
        self,
        current: "torch.Tensor",
        candidates: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return log-det scatter gain for candidate latent samples.

        Parameters
        ----------
        current : torch.Tensor
            Current standardized latent vectors with shape
            ``(N_past, N_pcs)``.
        candidates : torch.Tensor
            Candidate latent vectors with shape ``(..., N_pcs)``.

        Returns
        -------
        torch.Tensor
            Diversity gains with shape matching ``candidates.shape[:-1]``.
        """
        self._require_sampling_configured()
        current = current.double()
        candidates = candidates.double()
        num_past = current.shape[0]
        current_mean = current.mean(dim=0)
        centered = current - current_mean
        scatter = centered.T @ centered
        eye = torch.eye(
            self.num_pca_components, dtype=current.dtype, device=current.device
        )
        base = self.lambda_logdet * eye + scatter
        chol = torch.linalg.cholesky(base)
        diff = candidates - current_mean
        flat_diff = diff.reshape(-1, self.num_pca_components)
        solved = torch.cholesky_solve(flat_diff.T, chol).T
        mahalanobis = (flat_diff * solved).sum(dim=-1)
        factor = num_past / (num_past + 1)
        gains = torch.log1p(factor * mahalanobis)
        return gains.reshape(candidates.shape[:-1])

    def normalize_tensor(self, values: "torch.Tensor") -> "torch.Tensor":
        """Min-max normalize a tensor with the configured epsilon."""
        self._require_sampling_configured()
        return (values - values.min()) / (values.max() - values.min() + self.epsilon)

    def get_candidate_index(self, position: np.ndarray) -> int:
        """Return the exact candidate index for a mesh position."""
        self._require_sampling_configured()
        position = np.asarray(position, dtype=float).reshape(1, 2)
        matches = np.where(np.all(np.isclose(self.candidate_positions, position), axis=1))[0]
        if matches.size != 1:
            raise ValueError(f"Position is not a unique candidate: {position.ravel()}.")
        return int(matches[0])
