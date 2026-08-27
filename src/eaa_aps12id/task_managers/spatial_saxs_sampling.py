from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import eaa_core.matplotlib_setup  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.tool.base import BaseTool
from eaa_core.tool.mcp_adapter import MCPRPCWrapper

from eaa_aps12id.tools.spatial_saxs_sampling import (
    AcquisitionScores,
    SAXSMeasurement,
    SpatialSAXSAdaptiveSamplingEngineTool,
)

logger = logging.getLogger(__name__)


class SpatialSAXSAdaptiveSamplingTaskManager(BaseTaskManager):
    """Coordinate adaptive spatial SAXS acquisition with a separate engine."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        memory_config: MemoryManagerConfig | None = None,
        acquisition_tool: BaseTool | MCPRPCWrapper | None = None,
        engine_tool: SpatialSAXSAdaptiveSamplingEngineTool | None = None,
        checkpoint_db_path: str | None = "checkpoint.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the adaptive SAXS workflow controller.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration forwarded to ``BaseTaskManager``.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to ``BaseTaskManager``.
        acquisition_tool : BaseTool or MCPRPCWrapper
            Object exposing ``acquire_saxs`` with ``x`` and ``y`` arguments.
        engine_tool : SpatialSAXSAdaptiveSamplingEngineTool, optional
            Active-learning engine. A new engine is created when omitted.
        checkpoint_db_path : str, optional
            Checkpoint database path forwarded to ``BaseTaskManager``.
        build : bool, optional
            Whether to build the base task manager.
        """
        if acquisition_tool is None:
            raise ValueError("`acquisition_tool` must be provided.")
        if not isinstance(acquisition_tool, (BaseTool, MCPRPCWrapper)):
            raise TypeError(
                "`acquisition_tool` must be an instance of BaseTool or MCPRPCWrapper."
            )
        if not hasattr(acquisition_tool, "acquire_saxs"):
            raise ValueError("`acquisition_tool` must expose `acquire_saxs`.")
        if engine_tool is None:
            engine_tool = SpatialSAXSAdaptiveSamplingEngineTool()
        if not isinstance(engine_tool, SpatialSAXSAdaptiveSamplingEngineTool):
            raise TypeError(
                "`engine_tool` must be a "
                "SpatialSAXSAdaptiveSamplingEngineTool instance."
            )

        self.acquisition_tool = acquisition_tool
        self.engine_tool = engine_tool
        self.max_measurements: int | None = None
        self.num_candidates_per_suggestion = 1
        self._workflow_config: dict[str, Any] | None = None
        self.posterior_visualization_tile_id: str | None = None
        self.saxs_spectra_visualization_tile_id: str | None = None

        tools: list[BaseTool] = [engine_tool]
        if isinstance(acquisition_tool, BaseTool):
            tools.insert(0, acquisition_tool)
        super().__init__(
            *args,
            llm_config=llm_config,
            memory_config=memory_config,
            tools=tools,
            checkpoint_db_path=checkpoint_db_path,
            build=build,
            **kwargs,
        )

    @property
    def measurements(self) -> list[SAXSMeasurement]:
        """Return measurements recorded by the active-learning engine."""
        return self.engine_tool.measurements

    @property
    def latest_scores(self) -> AcquisitionScores | None:
        """Return the engine's most recent acquisition scores."""
        return self.engine_tool.latest_scores

    def run(
        self,
        candidate_positions: (
            np.ndarray
            | list[list[float]]
            | list[tuple[float, float]]
            | tuple[tuple[float, float], ...]
            | str
            | Path
        ),
        q_min: float = 0.001,
        q_max: float = 1.0,
        num_q_points: int = 256,
        epsilon_intensity: float = 1e-12,
        num_initial_samples: int = 5,
        max_measurements: int = 20,
        exclusion_radius: float | None = None,
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
        n_iterations: int | None = None,
        non_position_kwargs_for_acquisition_tool: dict[str, Any] | None = None,
        suggestion_exclusion_radius: float | None = None,
        num_candidates_per_suggestion: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """Run the acquisition/update loop up to the configured budget.

        Parameters
        ----------
        candidate_positions : array-like, str, or pathlib.Path
            Candidate spatial coordinates as a non-empty numeric array with
            shape ``(N, 2)``. Each row is one candidate position; the first
            column is ``y`` and the second is ``x``. All values must be finite,
            and rows must be unique. Rows retain the supplied order. A string
            or ``pathlib.Path`` must point to a ``.npy`` file containing this
            array, such as one created by ``numpy.save``.
        max_measurements : int, optional
            Total acquisition budget, including initial measurements.
        n_iterations : int, optional
            Maximum adaptive measurements to add in this call after initial
            measurements. When omitted, exhaust the remaining budget.
        non_position_kwargs_for_acquisition_tool : dict, optional
            Extra keyword arguments passed to ``acquire_saxs``.
        num_candidates_per_suggestion : int, optional
            Positions acquired before each batched engine update.
        *args, **kwargs
            Ignored compatibility arguments.
        """
        del args, kwargs
        if (
            isinstance(num_candidates_per_suggestion, (bool, np.bool_))
            or not isinstance(num_candidates_per_suggestion, (int, np.integer))
            or num_candidates_per_suggestion < 1
        ):
            raise ValueError("`num_candidates_per_suggestion` must be positive.")
        self.num_candidates_per_suggestion = int(num_candidates_per_suggestion)

        if isinstance(candidate_positions, (str, Path)):
            candidate_positions = np.load(candidate_positions)

        self.engine_tool.initialize(
            candidate_positions=candidate_positions,
            q_min=q_min,
            q_max=q_max,
            num_q_points=num_q_points,
            epsilon_intensity=epsilon_intensity,
            num_initial_samples=num_initial_samples,
            exclusion_radius=exclusion_radius,
            suggestion_exclusion_radius=suggestion_exclusion_radius,
            background_smoothness=background_smoothness,
            background_max_iterations=background_max_iterations,
            background_tolerance=background_tolerance,
            background_valley_smoothing_sigma=background_valley_smoothing_sigma,
            background_valley_min_prominence=background_valley_min_prominence,
            peak_smoothing_sigma=peak_smoothing_sigma,
            peak_min_height=peak_min_height,
            peak_min_prominence=peak_min_prominence,
            peak_min_width_log_q=peak_min_width_log_q,
            peak_max_width_log_q=peak_max_width_log_q,
            peak_window_width_factor=peak_window_width_factor,
            num_initial_peaks=num_initial_peaks,
            max_peaks_in_dict=max_peaks_in_dict,
            known_peak_q_values=known_peak_q_values,
            new_peak_min_relative_area=new_peak_min_relative_area,
            peak_map_min_concentration=peak_map_min_concentration,
            peak_observable=peak_observable,
            peak_observale_map_blur=peak_observale_map_blur,
            peak_area_scale=peak_area_scale,
            exploration_interval=exploration_interval,
            max_fit_gp_mll_iterations=max_fit_gp_mll_iterations,
            w_peak=w_peak,
            w_g=w_g,
            epsilon_acquisition=epsilon_acquisition,
            epsilon_normalization=epsilon_normalization,
            epsilon_z=epsilon_z,
            normalization_lower_percentile=normalization_lower_percentile,
            normalization_upper_percentile=normalization_upper_percentile,
            random_seed=random_seed,
        )
        self._configure_workflow(max_measurements)
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
        n_steps = (
            remaining_budget
            if n_iterations is None
            else min(int(n_iterations), remaining_budget)
        )
        completed_steps = 0
        while completed_steps < n_steps:
            suggestion_size = min(
                self.num_candidates_per_suggestion,
                n_steps - completed_steps,
            )
            positions = self.engine_tool.suggest(n_suggestions=suggestion_size)
            self._acquire_and_update(positions, acquisition_kwargs)
            completed_steps += len(positions)
            if len(self.measurements) >= self.max_measurements:
                break
        self._record_progress_message(
            "Adaptive SAXS sampling complete for this run with "
            f"{len(self.measurements)}/{self.max_measurements} measurements."
        )

    def _configure_workflow(self, max_measurements: int) -> None:
        """Validate and retain manager-owned workflow configuration."""
        max_measurements = int(max_measurements)
        config = {"max_measurements": max_measurements}
        if self.measurements and self._workflow_config not in (None, config):
            raise ValueError(
                "Cannot change `max_measurements` after measurements have been "
                "collected. Create a new task manager for a new run."
            )
        if not (
            self.engine_tool.num_initial_samples
            <= max_measurements
            <= self.engine_tool.candidate_positions.shape[0]
        ):
            raise ValueError(
                "Expected `num_initial_samples <= max_measurements <= "
                "number of candidate positions`."
            )
        self._workflow_config = config
        self.max_measurements = max_measurements

    def _resolve_acquisition_kwargs(
        self,
        non_position_kwargs_for_acquisition_tool: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return non-position keyword arguments for ``acquire_saxs``."""
        if non_position_kwargs_for_acquisition_tool is None:
            return {
                "q_min": self.engine_tool.q_min,
                "q_max": self.engine_tool.q_max,
            }
        return dict(non_position_kwargs_for_acquisition_tool)

    def collect_initial_measurements(
        self,
        non_position_kwargs_for_acquisition_tool: dict[str, Any] | None = None,
    ) -> None:
        """Acquire and update the engine with its optimized initial batch."""
        acquisition_kwargs = self._resolve_acquisition_kwargs(
            non_position_kwargs_for_acquisition_tool
        )
        self._record_progress_message(
            f"Collecting {self.engine_tool.num_initial_samples} initial SAXS "
            "measurements."
        )
        positions = self.engine_tool.suggest_initial_measurements()
        self._acquire_and_update(positions, acquisition_kwargs)

    def _acquire_and_update(
        self,
        positions: np.ndarray,
        acquisition_kwargs: dict[str, Any],
    ) -> None:
        """Acquire an ordered position batch and update the engine once."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        q_values: list[np.ndarray] = []
        intensities: list[np.ndarray] = []
        starting_count = len(self.measurements)
        for offset, position in enumerate(positions, start=1):
            self._record_progress_message(
                f"Selected next SAXS position x={position[1]:.6g}, y={position[0]:.6g}."
            )
            q, intensity = self.acquire_saxs(
                float(position[1]),
                float(position[0]),
                acquisition_kwargs,
            )
            q_values.append(q)
            intensities.append(intensity)
            message = (
                f"Measured SAXS at x={position[1]:.6g}, y={position[0]:.6g} "
                f"({starting_count + offset}/{self.max_measurements})."
            )
            logger.info(message)
            self._record_progress_message(message)

        self._record_progress_message(
            f"Updating peak-{self.engine_tool.peak_observable} Gaussian process "
            f"models with {starting_count + len(positions)} measurements."
        )
        self.engine_tool.update(positions, q_values, intensities)
        self._record_progress_message("Gaussian process model update complete.")
        self.publish_saxs_spectra_visualization()
        self.publish_posterior_visualization()

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

    def publish_posterior_visualization(self) -> None:
        """Publish the posterior figure to the WebUI visualization tile."""
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
        except Exception as exc:  # noqa: BLE001 - visualization is best effort
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
        except Exception as exc:  # noqa: BLE001 - visualization is best effort
            logger.warning("Failed to publish SAXS spectra visualization: %s", exc)
        finally:
            if fig is not None:
                plt.close(fig)

    def create_saxs_spectra_visualization(self) -> plt.Figure:
        """Return a log-log plot of all SAXS spectra collected so far."""
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4), constrained_layout=True)
        for measurement in self.measurements:
            q = np.asarray(measurement.q, dtype=float)
            intensity = np.asarray(measurement.intensity, dtype=float)
            positive = (
                np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
            )
            if np.any(positive):
                ax.loglog(
                    q[positive],
                    intensity[positive],
                    linewidth=1.0,
                    alpha=0.75,
                )
        ax.set_title(
            f"Collected SAXS spectra ({len(self.measurements)} sampled points)"
        )
        ax.set_xlabel("q")
        ax.set_ylabel("Intensity")
        ax.grid(True, which="both", alpha=0.25)
        return fig

    def create_posterior_visualization(self) -> plt.Figure:
        """Return a figure with posterior and acquisition-term maps."""
        engine = self.engine_tool
        if engine.gp_model is None:
            raise ValueError("Gaussian process model must be fit before plotting.")
        import torch

        candidate_x = torch.as_tensor(
            engine.candidate_positions_normalized,
            dtype=torch.double,
        )
        with torch.no_grad():
            posterior = engine.gp_model.posterior(candidate_x)
            peak_observable_mean = (
                engine.inverse_transform_peak_scores(posterior.mean)
                .detach()
                .cpu()
                .numpy()
            )
            variance = posterior.variance.clamp_min(0.0)
            uncertainty = torch.sqrt(variance.mean(dim=-1)).detach().cpu().numpy()

        scores = self._get_visualization_acquisition_scores()
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        fig.suptitle(f"Posterior status after {len(self.measurements)} sampled points")
        panels: list[tuple[str, np.ndarray, np.ndarray | None]] = [
            ("Posterior uncertainty", engine.candidate_positions, uncertainty),
            (
                f"Normalized maximum peak {engine.peak_observable}",
                scores.positions if scores is not None else engine.candidate_positions,
                scores.peak_observable_tilde if scores is not None else None,
            ),
            (
                "Gradient",
                scores.positions if scores is not None else engine.candidate_positions,
                scores.gradient_tilde if scores is not None else None,
            ),
        ]
        for peak_index in range(3):
            if peak_index < len(engine.modeled_peak_ids):
                peak = engine.peak_dict[engine.modeled_peak_ids[peak_index]]
                panels.append(
                    (
                        f"Peak at q={peak.q_position:.4g}",
                        engine.candidate_positions,
                        peak_observable_mean[:, peak_index],
                    )
                )
            else:
                panels.append(
                    (f"Peak {peak_index + 1}", engine.candidate_positions, None)
                )
        sampled_positions = engine.measured_positions
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
            return self.engine_tool.compute_acquisition_scores()
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
        engine = self.engine_tool
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if values is None:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
            ax.set_xlim(engine.position_min[1], engine.position_max[1])
            ax.set_ylim(engine.position_min[0], engine.position_max[0])
        else:
            scatter = ax.scatter(
                positions[:, 1],
                positions[:, 0],
                c=values,
                cmap="viridis",
                s=70,
            )
            plt.colorbar(scatter, ax=ax)
        if sampled_positions.size:
            ax.scatter(
                sampled_positions[:, 1],
                sampled_positions[:, 0],
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                s=110,
                label="sampled",
            )
            ax.scatter(
                sampled_positions[:, 1],
                sampled_positions[:, 0],
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                s=118,
            )
        if latest_position is not None:
            ax.scatter(
                [latest_position[1]],
                [latest_position[0]],
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
        figure: plt.Figure,
        conversation_id: str,
    ) -> None:
        """Update a visualization tile using the runtime controller API."""
        update_tile = getattr(runtime_controller, "update_tile", None)
        if callable(update_tile):
            update_tile(
                tile_id,
                figure=figure,
                conversation_id=conversation_id,
            )
            return
        runtime_controller.update_visualization_tile(
            tile_id,
            figure=figure,
            conversation_id=conversation_id,
        )

    def _record_progress_message(self, message: str) -> None:
        """Record a progress message locally and in the WebUI."""
        self.record_system_message(message, update_context=False)
