import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from eaa_core.gui.runtime import WebUIRuntimeController
from eaa_core.tool.base import BaseTool
from scipy.ndimage import gaussian_filter1d

from eaa_aps12id.task_managers.spatial_saxs_sampling import (
    SpatialSAXSAdaptiveSamplingTaskManager,
)
from eaa_aps12id.tools.spatial_saxs_sampling import (
    DetectedSAXSPeak,
    SAXSMeasurement,
    SAXSPeak,
    SpatialSAXSAdaptiveSamplingEngineTool,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("botorch") is None
    or importlib.util.find_spec("gpytorch") is None,
    reason="torch, botorch, and gpytorch are required for spatial SAXS tests",
)


class DummySAXSTool(BaseTool):
    def __init__(self):
        self.calls = []

    def acquire_saxs(self, x, y, q_min, q_max, q_step=None, exposure=None):
        self.calls.append(
            {
                "x": x,
                "y": y,
                "q_min": q_min,
                "q_max": q_max,
                "q_step": q_step,
                "exposure": exposure,
            }
        )
        q = np.geomspace(q_min, q_max, 64)
        intensity = 1.0 + 0.2 * x + 0.1 * y + np.exp(-((np.log(q) + 2.0) ** 2))
        return q, intensity


def default_sampling_kwargs(**kwargs):
    params = {
        "candidate_positions": [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        "q_min": 0.01,
        "q_max": 1.0,
        "num_q_points": 16,
        "num_initial_samples": 3,
        "max_measurements": 5,
        "background_smoothness": 1e3,
        "peak_min_height": 0.0,
        "peak_min_prominence": 0.0,
        "num_initial_peaks": 1,
        "max_peaks_in_dict": 2,
        "exploration_interval": None,
        "random_seed": 1,
    }
    params.update(kwargs)
    return params


def make_manager(**kwargs):
    params = {
        "acquisition_tool": DummySAXSTool(),
        "checkpoint_db_path": None,
        "build": False,
    }
    params.update(kwargs)
    return SpatialSAXSAdaptiveSamplingTaskManager(**params)


def configure_manager(manager, **kwargs):
    params = default_sampling_kwargs(**kwargs)
    params.pop("max_measurements")
    manager.engine_tool.initialize(**params)
    return manager.engine_tool


def record_candidate(engine, candidate_index):
    """Record a dummy spectrum without triggering an engine refit."""
    position = engine.candidate_positions[candidate_index]
    q, intensity = DummySAXSTool().acquire_saxs(
        x=float(position[1]),
        y=float(position[0]),
        q_min=engine.q_min,
        q_max=engine.q_max,
    )
    spectrum, background = engine._preprocess_spectrum_with_background(
        q, intensity
    )
    engine.measurements.append(
        SAXSMeasurement(
            position=position.copy(),
            q=q,
            intensity=intensity,
            spectrum=spectrum,
            background=background,
        )
    )
    engine.measured_candidate_indices.append(candidate_index)


def test_engine_exposes_standalone_interface_without_acquisition_tool():
    engine = SpatialSAXSAdaptiveSamplingEngineTool()

    assert isinstance(engine, BaseTool)
    assert not hasattr(engine, "acquisition_tool")
    exposed_names = {spec.name for spec in engine.exposed_tools}
    assert {
        "spatial_saxs_adaptive_sampling_engine.initialize",
        "spatial_saxs_adaptive_sampling_engine.suggest_initial_measurements",
        "spatial_saxs_adaptive_sampling_engine.suggest",
        "spatial_saxs_adaptive_sampling_engine.update",
    } <= exposed_names


def test_manager_accepts_injected_engine():
    engine = SpatialSAXSAdaptiveSamplingEngineTool()

    manager = make_manager(engine_tool=engine)

    assert manager.engine_tool is engine


def test_suggest_initial_measurements_returns_engine_optimized_path(monkeypatch):
    engine = configure_manager(make_manager())
    monkeypatch.setattr(engine, "get_initial_candidate_indices", lambda: [0, 1, 2])
    monkeypatch.setattr(
        engine,
        "_optimize_suggestion_path",
        lambda positions: positions[::-1].copy(),
    )

    suggestions = engine.suggest_initial_measurements()

    np.testing.assert_array_equal(
        suggestions,
        engine.candidate_positions[[2, 1, 0]],
    )


def test_engine_update_accepts_variable_length_spectrum_batch(monkeypatch):
    engine = configure_manager(make_manager(), num_q_points=64)
    refit_calls = []
    monkeypatch.setattr(engine, "refit_model", lambda: refit_calls.append(True))
    positions = engine.candidate_positions[:2]
    q_values = [
        np.geomspace(engine.q_min, engine.q_max, 32),
        np.geomspace(engine.q_min, engine.q_max, 48),
    ]
    intensities = [np.ones(32), np.full(48, 2.0)]

    engine.update(positions, q_values, intensities)

    assert len(engine.measurements) == 2
    assert [measurement.q.size for measurement in engine.measurements] == [32, 48]
    assert refit_calls == [True]
    with pytest.raises(ValueError, match="already been measured"):
        engine.update(positions[0], q_values[0], intensities[0])


def test_adaptive_suggest_requires_model_update():
    engine = configure_manager(make_manager())

    with pytest.raises(ValueError, match="suggest_initial_measurements"):
        engine.suggest()


def test_preprocess_spectrum_interpolates_and_subtracts_background(monkeypatch):
    manager = configure_manager(make_manager())
    q = np.geomspace(0.005, 1.5, 8)
    intensity = q * 2.0
    monkeypatch.setattr(
        manager,
        "fit_valley_background",
        lambda values, **kwargs: (
            np.full_like(values, np.log(0.01)),
            np.asarray([1, 2]),
        ),
    )

    spectrum = manager.preprocess_spectrum(q[::-1], intensity[::-1])

    expected = np.interp(manager.q_grid, q, intensity) - (
        0.01 - manager.epsilon_intensity
    )
    np.testing.assert_allclose(spectrum, expected)


def test_preprocess_spectrum_rejects_insufficient_q_coverage():
    manager = configure_manager(make_manager())
    q = np.geomspace(0.02, 0.8, 6)
    intensity = np.ones_like(q)

    with pytest.raises(ValueError, match="does not cover"):
        manager.preprocess_spectrum(q, intensity)


def test_initial_sobol_selection_is_unique_and_deterministic():
    manager_a = configure_manager(make_manager(), random_seed=10)
    manager_b = configure_manager(make_manager(), random_seed=10)

    indices_a = manager_a.get_initial_candidate_indices()
    indices_b = manager_b.get_initial_candidate_indices()

    assert indices_a == indices_b
    assert len(indices_a) == manager_a.num_initial_samples
    assert len(set(indices_a)) == manager_a.num_initial_samples


def test_initial_selection_respects_suggestion_exclusion_radius():
    manager = configure_manager(
        make_manager(),
        suggestion_exclusion_radius=1.01,
    )

    indices = manager.get_initial_candidate_indices()
    positions = manager.candidate_positions[indices]
    pairwise_distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    )

    assert np.all(
        pairwise_distances[np.triu_indices(len(indices), k=1)]
        > manager.suggestion_exclusion_radius
    )


def test_arpls_background_is_not_pulled_to_narrow_peak():
    x = np.linspace(0.0, 1.0, 200)
    background = 1.0 + 0.2 * x
    values = background + 3.0 * np.exp(-((x - 0.5) / 0.025) ** 2)

    fitted = SpatialSAXSAdaptiveSamplingEngineTool.fit_arpls_background(
        values,
        smoothness=1e5,
        max_iterations=50,
        tolerance=1e-4,
    )

    np.testing.assert_allclose(fitted, background, atol=0.08)


def test_valley_background_interpolates_between_peak_separating_valleys():
    x = np.linspace(0.0, 1.0, 256)
    intensity = np.ones_like(x)
    for center in (0.2, 0.5, 0.8):
        intensity += 2.0 * np.exp(-0.5 * ((x - center) / 0.035) ** 2)

    fitted, valley_indices = (
        SpatialSAXSAdaptiveSamplingEngineTool.fit_valley_background(
            np.log(intensity),
            smoothing_sigma=2.0,
            min_prominence=0.05,
        )
    )

    assert fitted is not None
    assert valley_indices.size == 2
    for center in (0.2, 0.5, 0.8):
        peak_index = int(np.argmin(np.abs(x - center)))
        assert fitted[peak_index] < np.log(intensity[peak_index]) - 0.5


def test_valley_background_defers_to_fallback_with_fewer_than_two_valleys():
    fitted, valley_indices = (
        SpatialSAXSAdaptiveSamplingEngineTool.fit_valley_background(
            np.linspace(1.0, 0.0, 64),
            smoothing_sigma=2.0,
            min_prominence=0.05,
        )
    )

    assert fitted is None
    assert valley_indices.size == 0


def test_preprocess_uses_smoothed_spectrum_when_valleys_are_insufficient():
    manager = configure_manager(
        make_manager(),
        num_q_points=64,
        background_valley_smoothing_sigma=2.0,
    )
    q = manager.q_grid.copy()
    q[0] = manager.q_min
    q[-1] = manager.q_max
    intensity = 2.0 * q**-0.5

    _, fitted = manager._preprocess_spectrum_with_background(
        q,
        intensity,
    )

    interpolated = np.interp(manager.q_grid, q, intensity)
    expected_log_background = gaussian_filter1d(
        np.log(interpolated + manager.epsilon_intensity),
        sigma=manager.background_valley_smoothing_sigma,
        mode="nearest",
    )
    expected = np.exp(expected_log_background) - manager.epsilon_intensity
    np.testing.assert_allclose(fitted, expected)


def test_detect_peaks_uses_width_derived_area():
    manager = configure_manager(
        make_manager(),
        num_q_points=256,
        peak_min_height=1.0,
        peak_min_prominence=3.0,
    )
    log_q = np.log(manager.q_grid)
    spectrum = np.exp(-0.5 * ((log_q - np.log(0.2)) / 0.04) ** 2)

    peaks = manager.detect_peaks(spectrum)

    assert len(peaks) == 1
    assert peaks[0].q_left < peaks[0].q_position < peaks[0].q_right
    assert peaks[0].integrated_area > 0


def test_detect_peaks_measures_width_on_linear_corrected_intensity():
    manager = configure_manager(
        make_manager(),
        num_q_points=256,
        peak_min_height=1.0,
        peak_min_prominence=1.0,
    )
    log_q = np.log(manager.q_grid)
    spectrum = 0.6 + 10.0 * np.exp(
        -0.5 * ((log_q - np.log(0.2)) / 0.05) ** 2
    )

    peaks = manager.detect_peaks(spectrum)

    assert len(peaks) == 1
    assert peaks[0].width_log_q < 0.2


def test_default_minimum_width_rejects_narrow_noise_peak():
    manager = configure_manager(
        make_manager(),
        num_q_points=512,
        peak_min_height=1.0,
        peak_min_prominence=1.0,
    )
    log_q = np.log(manager.q_grid)
    spectrum = np.exp(
        -0.5 * ((log_q - np.log(0.2)) / 0.005) ** 2
    )

    peaks = manager.detect_peaks(spectrum)

    assert manager.peak_min_width_log_q == 0.03
    assert peaks == []


def test_log_scale_detection_rejects_small_relative_ripple():
    manager = configure_manager(
        make_manager(),
        num_q_points=256,
        peak_min_height=1.0,
        peak_min_prominence=1.0,
    )
    log_q = np.log(manager.q_grid)
    spectrum = 0.6 + 0.005 * np.exp(
        -0.5 * ((log_q - np.log(0.03)) / 0.02) ** 2
    )

    peaks = manager.detect_peaks(spectrum)

    assert peaks == []


def test_known_peak_dictionary_is_authoritative_and_updates_widths(
    monkeypatch,
):
    manager = configure_manager(
        make_manager(),
        num_q_points=256,
        known_peak_q_values=[0.2, 0.6],
    )
    manager.measurements = [
        SimpleNamespace(spectrum=np.ones(manager.num_q_points))
    ]
    initial_detections = [
        DetectedSAXSPeak(
            q_position=0.21,
            log_q_position=np.log(0.21),
            q_left=0.15,
            q_right=0.27,
            width_log_q=0.12,
            height=8.0,
            prominence=7.0,
            integrated_area=1.0,
        ),
        DetectedSAXSPeak(
            q_position=0.4,
            log_q_position=np.log(0.4),
            q_left=0.35,
            q_right=0.45,
            width_log_q=0.05,
            height=8.0,
            prominence=7.0,
            integrated_area=10.0,
        ),
    ]
    monkeypatch.setattr(
        manager,
        "detect_peaks",
        lambda spectrum: initial_detections,
    )

    manager.update_peak_dictionary()

    assert list(manager.peak_dict) == [0, 1]
    np.testing.assert_array_equal(
        [peak.q_position for peak in manager.peak_dict.values()],
        np.array([0.2, 0.6]),
    )
    assert manager.peak_dict[0].width_log_q == 0.12
    assert manager.peak_dict[1].width_log_q == 0.0
    assert manager.get_peak_integrated_areas()[0, 1] == 0.0

    manager.measurements.append(
        SimpleNamespace(spectrum=np.full(manager.num_q_points, 2.0))
    )
    later_detection = DetectedSAXSPeak(
        q_position=0.62,
        log_q_position=np.log(0.62),
        q_left=0.5,
        q_right=0.72,
        width_log_q=0.1,
        height=8.0,
        prominence=7.0,
        integrated_area=2.0,
    )
    monkeypatch.setattr(
        manager,
        "detect_peaks",
        lambda spectrum: [later_detection],
    )

    manager.update_peak_dictionary()

    assert list(manager.peak_dict) == [0, 1]
    assert manager.peak_dict[1].width_log_q == 0.1
    assert manager.peak_dict[1].discovery_measurement_index == 1
    assert np.all(manager.get_peak_integrated_areas()[:, 1] > 0)


def test_unmatched_known_peak_height_uses_intensity_at_q(monkeypatch):
    known_q = 0.2
    manager = configure_manager(
        make_manager(),
        known_peak_q_values=[known_q],
        peak_observable="height",
    )
    spectrum = 1.0 + 2.0 * manager.q_grid
    manager.measurements = [SimpleNamespace(spectrum=spectrum)]
    monkeypatch.setattr(manager, "detect_peaks", lambda spectrum: [])

    manager.update_peak_dictionary()

    np.testing.assert_allclose(
        manager.get_peak_observables(),
        np.array([[1.0 + 2.0 * known_q]]),
    )
    assert manager.peak_dict[0].width_log_q == 0.0


def test_inverse_transform_peak_scores_returns_physical_observables():
    manager = configure_manager(make_manager(), peak_area_scale=2.0)

    import torch

    manager.peak_score_mean = torch.tensor([1.0, 2.0], dtype=torch.double)
    manager.peak_score_std = torch.tensor([0.5, 0.25], dtype=torch.double)
    standardized = torch.tensor([[2.0, -4.0]], dtype=torch.double)

    observables = manager.inverse_transform_peak_scores(standardized)

    expected = 2.0 * np.expm1([2.0, 1.0])
    np.testing.assert_allclose(
        observables.detach().cpu().numpy()[0],
        expected,
    )


def test_predicted_peak_acquisition_uses_pointwise_maximum(monkeypatch):
    manager = configure_manager(make_manager())

    import torch

    predicted_observables = torch.tensor(
        [[1.0, 4.0, 2.0], [3.0, 2.0, 1.0]],
        dtype=torch.double,
    )
    monkeypatch.setattr(
        manager,
        "inverse_transform_peak_scores",
        lambda standardized_mean: predicted_observables,
    )

    maximum = manager.compute_predicted_max_peak_observable(
        torch.zeros_like(predicted_observables)
    )

    np.testing.assert_array_equal(
        maximum.detach().cpu().numpy(),
        np.array([4.0, 3.0]),
    )


def test_peak_map_score_rejects_uniform_map_and_normalizes_per_peak(monkeypatch):
    manager = configure_manager(make_manager())

    import torch

    localized = torch.cat(
        (
            torch.zeros(90, dtype=torch.double),
            torch.ones(10, dtype=torch.double),
        )
    )
    uniform = torch.ones(100, dtype=torch.double)
    predicted_observables = torch.stack((localized, uniform), dim=-1)
    monkeypatch.setattr(
        manager,
        "inverse_transform_peak_scores",
        lambda standardized_mean: predicted_observables,
    )

    score = manager.compute_concentration_gated_peak_score(
        torch.zeros_like(predicted_observables)
    )

    assert manager.peak_map_min_concentration == 0.2
    np.testing.assert_allclose(
        score.detach().cpu().numpy(),
        localized.detach().cpu().numpy(),
    )


def test_peak_map_score_is_zero_when_all_maps_are_uniform(monkeypatch):
    manager = configure_manager(make_manager())

    import torch

    predicted_observables = torch.ones(100, 2, dtype=torch.double)
    monkeypatch.setattr(
        manager,
        "inverse_transform_peak_scores",
        lambda standardized_mean: predicted_observables,
    )

    score = manager.compute_concentration_gated_peak_score(
        torch.zeros_like(predicted_observables)
    )

    np.testing.assert_array_equal(
        score.detach().cpu().numpy(),
        np.zeros(100),
    )


def test_peak_observable_map_blur_uses_physical_axis_units():
    candidate_positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [0.5, 0.0],
            [0.5, 2.0],
            [1.0, 4.0],
        ]
    )
    manager = configure_manager(
        make_manager(),
        candidate_positions=candidate_positions,
        peak_observale_map_blur=1.0,
    )

    import torch

    observables = torch.zeros(5, 1, dtype=torch.double)
    observables[2, 0] = 1.0

    blurred = manager.blur_peak_observable_maps(observables)

    squared_distances = np.sum(
        (candidate_positions[:, None] - candidate_positions[None, :]) ** 2,
        axis=-1,
    )
    weights = np.exp(-squared_distances / 2.0)
    weights[squared_distances > 4.0**2] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    expected = weights @ observables.numpy()
    np.testing.assert_allclose(
        blurred.detach().cpu().numpy(),
        expected,
    )
    cached_weight_matrix = manager._peak_blur_weight_matrix

    manager.blur_peak_observable_maps(observables)

    assert manager._peak_blur_weight_matrix is cached_weight_matrix
    assert cached_weight_matrix._nnz() < candidate_positions.shape[0] ** 2


def test_peak_height_observable_is_maximum_in_frozen_interval():
    manager = configure_manager(make_manager(), peak_observable="height")
    spectrum = np.zeros(manager.num_q_points)
    spectrum[4:8] = [-1.0, 2.0, 5.0, 3.0]
    manager.measurements = [SimpleNamespace(spectrum=spectrum)]
    manager.peak_dict = {
        0: SAXSPeak(
            0,
            manager.q_grid[6],
            np.log(manager.q_grid[6]),
            manager.q_grid[4],
            manager.q_grid[7],
            0.1,
            1.0,
            0,
        )
    }

    observables = manager.get_peak_observables()

    assert manager.peak_observable == "height"
    np.testing.assert_array_equal(observables, np.array([[5.0]]))


def test_height_observable_is_used_for_gp_targets(monkeypatch):
    manager = configure_manager(make_manager(), peak_observable="height")
    manager.measurements = [
        SimpleNamespace(
            position=np.array([float(index), 0.0]),
            spectrum=np.eye(3, manager.num_q_points)[index] * height,
        )
        for index, height in enumerate((1.0, 3.0, 7.0))
    ]
    manager.peak_dict = {
        0: SAXSPeak(
            0,
            manager.q_grid[1],
            np.log(manager.q_grid[1]),
            manager.q_grid[0],
            manager.q_grid[2],
            0.1,
            1.0,
            0,
        )
    }
    fitted_targets = None

    def fake_fit_gp(
        train_x,
        train_y,
        max_fit_gp_mll_iterations=None,
        fit_mll=True,
    ):
        nonlocal fitted_targets
        fitted_targets = train_y
        return object()

    monkeypatch.setattr(manager, "fit_gp", fake_fit_gp)

    standardized, _, _, _, _ = manager._fit_model(
        [0, 1, 2],
        fit_mll=False,
    )

    expected_scores = np.log1p([1.0, 3.0, 7.0])
    expected = (
        expected_scores - expected_scores.mean()
    ) / expected_scores.std()
    np.testing.assert_allclose(
        standardized.detach().cpu().numpy()[:, 0],
        expected,
    )
    assert fitted_targets is standardized


def test_new_peak_evicts_dictionary_entry_with_smallest_maximum_area(monkeypatch):
    manager = configure_manager(
        make_manager(),
        num_initial_peaks=1,
        max_peaks_in_dict=2,
    )
    manager.peak_dict = {
        0: SAXSPeak(0, 0.1, np.log(0.1), 0.09, 0.11, 0.02, 1.0, 0),
        1: SAXSPeak(1, 0.3, np.log(0.3), 0.28, 0.32, 0.04, 10.0, 0),
    }
    manager._next_peak_id = 2
    manager._peak_detection_measurement_count = 1
    manager.measurements = [
        SimpleNamespace(spectrum=np.zeros(manager.num_q_points)),
        SimpleNamespace(spectrum=np.zeros(manager.num_q_points)),
    ]
    area_by_left = {0.09: 1.0, 0.28: 10.0, 0.48: 5.0}
    monkeypatch.setattr(
        manager,
        "integrate_peak_area",
        lambda spectrum, q_left, q_right: area_by_left[round(q_left, 2)],
    )
    monkeypatch.setattr(
        manager,
        "detect_peaks",
        lambda spectrum: [
            DetectedSAXSPeak(
                q_position=0.5,
                log_q_position=np.log(0.5),
                q_left=0.48,
                q_right=0.52,
                width_log_q=0.04,
                height=8.0,
                prominence=7.0,
                integrated_area=5.0,
            )
        ],
    )

    manager.update_peak_dictionary()

    assert set(manager.peak_dict) == {1, 2}
    assert manager.peak_dict[2].max_integrated_area == 5.0


def test_new_peak_below_relative_area_gate_is_not_admitted(monkeypatch):
    manager = configure_manager(make_manager())
    manager.peak_dict = {
        0: SAXSPeak(0, 0.1, np.log(0.1), 0.09, 0.11, 0.02, 10.0, 0),
    }
    manager._next_peak_id = 1
    manager._peak_detection_measurement_count = 1
    manager.measurements = [
        SimpleNamespace(spectrum=np.zeros(manager.num_q_points)),
        SimpleNamespace(spectrum=np.zeros(manager.num_q_points)),
    ]
    monkeypatch.setattr(
        manager,
        "integrate_peak_area",
        lambda spectrum, q_left, q_right: 10.0,
    )
    monkeypatch.setattr(
        manager,
        "detect_peaks",
        lambda spectrum: [
            DetectedSAXSPeak(
                q_position=0.5,
                log_q_position=np.log(0.5),
                q_left=0.48,
                q_right=0.52,
                width_log_q=0.04,
                height=8.0,
                prominence=7.0,
                integrated_area=0.005,
            )
        ],
    )

    manager.update_peak_dictionary()

    assert manager.new_peak_min_relative_area == 0.001
    assert set(manager.peak_dict) == {0}
    assert manager._next_peak_id == 1


def test_scheduled_exploration_selects_farthest_unsampled_position():
    manager = configure_manager(
        make_manager(),
        num_initial_samples=1,
        exploration_interval=2,
    )
    manager.measurements = [SimpleNamespace(), SimpleNamespace()]
    manager.measured_candidate_indices = [4]

    assert manager._is_farthest_exploration_step()
    np.testing.assert_array_equal(
        manager.get_farthest_unsampled_position(),
        np.array([0.0, 0.0]),
    )


def test_batch_crossing_exploration_interval_includes_farthest_position():
    manager = configure_manager(
        make_manager(),
        num_initial_samples=1,
        exploration_interval=5,
    )
    manager.measurements = [SimpleNamespace() for _ in range(4)]
    manager.measured_candidate_indices = [4]
    manager.gp_model = object()
    manager.compute_acquisition_scores = lambda: SimpleNamespace(
        positions=np.array([[2.0, 2.0], [2.0, 1.0], [0.0, 2.0]]),
        acquisition=np.array([3.0, 2.0, 1.0]),
    )

    suggestions = manager.suggest(n_suggestions=2)

    assert manager._batch_includes_farthest_exploration(2)
    assert {tuple(position) for position in suggestions} == {
        (0.0, 0.0),
        (2.0, 2.0),
    }


def test_farthest_position_participates_in_suggestion_exclusion():
    manager = configure_manager(
        make_manager(),
        num_initial_samples=1,
        exploration_interval=5,
        suggestion_exclusion_radius=1.01,
    )
    manager.measurements = [SimpleNamespace() for _ in range(4)]
    manager.measured_candidate_indices = [4]
    manager.gp_model = object()
    manager.compute_acquisition_scores = lambda: SimpleNamespace(
        positions=np.array([[0.0, 1.0], [2.0, 2.0], [2.0, 1.0]]),
        acquisition=np.array([3.0, 2.0, 1.0]),
    )

    suggestions = manager.suggest(n_suggestions=2)

    assert {tuple(position) for position in suggestions} == {
        (0.0, 0.0),
        (2.0, 2.0),
    }


def test_exclusion_radius_filters_adaptive_suggestions():
    manager = configure_manager(
        make_manager(),
        exclusion_radius=1.01,
        w_peak=0.0,
        w_g=0.0,
    )
    manager.measurements = [SimpleNamespace()]
    manager.measured_candidate_indices = [4]

    import torch

    class GPModel:
        @staticmethod
        def posterior(candidate_x):
            count = candidate_x.shape[0]
            return SimpleNamespace(
                mean=torch.zeros(count, 1, dtype=torch.double),
                variance=torch.arange(
                    1,
                    count + 1,
                    dtype=torch.double,
                ).unsqueeze(-1),
            )

    manager.gp_model = GPModel()

    scores = manager.compute_acquisition_scores()
    suggestion = manager.suggest()

    expected_positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [2.0, 2.0],
        ]
    )
    np.testing.assert_array_equal(scores.positions, expected_positions)
    assert np.linalg.norm(suggestion[0] - np.array([1.0, 1.0])) >= 1.01


def test_suggestion_exclusion_radius_spaces_multi_candidate_suggestion():
    manager = configure_manager(
        make_manager(),
        suggestion_exclusion_radius=1.01,
    )
    manager.measurements = [SimpleNamespace()]
    manager.measured_candidate_indices = [0]
    manager.gp_model = object()
    manager.compute_acquisition_scores = lambda: SimpleNamespace(
        positions=np.array(
            [
                [2.0, 2.0],
                [2.0, 1.0],
                [0.0, 2.0],
                [1.0, 0.0],
            ]
        ),
        acquisition=np.array([4.0, 3.0, 2.0, 1.0]),
    )

    suggestions = manager.suggest(n_suggestions=2)
    pairwise_distance = np.linalg.norm(suggestions[0] - suggestions[1])

    assert {tuple(position) for position in suggestions} == {
        (2.0, 2.0),
        (0.0, 2.0),
    }
    assert pairwise_distance > manager.suggestion_exclusion_radius


def test_exclusion_radius_does_not_space_suggestions_from_each_other():
    manager = configure_manager(
        make_manager(),
        exclusion_radius=1.01,
    )
    manager.measurements = [SimpleNamespace()]
    manager.measured_candidate_indices = [0]
    manager.gp_model = object()
    manager.compute_acquisition_scores = lambda: SimpleNamespace(
        positions=np.array([[2.0, 2.0], [2.0, 1.0]]),
        acquisition=np.array([2.0, 1.0]),
    )

    suggestions = manager.suggest(n_suggestions=2)

    assert np.linalg.norm(suggestions[0] - suggestions[1]) == pytest.approx(1.0)


def test_multi_candidate_suggestions_are_ordered_for_short_travel():
    manager = configure_manager(make_manager())
    manager.measured_candidate_indices = [0]
    positions = np.array([[10.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    ordered = manager._optimize_suggestion_path(positions)

    np.testing.assert_array_equal(
        ordered,
        np.array([[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]]),
    )


def test_exclusion_radius_excludes_candidate_on_boundary():
    manager = configure_manager(
        make_manager(),
        exclusion_radius=1.0,
    )
    manager.measured_candidate_indices = [4]

    eligible = manager.get_eligible_candidate_indices()

    np.testing.assert_array_equal(
        eligible,
        np.array([0, 2, 6, 8]),
    )


def test_exclusion_radius_reports_when_no_candidate_is_eligible():
    manager = configure_manager(
        make_manager(),
        exclusion_radius=10.0,
    )
    manager.measured_candidate_indices = [4]

    with pytest.raises(ValueError, match="No eligible candidate"):
        manager.get_farthest_unsampled_position()


def test_normalize_tensor_clips_to_configured_percentiles():
    manager = configure_manager(
        make_manager(),
        normalization_lower_percentile=25.0,
        normalization_upper_percentile=75.0,
    )

    import torch

    normalized = manager.normalize_tensor(
        torch.tensor([0.0, 1.0, 2.0, 100.0], dtype=torch.double)
    )

    assert normalized[0] == 0
    assert normalized[-1] == pytest.approx(1.0)
    assert 0 < normalized[1] < normalized[2] < 1


def test_run_collects_unique_measurements():
    manager = make_manager()

    manager.run(
        **default_sampling_kwargs(
            num_q_points=64,
            max_measurements=4,
        )
    )

    assert len(manager.measurements) == 4
    assert len(set(manager.engine_tool.measured_candidate_indices)) == 4
    assert manager.latest_scores is not None


def test_run_acquires_multiple_candidates_before_refitting():
    manager = make_manager()
    suggestion_sizes = []
    refit_count = 0
    engine = manager.engine_tool

    def suggest(n_suggestions=1):
        suggestion_sizes.append(n_suggestions)
        indices = engine.get_unsampled_candidate_indices()[:n_suggestions]
        return engine.candidate_positions[indices]

    def update(positions, q_values, intensities):
        nonlocal refit_count
        refit_count += 1
        for position, q, intensity in zip(
            positions, q_values, intensities, strict=True
        ):
            candidate_index = engine.get_candidate_index(position)
            engine.measurements.append(
                SAXSMeasurement(
                    position=np.asarray(position),
                    q=np.asarray(q),
                    intensity=np.asarray(intensity),
                    spectrum=np.zeros(engine.num_q_points),
                    background=np.zeros(engine.num_q_points),
                )
            )
            engine.measured_candidate_indices.append(candidate_index)

    engine.suggest = suggest
    engine.update = update

    manager.run(
        **default_sampling_kwargs(
            num_q_points=64,
            max_measurements=7,
        ),
        num_candidates_per_suggestion=3,
    )

    assert suggestion_sizes == [3, 1]
    assert refit_count == 3
    assert len(manager.measurements) == 7


def test_refit_excludes_measurement_when_mll_prior_sampling_fails(monkeypatch):
    manager = configure_manager(make_manager())
    for candidate_index in range(3):
        record_candidate(manager, candidate_index)

    calls = []

    def fake_fit_gp(
        train_x,
        train_y,
        max_fit_gp_mll_iterations=None,
        fit_mll=True,
    ):
        calls.append((train_x.shape[0], fit_mll))
        if train_x.shape[0] == 4 and fit_mll and len(manager.measurements) == 4:
            raise RuntimeError(
                "Must provide inverse transform to be able to sample from prior."
            )
        return object()

    monkeypatch.setattr(manager, "fit_gp", fake_fit_gp)
    manager.refit_model()
    record_candidate(manager, 3)

    manager.refit_model()

    assert manager.excluded_measurement_indices == {3}
    assert calls[-2:] == [(4, True), (3, False)]

    record_candidate(manager, 4)
    manager.refit_model()

    assert calls[-1] == (4, True)
    assert manager.excluded_measurement_indices == {3}


def test_fit_gp_creates_independent_component_models_and_configures_iterations(
    monkeypatch,
):
    import torch

    calls = []
    monkeypatch.setattr(
        "eaa_aps12id.tools.spatial_saxs_sampling.fit_gpytorch_mll",
        lambda mll, **kwargs: calls.append(kwargs),
    )
    model = SpatialSAXSAdaptiveSamplingEngineTool.fit_gp(
        torch.rand(4, 2, dtype=torch.double),
        torch.rand(4, 2, dtype=torch.double),
        max_fit_gp_mll_iterations=7,
    )

    assert len(model.models) == 2
    assert model.models[0].covar_module is not model.models[1].covar_module
    assert model.models[0].covar_module.base_kernel.ard_num_dims == 2
    assert calls == [
        {"optimizer_kwargs": {"options": {"maxiter": 7}}},
        {"optimizer_kwargs": {"options": {"maxiter": 7}}},
    ]

    SpatialSAXSAdaptiveSamplingEngineTool.fit_gp(
        torch.rand(4, 2, dtype=torch.double),
        torch.rand(4, 2, dtype=torch.double),
        max_fit_gp_mll_iterations=None,
    )

    assert calls[-2:] == [{}, {}]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_fit_gp_mll_iterations": 0}, "`max_fit_gp_mll_iterations`"),
    ],
)
def test_configure_rejects_invalid_gp_fit_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        configure_manager(make_manager(), **kwargs)


def test_configure_rejects_invalid_peak_observable():
    with pytest.raises(ValueError, match="`peak_observable`"):
        configure_manager(make_manager(), peak_observable="prominence")


@pytest.mark.parametrize("peak_observale_map_blur", [-0.1, np.inf])
def test_configure_rejects_invalid_peak_observable_map_blur(
    peak_observale_map_blur,
):
    with pytest.raises(ValueError, match="`peak_observale_map_blur`"):
        configure_manager(
            make_manager(),
            peak_observale_map_blur=peak_observale_map_blur,
        )


def test_configure_accepts_irregular_candidate_positions():
    candidate_positions = np.array([[0.0, 0.0], [0.2, 1.0], [1.0, 0.3], [2.0, 3.0]])

    engine = configure_manager(
        make_manager(),
        candidate_positions=candidate_positions,
        peak_observale_map_blur=1.0,
        num_initial_samples=3,
    )

    np.testing.assert_array_equal(engine.candidate_positions, candidate_positions)


@pytest.mark.parametrize(
    ("candidate_positions", "message"),
    [
        ([], r"shape \(N, 2\)"),
        ([[0.0, 0.0, 1.0]], r"shape \(N, 2\)"),
        ([[0.0, np.nan], [1.0, 1.0]], "finite"),
        ([[0.0, 0.0], [0.0, 0.0]], "duplicates"),
    ],
)
def test_configure_rejects_invalid_candidate_positions(
    candidate_positions,
    message,
):
    with pytest.raises(ValueError, match=message):
        configure_manager(
            make_manager(),
            candidate_positions=candidate_positions,
        )


@pytest.mark.parametrize(
    ("known_peak_q_values", "message"),
    [
        ([], "at least one"),
        ([np.nan], "finite positive"),
        ([0.2, 0.2], "duplicates"),
        ([0.001], r"\[q_min, q_max\]"),
    ],
)
def test_configure_rejects_invalid_known_peak_q_values(
    known_peak_q_values,
    message,
):
    with pytest.raises(ValueError, match=message):
        configure_manager(
            make_manager(),
            known_peak_q_values=known_peak_q_values,
        )


def test_configure_rejects_negative_exclusion_radius():
    with pytest.raises(ValueError, match="`exclusion_radius`"):
        configure_manager(make_manager(), exclusion_radius=-0.1)


def test_configure_rejects_negative_suggestion_exclusion_radius():
    with pytest.raises(ValueError, match="`suggestion_exclusion_radius`"):
        configure_manager(make_manager(), suggestion_exclusion_radius=-0.1)


def test_run_rejects_invalid_num_candidates_per_suggestion():
    with pytest.raises(ValueError, match="`num_candidates_per_suggestion`"):
        make_manager().run(
            **default_sampling_kwargs(),
            num_candidates_per_suggestion=0,
        )


def test_configure_rejects_negative_new_peak_relative_area():
    with pytest.raises(ValueError, match="`new_peak_min_relative_area`"):
        configure_manager(
            make_manager(),
            new_peak_min_relative_area=-0.1,
        )


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_configure_rejects_invalid_peak_map_concentration(value):
    with pytest.raises(ValueError, match="`peak_map_min_concentration`"):
        configure_manager(
            make_manager(),
            peak_map_min_concentration=value,
        )


def test_zero_weights_reduce_acquisition_to_uncertainty_baseline():
    manager = configure_manager(
        make_manager(),
        w_peak=0.0,
        w_g=0.0,
    )

    import torch

    class Posterior:
        mean = torch.zeros(9, 2, dtype=torch.double)
        variance = torch.arange(1, 10, dtype=torch.double).unsqueeze(-1).repeat(1, 2)

    class GPModel:
        @staticmethod
        def posterior(candidate_x):
            return Posterior()

    manager.gp_model = GPModel()
    manager.compute_peak_gradient_magnitude = lambda candidate_x: pytest.fail(
        "gradient calculation should be skipped"
    )
    manager.compute_concentration_gated_peak_score = lambda mean: pytest.fail(
        "peak-area calculation should be skipped"
    )

    scores = manager.compute_acquisition_scores()

    assert manager.epsilon_acquisition == 1e-3
    np.testing.assert_array_equal(scores.gradient_tilde, np.zeros(9))
    np.testing.assert_array_equal(scores.peak_area_tilde, np.zeros(9))
    np.testing.assert_allclose(
        scores.acquisition,
        manager.epsilon_acquisition * scores.sigma_tilde,
    )


def test_run_forwards_non_position_acquisition_kwargs():
    manager = make_manager()

    manager.run(
        **default_sampling_kwargs(
            num_q_points=64,
            max_measurements=4,
        ),
        non_position_kwargs_for_acquisition_tool={
            "q_min": 0.01,
            "q_max": 1.0,
            "q_step": 0.001,
            "exposure": 0.5,
        }
    )

    assert manager.acquisition_tool.calls
    assert all(call["q_step"] == 0.001 for call in manager.acquisition_tool.calls)
    assert all(call["exposure"] == 0.5 for call in manager.acquisition_tool.calls)
    for call, measurement in zip(
        manager.acquisition_tool.calls,
        manager.measurements,
        strict=True,
    ):
        np.testing.assert_array_equal(
            measurement.position,
            [call["y"], call["x"]],
        )


def test_run_publishes_webui_progress_and_posterior_tile(tmp_path):
    manager = make_manager()
    manager.runtime_controller = WebUIRuntimeController(
        manager,
        upload_dir=str(tmp_path),
    )
    manager.runtime_controller.build()

    manager.run(
        **default_sampling_kwargs(
            num_q_points=64,
            max_measurements=4,
        )
    )

    snapshot = manager.runtime_controller.snapshot()
    messages = snapshot["messages"]
    assert any(
        message["content"].startswith("Starting adaptive SAXS sampling")
        for message in messages
    )
    assert any(
        message["content"] == "Gaussian process model update complete."
        for message in messages
    )
    primary = next(
        conversation
        for conversation in snapshot["conversations"]
        if conversation["id"] == "primary"
    )
    assert len(primary["visualization_tiles"]) == 2
    for tile in primary["visualization_tiles"]:
        assert tile["content"]["type"] == "image"
        assert (
            tile["content"].get("image_path", "").endswith(".png")
            or tile["content"].get("image_url", "").startswith("data:image/png")
        )
