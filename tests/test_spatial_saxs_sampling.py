import importlib.util

import numpy as np
import pytest

from eaa_aps12id.task_managers.spatial_saxs_sampling import (
    SpatialSAXSAdaptiveSamplingTaskManager,
)
from eaa_core.gui.runtime import WebUIRuntimeController
from eaa_core.tool.base import BaseTool


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
        "x_values": [0.0, 1.0, 2.0],
        "y_values": [0.0, 1.0, 2.0],
        "q_min": 0.01,
        "q_max": 1.0,
        "num_q_points": 16,
        "num_initial_samples": 3,
        "max_measurements": 5,
        "num_components": 2,
        "num_mc_samples": 4,
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
    manager._configure_sampling_run(**default_sampling_kwargs(**kwargs))
    return manager


def test_preprocess_spectrum_interpolates_and_log_transforms():
    manager = configure_manager(make_manager())
    q = np.geomspace(0.005, 1.5, 8)
    intensity = q * 2.0

    spectrum = manager.preprocess_spectrum(q[::-1], intensity[::-1])

    expected = np.log(
        np.interp(manager.q_grid, q, intensity) + manager.epsilon_intensity
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


@pytest.mark.parametrize(
    "dimension_reduction_method", ["pca", "sparse_pca", "nmf"]
)
def test_dimensionality_reduction_and_standardization_shapes(
    dimension_reduction_method,
):
    manager = configure_manager(make_manager())
    spectra = np.vstack(
        [
            np.linspace(0, 1, manager.num_q_points),
            np.linspace(1, 0, manager.num_q_points),
            np.sin(np.linspace(0, np.pi, manager.num_q_points)),
            np.cos(np.linspace(0, np.pi, manager.num_q_points)),
        ]
    )

    import torch

    latent_scores = manager.fit_dimensionality_reduction(
        torch.as_tensor(spectra, dtype=torch.double),
        manager.num_components,
        dimension_reduction_method,
        random_seed=manager.random_seed,
    )
    standardized_latent_scores = manager.standardize_latent_scores(
        latent_scores, manager.epsilon_z
    )

    assert latent_scores.shape == (4, manager.num_components)
    assert standardized_latent_scores.shape == (4, manager.num_components)
    np.testing.assert_allclose(
        standardized_latent_scores.mean(dim=0).detach().cpu().numpy(),
        np.zeros(manager.num_components),
        atol=1e-12,
    )


def test_configure_rejects_unknown_dimensionality_reduction_method():
    with pytest.raises(ValueError, match="`dimension_reduction_method` must be one of"):
        configure_manager(make_manager(), dimension_reduction_method="tsne")


def test_logdet_diversity_gain_matches_explicit_recompute():
    manager = configure_manager(make_manager())

    import torch

    current = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=torch.double
    )
    candidates = torch.tensor([[0.5, 0.5], [2.0, -1.0]], dtype=torch.double)

    gains = manager.logdet_diversity_gain(current, candidates)

    current_centered = current - current.mean(dim=0)
    base = (
        manager.lambda_logdet
        * torch.eye(manager.num_components, dtype=torch.double)
        + current_centered.T @ current_centered
    )
    expected = []
    for candidate in candidates:
        appended = torch.cat([current, candidate[None, :]], dim=0)
        appended_centered = appended - appended.mean(dim=0)
        new = (
            manager.lambda_logdet
            * torch.eye(manager.num_components, dtype=torch.double)
            + appended_centered.T @ appended_centered
        )
        expected.append(torch.logdet(new) - torch.logdet(base))
    expected = torch.stack(expected)

    torch.testing.assert_close(gains, expected)


def test_run_collects_unique_measurements():
    manager = make_manager()

    manager.run(
        **default_sampling_kwargs(
            num_q_points=8,
            num_mc_samples=2,
            max_measurements=4,
        )
    )

    assert len(manager.measurements) == 4
    assert len(set(manager.measured_candidate_indices)) == 4
    assert manager.latest_scores is not None


def test_refit_excludes_measurement_when_mll_prior_sampling_fails(monkeypatch):
    manager = configure_manager(make_manager())
    for candidate_index in range(3):
        manager.measure_candidate(candidate_index)

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
    manager.measure_candidate(3)

    manager.refit_model()

    assert manager.excluded_measurement_indices == {3}
    assert calls[-2:] == [(4, True), (3, False)]

    manager.measure_candidate(4)
    manager.refit_model()

    assert calls[-1] == (4, True)
    assert manager.excluded_measurement_indices == {3}


def test_fit_gp_creates_independent_component_models_and_configures_iterations(
    monkeypatch,
):
    import torch

    calls = []
    monkeypatch.setattr(
        "eaa_aps12id.task_managers.spatial_saxs_sampling.fit_gpytorch_mll",
        lambda mll, **kwargs: calls.append(kwargs),
    )
    model = SpatialSAXSAdaptiveSamplingTaskManager.fit_gp(
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

    SpatialSAXSAdaptiveSamplingTaskManager.fit_gp(
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


def test_zero_weights_reduce_acquisition_to_uncertainty_baseline():
    manager = configure_manager(
        make_manager(),
        w_d=0.0,
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
    manager.compute_latent_gradient_magnitude = lambda candidate_x: torch.ones(9)
    manager.compute_expected_logdet_diversity = lambda posterior: torch.ones(9)

    scores = manager.compute_acquisition_scores()

    assert manager.epsilon_acquisition == 1e-3
    np.testing.assert_allclose(
        scores.acquisition,
        manager.epsilon_acquisition * scores.sigma_tilde,
    )


def test_run_forwards_non_position_acquisition_kwargs():
    manager = make_manager()

    manager.run(
        **default_sampling_kwargs(
            num_q_points=8,
            num_mc_samples=2,
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


def test_run_publishes_webui_progress_and_posterior_tile(tmp_path):
    manager = make_manager()
    manager.runtime_controller = WebUIRuntimeController(
        manager,
        upload_dir=str(tmp_path),
    )
    manager.runtime_controller.build()

    manager.run(
        **default_sampling_kwargs(
            num_q_points=8,
            num_mc_samples=2,
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
        assert tile["content"]["image_path"].endswith(".png")
