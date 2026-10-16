"""Architecture, gradient and leakage checks; random-init tests are not benchmarks."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from agentic_cate.backend import CausalFMBackend


@pytest.fixture(autouse=True)
def small_cpu_workload():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def observation():
    generator = torch.Generator().manual_seed(13)
    return {
        "x_context": torch.randn(8, 3, generator=generator),
        "a_context": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        "y_context": torch.tensor([5., 7., 6., 8., 3., 7., 6., 10.]),
        "query_x": torch.randn(4, 3, generator=generator),
    }


def test_random_init_real_architecture_finite_shapes():
    backend = CausalFMBackend(None, nlayers=1)
    assert not backend.checkpoint_loaded
    out = backend.predict(observation(), return_representations=True)
    assert out["cate_mean"].shape == (4,)
    assert out["representations"].shape == (4, 128)
    for key in ("gmm_pi", "gmm_mu", "gmm_sigma"):
        assert out[key].shape == (4, 5)
    assert all(torch.isfinite(value).all() for value in out.values())
    torch.testing.assert_close(out["gmm_pi"].sum(-1), torch.ones(4))
    torch.testing.assert_close(out["cate_mean"], (out["gmm_pi"] * out["gmm_mu"]).sum(-1))
    assert (out["gmm_sigma"] > 0).all()


def test_policy_representation_gradient_reaches_last_encoder():
    backend = CausalFMBackend(None, nlayers=2, trainable_layers=1)
    out = backend.predict(observation(), return_representations=True)
    policy = torch.nn.Linear(128, 3)
    # A discrete-action log probability provides a genuine policy gradient path.
    logits = policy(out["representations"].mean(0))
    loss = -logits.log_softmax(-1)[1]
    loss.backward()
    last = list(backend.model.transformer_encoder.layers[-1].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in last)
    assert all(not p.requires_grad and p.grad is None for p in backend.model.transformer_encoder.layers[0].parameters())
    assert all(not p.requires_grad for p in backend.model.x_encoder.parameters())


def test_scaler_ignores_oracle_fields_and_effect_has_no_location_shift():
    backend = CausalFMBackend(None, nlayers=1)
    obs = observation()
    original = backend.predict(obs)
    poisoned = dict(obs, y_query=torch.full((4,), 1e20), y0=torch.full((8,), -1e20), ite=torch.full((8,), 1e20))
    for key, value in backend.predict(poisoned).items():
        torch.testing.assert_close(value, original[key])
    transformed = dict(obs, y_context=obs["y_context"] * 3 + 50)
    scaled = backend.predict(transformed)
    torch.testing.assert_close(scaled["cate_mean"], original["cate_mean"] * 3)
    torch.testing.assert_close(scaled["gmm_mu"], original["gmm_mu"] * 3)
    torch.testing.assert_close(scaled["gmm_sigma"], original["gmm_sigma"] * 3)


def test_fixed_initial_support_scaler_is_accepted():
    backend = CausalFMBackend(None, nlayers=1)
    obs = observation()
    stats = backend.normalization_from_support(obs["x_context"], obs["y_context"])
    direct = backend.predict(obs)
    fixed = backend.predict(dict(obs, normalization=stats))
    torch.testing.assert_close(direct["cate_mean"], fixed["cate_mean"])
    torch.testing.assert_close(stats["y_mean"], obs["y_context"].mean())
    bad = dict(obs, y_scale=1.)
    with pytest.raises(ValueError, match="all four"):
        backend.predict(bad)


def test_safe_checkpoint_round_trip(tmp_path):
    backend = CausalFMBackend(None, nlayers=1, seed=9)
    path = tmp_path / "state.pt"
    torch.save({"model_state_dict": backend.model.state_dict()}, path)
    loaded = CausalFMBackend(str(path), nlayers=1, seed=9)
    assert loaded.checkpoint_loaded
    torch.testing.assert_close(loaded.predict(observation())["cate_mean"], backend.predict(observation())["cate_mean"])


def test_core_import_does_not_eagerly_import_sklearn_estimators():
    root = Path(__file__).resolve().parents[1]
    code = "from tabpfn.model.causalFM import PerFeatureTransformerCATE; import sys; assert 'tabpfn.classifier' not in sys.modules; assert 'tabpfn.regressor' not in sys.modules"
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)
