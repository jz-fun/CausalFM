import numpy as np
import pytest
import torch

from agentic_cate.backend import CausalFMBackend
from agentic_cate.data import make_task
from agentic_cate.environment import CausalAcquisitionEnv
from agentic_cate.policy import AcquisitionPolicy, advantages, clipped_policy_loss
from agentic_cate.train import rollout


@pytest.fixture
def small_policy():
    """Exercise the real model while keeping CPU regression checks inexpensive."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            backend = CausalFMBackend(None, nlayers=2, trainable_layers=1)
            yield AcquisitionPolicy(backend)
    finally:
        torch.set_num_threads(old_threads)


def test_rloo_uses_other_samples():
    got = advantages([1., 2., 6., 4., 4., 4.], 3, "rloo")
    torch.testing.assert_close(got, torch.tensor([-3., -1.5, 4.5, 0., 0., 0.]))


def test_global_not_group_standard_deviation():
    got = advantages([0., .01, 0., 1.], 2)
    assert got[3] / got[1] == pytest.approx(100, rel=1e-5)
    assert got.mean().abs() < 1e-6
    torch.testing.assert_close(got.std(correction=0), torch.tensor(1.))


def test_constant_group_advantages_are_zero():
    assert not advantages([1., 1., 2., 2.], 2).any()


def test_clipping_removes_extra_positive_gain():
    current = torch.tensor([np.log(1.5)], requires_grad=True)
    loss = clipped_policy_loss(current, torch.zeros(1), torch.tensor(1.), .2)
    loss.backward()
    assert loss.item() == pytest.approx(-1.2)
    assert current.grad.item() == 0


def test_old_log_probs_and_advantages_are_frozen():
    current = torch.zeros(2, requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    adv = advantages([0., 1.], 2)
    clipped_policy_loss(current, old, adv[1]).backward()
    assert old.grad is None
    assert not adv.requires_grad


def test_invalid_groups_fail():
    with pytest.raises(ValueError):
        advantages([1., 2., 3.], 2)


def test_policy_only_optimizer_step_updates_registered_encoder(small_policy):
    """Catch an unregistered backend even when a direct backend gradient works."""
    policy = small_policy
    trainable_encoder = {
        name: parameter
        for name, parameter in policy.backend.model.named_parameters()
        if parameter.requires_grad and "transformer_encoder" in name
    }
    assert trainable_encoder
    before = {name: parameter.detach().clone()
              for name, parameter in trainable_encoder.items()}
    optimizer = torch.optim.SGD(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups
                     for parameter in group["params"]}
    assert all(id(parameter) in optimizer_ids
               for parameter in trainable_encoder.values())

    obs = CausalAcquisitionEnv(
        make_task(1, n_initial=4, n_pool=3, n_query=2, n_features=2),
        budget=2,
    ).observation()
    distribution = policy.distribution(obs)
    current = distribution.log_prob(torch.tensor(0)).reshape(1)
    # Deliberately exclude prediction, retention and entropy losses: this update
    # must be attributable to the discrete policy objective alone.
    loss = clipped_policy_loss(current, current.detach(), torch.tensor(1.0))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in trainable_encoder.values())
    optimizer.step()
    assert any(not torch.equal(parameter.detach(), before[name])
               for name, parameter in trainable_encoder.items())
    assert all(parameter.grad is None
               for parameter in policy.backend.model.transformer_encoder.layers[0].parameters())


def test_rollout_log_probabilities_recompute_and_stay_frozen(small_policy):
    policy = small_policy
    task = make_task(1, n_initial=4, n_pool=3, n_query=2, n_features=2)
    torch.manual_seed(0)
    trace = rollout(policy, task, budget=2)
    assert len(trace["actions"]) == 2  # Include a changed context in replay.
    assert len(trace["old_log_probs"]) == len(trace["actions"])
    assert all(not old.requires_grad and old.grad_fn is None
               for old in trace["old_log_probs"])
    saved = torch.stack(trace["old_log_probs"]).clone()

    def replay_log_probabilities():
        return torch.stack([
            policy.distribution(obs).log_prob(torch.tensor(action))
            for obs, action in zip(trace["observations"], trace["actions"])
        ])

    current = replay_log_probabilities()
    torch.testing.assert_close(current.detach(), saved, rtol=0, atol=1e-7)
    optimizer = torch.optim.SGD(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    clipped_policy_loss(current, torch.stack(trace["old_log_probs"]),
                        torch.tensor(1.0)).backward()
    optimizer.step()
    with torch.no_grad():
        updated = replay_log_probabilities()
    assert not torch.allclose(updated, saved, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.stack(trace["old_log_probs"]), saved,
                               rtol=0, atol=0)
