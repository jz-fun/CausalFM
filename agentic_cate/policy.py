"""A categorical acquisition policy with a differentiable CausalFM encoder."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class AcquisitionPolicy(nn.Module):
    def __init__(self, backend, hidden=64, exploration=0.02):
        super().__init__()
        if not 0 <= exploration < 1:
            raise ValueError("exploration must be in [0, 1)")
        self.backend = backend
        self.exploration = exploration
        dim = 2 * backend.hidden_size + 1
        self.action_head = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.stop_head = nn.Sequential(
            nn.Linear(backend.hidden_size + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.to(backend.device)

    def distribution(self, obs):
        if obs.remaining_budget <= 0 or not obs.available.any():
            raise ValueError("cannot act in a terminal observation")
        targets = np.concatenate((obs.pool_x, obs.query_x), axis=0)
        out = self.backend.predict(obs, targets=targets, return_representations=True)
        reps = out["representations"]
        n_pool = len(obs.pool_x)
        pool_rep, query_rep = reps[:n_pool], reps[n_pool:]
        summary = query_rep.mean(0)
        budget = reps.new_tensor([obs.remaining_budget / max(obs.initial_budget, 1)])
        features = torch.cat((pool_rep, summary.expand(n_pool, -1), budget.expand(n_pool, 1)), -1)
        logits = torch.cat((self.action_head(features).reshape(-1),
                            self.stop_head(torch.cat((summary, budget))).reshape(1)))
        mask = torch.as_tensor(np.r_[np.repeat(obs.available, 2), True], device=logits.device)
        probs = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        uniform = mask.to(probs.dtype) / mask.sum()
        probs = (1 - self.exploration) * probs + self.exploration * uniform
        return Categorical(probs=probs)

    def head_parameters(self):
        yield from self.action_head.parameters()
        yield from self.stop_head.parameters()


def advantages(returns, group_size, estimator="reinforcepp_baseline"):
    """Task-local centering, with optional global trajectory-level whitening.

    All groups have equal K. Whitening is over trajectories, and the loss sums
    actions within a trajectory, preserving equal task/trajectory weighting.
    This short-trajectory adaptation is not OpenRLHF's token-weighted pipeline.
    """
    if group_size <= 1 or len(returns) % group_size:
        raise ValueError("need complete groups of at least two trajectories")
    values = torch.as_tensor(returns, dtype=torch.float32).reshape(-1, group_size)
    if estimator == "rloo":
        centered = values - (values.sum(-1, keepdim=True) - values) / (group_size - 1)
    elif estimator == "reinforcepp_baseline":
        centered = values - values.mean(-1, keepdim=True)
        centered = (centered - centered.mean()) / centered.std(correction=0).clamp_min(1e-6)
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    return centered.reshape(-1).detach()


def clipped_policy_loss(log_probs, old_log_probs, advantage, clip=0.2):
    ratio = (log_probs - old_log_probs.detach()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    return -surrogate.sum()


def gmm_nll(output, truth):
    """NLL on the original outcome-difference scale, restricted to query rows."""
    target = torch.as_tensor(truth, dtype=output["gmm_mu"].dtype,
                             device=output["gmm_mu"].device).reshape(-1, 1)
    log_component = torch.distributions.Normal(output["gmm_mu"], output["gmm_sigma"]).log_prob(target)
    return -torch.logsumexp(output["gmm_pi"].clamp_min(1e-12).log() + log_component, dim=-1).mean()
