"""Differentiable standard-CATE CausalFM backend.

The original estimator helper has ``@torch.no_grad``; this module intentionally
uses ``_pack_eval_io`` and ``forward`` instead.  The default adaptation updates
the last encoder block and GMM head.  It is partial fine-tuning, not an adapter
or LoRA implementation.  ``checkpoint=None`` creates a random-init architecture
for smoke tests, and must not be reported as a pretrained CausalFM result.

Only factual, visible support observations are used to fit normalization.  An
episode can supply fixed ``x_mean/x_scale/y_mean/y_scale`` fields (or the same
fields under ``normalization``), fitted on its initial support.  Without these
fields the current support is fitted on every call; that legal fallback changes
the normalization as new observations arrive.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


# This repository originally has no installable package metadata.  Resolve its
# vendored model explicitly instead of depending on an unrelated pip tabpfn.
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tabpfn.model.causalFM import PerFeatureTransformerCATE  # noqa: E402


def _field(observation: Any, name: str, default: Any = None) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    return getattr(observation, name, default)


class CausalFMBackend(nn.Module):
    """One-task-at-a-time differentiable interface to the actual CausalFM.

    ``targets`` denotes target covariates, not outcome or treatment-effect labels.
    Returned effect means, mixture locations and standard deviations use the
    original outcome units.  Centering an outcome cancels in a treatment
    difference, so its mean is never added to CATE predictions.
    """

    hidden_size = 128

    def __init__(
        self,
        checkpoint: str | None,
        device: str = "cpu",
        trainable_layers: int = 1,
        seed: int = 0,
        nlayers: int = 10,
    ) -> None:
        super().__init__()
        if nlayers < 1 or not 0 <= trainable_layers <= nlayers:
            raise ValueError("Require nlayers >= 1 and 0 <= trainable_layers <= nlayers")
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.checkpoint_loaded = False
        self.forward_calls = 0
        # Initialization is reproducible without perturbing the caller's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = PerFeatureTransformerCATE(nlayers=nlayers, seed=seed)

        if checkpoint is not None:
            # Safe loader only.  A legacy pickled-object checkpoint is not
            # silently retried with unrestricted object deserialization.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError("Checkpoint must be a tensor state_dict or contain model_state_dict/state_dict")
            state = payload.get("model_state_dict", payload.get("state_dict", payload))
            if not isinstance(state, Mapping) or not state:
                raise ValueError("Checkpoint model state is empty or not a mapping")
            if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
                raise ValueError("Checkpoint model state must map string keys to tensors")
            self.model.load_state_dict(state, strict=True)
            self.checkpoint_loaded = True

        self.model.to(self.device)
        self.model.eval()  # eval does not disable the policy/prediction gradient.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if trainable_layers:
            for layer in self.model.transformer_encoder.layers[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        for parameter in self.model.gmm_head.parameters():
            parameter.requires_grad_(True)

    @staticmethod
    def normalization_from_support(x_context: Any, y_context: Any) -> dict[str, torch.Tensor]:
        """Fit population-standardization statistics using visible factual data."""
        x = torch.as_tensor(x_context, dtype=torch.float32).detach()
        y = torch.as_tensor(y_context, dtype=torch.float32).detach().reshape(-1)
        if x.ndim != 2 or len(x) == 0 or len(y) != len(x):
            raise ValueError("Support must contain a nonempty 2-D X and matching factual Y")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Support contains nonfinite covariates or factual outcomes")
        x_scale = x.std(dim=0, correction=0)
        y_scale = y.std(correction=0)
        # Constant columns are centered but not divided by a tiny value.
        x_scale = torch.where(x_scale > 1e-6, x_scale, torch.ones_like(x_scale))
        y_scale = torch.where(y_scale > 1e-6, y_scale, torch.ones_like(y_scale))
        return {
            "x_mean": x.mean(dim=0),
            "x_scale": x_scale,
            "y_mean": y.mean(),
            "y_scale": y_scale,
        }

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def predict(
        self,
        observation: Any,
        targets: Any = None,
        return_representations: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict on visible query covariates with an intact autograd graph."""
        x = self._tensor(_field(observation, "x_context"))
        a = self._tensor(_field(observation, "a_context")).reshape(-1, 1)
        y = self._tensor(_field(observation, "y_context")).reshape(-1, 1)
        target_x = self._tensor(_field(observation, "query_x") if targets is None else targets)
        if x.ndim != 2 or target_x.ndim != 2 or len(x) == 0 or len(target_x) == 0:
            raise ValueError("Context and targets must be nonempty 2-D covariate matrices")
        if len(x) != len(a) or len(x) != len(y) or x.shape[1] != target_x.shape[1]:
            raise ValueError("Support lengths or support/target feature dimensions do not match")
        if not all(torch.isfinite(t).all() for t in (x, a, y, target_x)):
            raise ValueError("Backend expects finite factual observations and covariates")
        if not torch.all((a == 0) | (a == 1)):
            raise ValueError("The standard-CATE backend expects binary treatments")

        normalization = _field(observation, "normalization", observation)
        names = ("x_mean", "x_scale", "y_mean", "y_scale")
        values = [_field(normalization, name) for name in names]
        if all(value is None for value in values):
            stats = self.normalization_from_support(x, y)
        elif any(value is None for value in values):
            raise ValueError("Provide all four fixed normalization statistics, or none")
        else:
            stats = dict(zip(names, values))
        stats = {name: self._tensor(value).detach() for name, value in stats.items()}
        if stats["x_mean"].numel() != x.shape[1] or stats["x_scale"].numel() != x.shape[1]:
            raise ValueError("Fixed covariate normalization has the wrong feature count")
        if stats["y_mean"].numel() != 1 or stats["y_scale"].numel() != 1:
            raise ValueError("Outcome normalization statistics must be scalars")
        if not all(torch.isfinite(t).all() for t in stats.values()):
            raise ValueError("Normalization statistics must be finite")
        if not torch.all(stats["x_scale"] > 0) or not torch.all(stats["y_scale"] > 0):
            raise ValueError("Normalization scales must be strictly positive")

        x_norm = (x - stats["x_mean"].reshape(1, -1)) / stats["x_scale"].reshape(1, -1)
        targets_norm = (target_x - stats["x_mean"].reshape(1, -1)) / stats["x_scale"].reshape(1, -1)
        y_norm = (y - stats["y_mean"]) / stats["y_scale"]
        packed_x, packed_a, packed_y, split = self.model._pack_eval_io(x_norm, a, y_norm, targets_norm)
        self.forward_calls += 1
        raw = self.model(
            packed_x,
            packed_a,
            packed_y,
            single_eval_pos=split,
            return_representations=return_representations,
        )
        result = {
            "cate_mean": raw["cate_mean"][0, split:, 0] * stats["y_scale"],
            "gmm_pi": raw["gmm_pi"][0, split:, :],
            "gmm_mu": raw["gmm_mu"][0, split:, :] * stats["y_scale"],
            "gmm_sigma": raw["gmm_sigma"][0, split:, :] * stats["y_scale"],
        }
        if return_representations:
            result["representations"] = raw["representations"][0, split:, :]
        return result
