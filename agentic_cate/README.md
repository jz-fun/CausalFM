# Active CATE acquisition: runnable agentic-RL MVP

This optional prototype post-trains the repository's **actual standard-CATE
CausalFM**, starting from a pinned pretrained checkpoint. A policy chooses which
candidate unit to acquire and which binary treatment to assign. The simulator
reveals only that unit's factual outcome. The policy can acquire another unit or
stop. The default budget is two acquisitions.

The implementation is a small CPU pilot and an implementation/gradient check.
It does not establish an accuracy improvement, new causal identification result,
or out-of-distribution generalization. It neither changes nor repairs the
original front-door generator or datasets.

## Install and test

Run all commands from the repository root. Use Python 3.10 or newer and a fresh
virtual environment. Install a suitable CPU or CUDA PyTorch wheel first if your
environment requires a particular wheel source; the optional requirements list
the minimum dependencies for this prototype.

```bash
python -m pip install -r requirements-agentic.txt
python -m pytest tests/test_agentic_environment.py tests/test_agentic_backend.py tests/test_agentic_policy.py tests/test_agentic_download_checkpoint.py -q
```

The architecture tests use small random-initialized models and do not require a
download. A random-initialized smoke test is not a pretrained CausalFM result.

## Download the pretrained standard-CATE checkpoint

```bash
python -m agentic_cate.download_checkpoint
```

This downloads `checkpoints_agentic/standard.pth` from
`yccm/CausalFM-toolkit`, pinned at commit
`bb74ef729c70d274fe2bb422c3b0edff4754fa37`. Its required SHA-256 is:

```text
9b75ebf02016a61b3a6e8fc075f52f9c56189ea5d7e7633137a5c9b70f25774e
```

An existing matching file is reused. A mismatching file is preserved unless
`--overwrite` is explicitly supplied. The downloader verifies bytes before
installing them atomically; the backend subsequently loads the tensor state with
`torch.load(..., weights_only=True)` and strict architecture matching.

## Quick pretrained smoke run

```bash
python -m agentic_cate.train \
  --checkpoint checkpoints_agentic/standard.pth \
  --device cpu --mode joint \
  --steps 1 --tasks-per-batch 2 --rollouts 4 --update-epochs 2 \
  --budget 2 --eval-tasks 2 \
  --output results/agentic_cate/pretrained_smoke.json
```

This writes a JSON report and a local `.pt` state file. The report includes
policy-to-encoder and policy-to-head gradient norms, changed parameter names,
forward-call counts, training trajectories and held-out errors. These checks
show whether the intended training path executes; a one-step run is not efficacy
evidence.

## Full predeclared pilot

```bash
python -m agentic_cate.study \
  --checkpoint checkpoints_agentic/standard.pth \
  --steps 16 --eval-tasks 64 \
  --seeds 2026 2027 2028 \
  --output-dir results/agentic_cate/pilot
```

The study runs `joint`, `head_only`, and matched-trajectory `sft` for each of the
three optimizer seeds. Its fixed training defaults are:

| Setting | Value |
|---|---:|
| Training iterations | 16 |
| Tasks per iteration | 2 |
| Rollouts per task, K | 4 |
| Optimizer epochs per rollout batch | 2 |
| Acquisition budget, B | 2 |
| Initial factual support | 16 units |
| Candidate acquisition pool | 8 units |
| Query set | 8 units |
| Covariates | 3 |
| Held-out tasks | 64 |
| Optimizer seeds | 2026, 2027, 2028 |

Training task seeds start at 10000; the study's held-out task seeds start at
200000. Retention tasks use a separate range starting at 500000. The same 64
held-out tasks are reused across optimizer seeds and methods. Therefore
optimizer-seed standard deviations do not represent independent task samples.
No held-out metric is used to choose a checkpoint: each run evaluates its final
iteration. The study CLI exposes steps, evaluation-task count and optimizer
seeds; other settings above come from `train.py` defaults.

The study writes `results/agentic_cate/pilot/summary.json` after all runs finish,
plus each run's JSON report. The accompanying experiment write-up is
`docs/agentic_cate_experiment.tex`. `.pt` trained states and `.pth` downloaded
weights are intentionally ignored by git; JSON reports are the reviewable
artifacts. Rerunning in an existing output directory replaces that run's reports
and states.

## Episode and data contract

`make_task(seed)` samples one nonlinear SCM and holds it fixed throughout the
episode. It creates initial factual support, a candidate covariate pool, and
query covariates with hidden analytic CATE labels. The potential outcomes share
one additive noise draw per unit, so their difference equals analytic CATE in
this deliberately simple simulator.

- `2 * i + a` assigns treatment `a in {0, 1}` to pool unit `i` and reveals
  `Y_i(a)`.
- `2 * n_pool` is STOP and consumes no acquisition budget.
- After either treatment is queried, both actions for that unit become illegal.
- Exhausting the budget or candidate pool terminates the episode automatically.
- The policy sees factual context `(X, A, Y)`, candidate/query covariates,
  availability, budget, and fixed normalization statistics. It never receives
  unobserved potential outcomes or query CATE labels.
- `terminal_truth()` is reserved for the training/evaluation controller and
  raises before episode termination.

Covariate and outcome normalization are fitted **only on the initial factual
support** and held fixed for all trajectories of that task. Neither query truth
nor unobserved potential outcomes are used to fit a scaler. Effect predictions
are rescaled by the outcome scale without adding the outcome mean.

The initial treatment allocation is randomized and balanced. All tasks use the
same mechanism family with new parameters and seeds. Thus this pilot does not
test observational confounding, latent unit-specific effects beyond analytic
CATE, a new SCM family, instrumental variables or front-door identification.

## Model and training objective

`backend.py` uses the differentiable CausalFM forward path, bypassing the
original estimator's no-gradient convenience method. It exposes query
representations to the acquisition/STOP heads. The default joint run unfreezes
the **last Transformer block and GMM head**. This is partial fine-tuning, not
LoRA or an adapter implementation. The discrete-action policy gradient reaches
the last block through its representations; the GMM head also receives the
predictive loss.

The terminal policy reward is

```text
frozen_initial_CATE_MSE - current_terminal_CATE_MSE
    - 0.005 * number_of_acquired_units
```

The initial reference is a frozen copy of the pretrained model and uses the
initial context, independently of the sampled actions. The default estimator
centers rewards within each task's K trajectories, then whitens over all
trajectories in that batch. A PPO-clipped action-probability surrogate is summed
over actions and averaged over trajectories. This is a **REINFORCE++-style
short-trajectory adaptation**, not an exact reproduction of OpenRLHF's
token-weighted REINFORCE++ pipeline. `--estimator rloo` provides an alternative
leave-one-out baseline.

Joint training combines the clipped policy objective, entropy bonus, GMM query
negative log-likelihood, and reference retention:

```text
policy_loss - 0.001 * entropy
    + 0.1 * terminal_GMM_NLL
    + 0.1 * ordinary_task_reference_mean_MSE
```

The hidden query labels supervise the terminal reward and GMM NLL only after
the episode finishes. Retention uses reference predictions on the initial
context of an independent ordinary task; it does not open that task's oracle.
The setup is a hybrid RL/supervised post-training objective, so a joint gain
cannot automatically be attributed to policy gradients alone.

## Comparisons and interpretation

- **Frozen / initial:** pretrained predictor with no new observations.
- **Frozen / random budget-2:** pretrained predictor after random acquisition
  that spends the full budget.
- **Head-only RL:** freeze the entire predictor and train only acquisition and
  STOP heads.
- **Joint RL:** train the policy heads, last Transformer block and GMM head.
- **Matched SFT:** replay exactly the joint run's training actions and final
  contexts; update the same predictor parameters using GMM NLL and retention,
  with no policy loss. Its untrained action head is not the comparison target.
- **Fixed-context comparison:** evaluate joint and SFT backbones on exactly the
  contexts selected by joint RL on the held-out tasks. Also compare their
  initial and shared random contexts.

Equal iterations and replayed trajectories do not imply equal computation.
Reports record model forward calls; these counts do not include backward FLOPs.
Report prediction quality, acquisition counts, gradient checks and cost
together. Learned policies are sampled at evaluation, with reproducible task
seeds; the random baseline is required to spend its full budget.

The present deliverable is a runnable route for testing whether acquisition
feedback can help CausalFM and whether updating its backbone adds value beyond
a controller or matched supervised adaptation. A larger, independently
validated experiment is needed before making an accuracy or novelty claim.
