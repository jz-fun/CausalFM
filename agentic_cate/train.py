"""Small CPU/GPU pilot. Run ``python -m agentic_cate.train --help``.

No simulator labels are supplied to action selection. The training evaluator
opens hidden query CATE only after each episode ends. A frozen initial CausalFM
provides an action-independent reference score and ordinary-task retention.
"""

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from .backend import CausalFMBackend
from .data import make_task
from .environment import CausalAcquisitionEnv
from .policy import AcquisitionPolicy, advantages, clipped_policy_loss, gmm_nll


def norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def mse(output, truth):
    target = torch.as_tensor(truth, device=output["cate_mean"].device)
    return float((output["cate_mean"].detach() - target).square().mean())


def rollout(policy, task, budget, *, random_policy=False, force_stop=False):
    env = CausalAcquisitionEnv(task, budget)
    observations, actions, log_probs = [], [], []
    while not env.done:
        obs = env.observation()
        if force_stop:
            action = env.stop_action
        elif random_policy:
            # Strong fixed-budget random baseline: spend the full acquisition budget.
            available = np.flatnonzero(env.legal_mask()[:-1])
            action = int(available[torch.randint(len(available), ()).item()])
        else:
            with torch.no_grad():
                dist = policy.distribution(obs)
                choice = dist.sample()
                action = int(choice)
                log_probs.append(dist.log_prob(choice).detach())
        observations.append(obs)
        actions.append(action)
        env.step(action)
    return {"observations": observations, "actions": actions, "old_log_probs": log_probs,
            "final": env.observation(), "initial": env.initial_observation(),
            "truth": env.terminal_truth(), "acquired": env.acquired_count}


def replay(task, budget, actions):
    """Replay actions from a training task; no oracle is read until termination."""
    env = CausalAcquisitionEnv(task, budget)
    for action in actions:
        env.step(action)
    if not env.done:
        raise ValueError("saved training trajectory did not terminate")
    return {"observations": [], "actions": list(actions), "old_log_probs": [],
            "final": env.observation(), "initial": env.initial_observation(),
            "truth": env.terminal_truth(), "acquired": env.acquired_count}


@torch.no_grad()
def evaluate(policy, reference, seeds, args):
    rows = []
    for seed in seeds:
        task = make_task(seed, n_initial=args.n_initial, n_pool=args.n_pool,
                         n_query=args.n_query, n_features=args.n_features)
        # Reuse exactly the same random choices across models/modes for evaluation.
        torch.manual_seed(seed)
        learned = rollout(policy, task, args.budget)
        torch.manual_seed(seed + 12345)
        random_trace = rollout(policy, task, args.budget, random_policy=True)
        initial = learned["initial"]
        y = learned["truth"]
        rows.append({"seed": seed,
                     "initial_reference_mse": mse(reference.predict(initial), y),
                     "initial_posttrained_mse": mse(policy.backend.predict(initial), y),
                     "random_reference_mse": mse(reference.predict(random_trace["final"]), y),
                     "random_posttrained_mse": mse(policy.backend.predict(random_trace["final"]), y),
                     "learned_reference_mse": mse(reference.predict(learned["final"]), y),
                     "learned_posttrained_mse": mse(policy.backend.predict(learned["final"]), y),
                     "learned_acquisitions": learned["acquired"],
                     "random_acquisitions": random_trace["acquired"],
                     "learned_actions": learned["actions"]})
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "seed"]
    summary = {key: {"mean": float(np.mean([r[key] for r in rows])),
                     "std": float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
               for key in keys}
    return {"summary": summary, "tasks": rows}


def run(args):
    if args.rollouts < 2 or args.steps < 1 or args.tasks_per_batch < 1:
        raise ValueError("steps/tasks_per_batch must be positive; rollouts must be >=2")
    if args.budget <= 0 or args.update_epochs < 1 or args.eval_tasks < 1:
        raise ValueError("training budget, update_epochs and eval_tasks must be positive")
    training_seeds = set(range(args.train_seed_start, args.train_seed_start + args.steps * args.tasks_per_batch))
    evaluation_seeds = set(range(args.eval_seed_start, args.eval_seed_start + args.eval_tasks))
    retention_seeds = set(range(500000, 500000 + args.steps))
    if training_seeds & evaluation_seeds or retention_seeds & (training_seeds | evaluation_seeds):
        raise ValueError("training, retention and evaluation task seeds must be disjoint")
    saved_traces = None
    if args.replay_trajectories:
        if args.mode != "sft":
            raise ValueError("trajectory replay is a supervised-only comparator")
        saved = json.loads(Path(args.replay_trajectories).read_text())
        for key in ("n_initial", "n_pool", "n_query", "n_features", "budget", "steps", "tasks_per_batch", "rollouts", "train_seed_start"):
            if saved["config"][key] != getattr(args, key):
                raise ValueError(f"replay configuration differs in {key}")
        saved_traces = saved["training_trajectories"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    started = time.perf_counter()
    checkpoint = str(Path(args.checkpoint).resolve()) if args.checkpoint else None
    backend = CausalFMBackend(checkpoint=checkpoint, device=args.device,
                              trainable_layers=args.trainable_layers, seed=args.seed,
                              nlayers=args.layers)
    reference = copy.deepcopy(backend)
    reference.eval()
    reference.requires_grad_(False)
    policy = AcquisitionPolicy(backend, exploration=args.exploration)
    if args.mode == "head_only":
        backend.requires_grad_(False)
    elif args.mode == "sft":
        for p in policy.head_parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    backend_parameters = [p for p in backend.parameters() if p.requires_grad]
    head_parameters = [p for p in policy.head_parameters() if p.requires_grad]
    encoder_parameters = [p for name, p in backend.model.named_parameters()
                          if p.requires_grad and "transformer_encoder" in name]
    initial_weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    config = vars(args).copy()
    config["checkpoint"] = checkpoint
    config["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None
    config["initialization"] = "pretrained_standard_causalfm" if checkpoint else "random_architecture_test"
    config["trainable_parameters"] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    config["trainable_encoder_parameters"] = sum(p.numel() for p in encoder_parameters)
    history, gradient_check, training_trajectories = [], None, []
    for iteration in range(args.steps):
        calls_before = backend.forward_calls + reference.forward_calls
        batch = []
        for j in range(args.tasks_per_batch):
            task_seed = args.train_seed_start + iteration * args.tasks_per_batch + j
            task = make_task(task_seed, n_initial=args.n_initial, n_pool=args.n_pool,
                             n_query=args.n_query, n_features=args.n_features)
            for k in range(args.rollouts):
                if saved_traces is None:
                    trace = rollout(policy, task, args.budget, random_policy=args.mode == "sft")
                else:
                    record = saved_traces[iteration][j * args.rollouts + k]
                    if record["task_seed"] != task_seed:
                        raise ValueError("replay task seed differs")
                    trace = replay(task, args.budget, record["actions"])
                trace["task_seed"] = task_seed
                with torch.no_grad():
                    initial_error = mse(reference.predict(trace["initial"]), trace["truth"])
                    final_error = mse(backend.predict(trace["final"]), trace["truth"])
                trace["reward"] = initial_error - final_error - args.cost * trace["acquired"]
                batch.append(trace)
        training_trajectories.append([{k: trace[k] for k in ("task_seed", "actions", "acquired", "reward")} for trace in batch])
        adv = advantages([r["reward"] for r in batch], args.rollouts, args.estimator).to(args.device)
        # Retention is evaluated on an independent, ordinary task at its initial context.
        retention_env = CausalAcquisitionEnv(make_task(500000 + iteration, n_initial=args.n_initial,
            n_pool=args.n_pool, n_query=args.n_query, n_features=args.n_features), args.budget)
        retain_obs = retention_env.observation()
        with torch.no_grad():
            retain_target = reference.predict(retain_obs)["cate_mean"]
        for epoch in range(args.update_epochs):
            optimizer.zero_grad(set_to_none=True)
            policy_terms, predictive_terms, entropy_terms = [], [], []
            for i, trace in enumerate(batch):
                if args.mode != "sft":
                    current, entropies = [], []
                    for obs, action in zip(trace["observations"], trace["actions"]):
                        dist = policy.distribution(obs)
                        current.append(dist.log_prob(torch.tensor(action, device=args.device)))
                        entropies.append(dist.entropy())
                    policy_terms.append(clipped_policy_loss(torch.stack(current),
                        torch.stack(trace["old_log_probs"]), adv[i], args.clip))
                    entropy_terms.append(torch.stack(entropies).mean())
                predictive_terms.append(gmm_nll(backend.predict(trace["final"]), trace["truth"]))
            predictive_loss = torch.stack(predictive_terms).mean()
            policy_loss = torch.stack(policy_terms).mean() if policy_terms else predictive_loss.new_zeros(())
            entropy = torch.stack(entropy_terms).mean() if entropy_terms else predictive_loss.new_zeros(())
            retention_loss = (backend.predict(retain_obs)["cate_mean"] - retain_target).square().mean()
            if args.mode == "joint" and gradient_check is None:
                gradients = torch.autograd.grad(policy_loss, encoder_parameters + head_parameters,
                                                retain_graph=True, allow_unused=True)
                n_encoder = len(encoder_parameters)
                grad_norm = lambda gs: float(torch.stack([g.detach().square().sum() for g in gs
                                               if g is not None]).sum().sqrt()) if any(g is not None for g in gs) else 0.0
                gradient_check = {"policy_to_encoder_norm": grad_norm(gradients[:n_encoder]),
                                  "policy_to_head_norm": grad_norm(gradients[n_encoder:])}
            loss = policy_loss - args.entropy * entropy
            if args.mode != "head_only":
                loss = loss + args.prediction_coef * predictive_loss + args.retention_coef * retention_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0,
                                          error_if_nonfinite=True)
            gradients_row = {"encoder_grad_norm": norm(encoder_parameters), "head_grad_norm": norm(head_parameters)}
            optimizer.step()
        row = {"step": iteration, "reward_mean": float(np.mean([r["reward"] for r in batch])),
               "reward_std": float(np.std([r["reward"] for r in batch])),
               "policy_loss": float(policy_loss.detach()), "prediction_loss": float(predictive_loss.detach()),
               "retention_loss": float(retention_loss.detach()), "entropy": float(entropy.detach()),
               "acquisitions_mean": float(np.mean([r["acquired"] for r in batch])),
               "model_forward_calls": backend.forward_calls + reference.forward_calls - calls_before,
               **gradients_row}
        history.append(row)
        print(json.dumps(row), flush=True)
    changed = [key for key, value in policy.state_dict().items()
               if not torch.equal(value, initial_weights[key])]
    # Evaluation seeds are never used for training or checkpoint selection.
    calls_before_evaluation = backend.forward_calls + reference.forward_calls
    evaluation = evaluate(policy, reference, range(args.eval_seed_start, args.eval_seed_start + args.eval_tasks), args)
    evaluation["model_forward_calls"] = backend.forward_calls + reference.forward_calls - calls_before_evaluation
    report = {"status": "small_preliminary_cpu_pilot_not_efficacy_evidence", "config": config,
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                          "device": args.device, "wall_seconds": time.perf_counter() - started},
              "gradient_check": gradient_check, "changed_parameter_names": changed,
              "history": history, "training_trajectories": training_trajectories, "evaluation": evaluation,
              "model_forward_calls_total": backend.forward_calls + reference.forward_calls}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"policy_state_dict": policy.state_dict(), "config": config}, output.with_suffix(".pt"))
    print(json.dumps({"output": str(output), "wall_seconds": report["runtime"]["wall_seconds"],
                      "gradient_check": gradient_check, "evaluation": evaluation["summary"]}), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mode", choices=["joint", "head_only", "sft"], default="joint")
    p.add_argument("--estimator", choices=["rloo", "reinforcepp_baseline"], default="reinforcepp_baseline")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--tasks-per-batch", type=int, default=2)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--trainable-layers", type=int, default=1)
    p.add_argument("--layers", type=int, default=10, help="architecture depth; must match a loaded checkpoint")
    p.add_argument("--n-initial", type=int, default=16)
    p.add_argument("--n-pool", type=int, default=8)
    p.add_argument("--n-query", type=int, default=8)
    p.add_argument("--n-features", type=int, default=3)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--cost", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--prediction-coef", type=float, default=0.1)
    p.add_argument("--retention-coef", type=float, default=0.1)
    p.add_argument("--entropy", type=float, default=0.001)
    p.add_argument("--exploration", type=float, default=0.02)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-seed-start", type=int, default=10000)
    p.add_argument("--eval-seed-start", type=int, default=90000)
    p.add_argument("--eval-tasks", type=int, default=16)
    p.add_argument("--output", default="results/agentic_cate/joint.json")
    p.add_argument("--replay-trajectories", default=None, help="joint training JSON whose contexts SFT replays exactly")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
