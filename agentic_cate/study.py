"""Predeclared small pilot with matched-trajectory supervised comparisons.

Run from the repository root. No metric is used for checkpoint selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .backend import CausalFMBackend
from .data import make_task
from .policy import AcquisitionPolicy
from .train import mse, parser, replay, run


def fixed_context_evaluation(joint, sft, checkpoint, joint_weights, sft_weights):
    """Evaluate both backbones on exactly the held-out contexts chosen by joint RL."""
    models = []
    for weights in (joint_weights, sft_weights):
        backend = CausalFMBackend(checkpoint=checkpoint, device="cpu", trainable_layers=1)
        policy = AcquisitionPolicy(backend)
        policy.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["policy_state_dict"])
        models.append(policy.backend.eval())
    c = joint["config"]
    rows = []
    with torch.no_grad():
        for row in joint["evaluation"]["tasks"]:
            task = make_task(row["seed"], n_initial=c["n_initial"], n_pool=c["n_pool"],
                             n_query=c["n_query"], n_features=c["n_features"])
            trace = replay(task, c["budget"], row["learned_actions"])
            values = [mse(model.predict(trace["final"]), trace["truth"]) for model in models]
            if not np.isclose(values[0], row["learned_posttrained_mse"], rtol=1e-5, atol=1e-6):
                raise AssertionError("replayed joint context did not reproduce its recorded MSE")
            rows.append({"seed": row["seed"], "joint_mse": values[0], "sft_mse": values[1]})
    return {"tasks": rows, "joint_mse": float(np.mean([r["joint_mse"] for r in rows])),
            "sft_mse": float(np.mean([r["sft_mse"] for r in rows]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="results/agentic_cate/pilot")
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--eval-tasks", type=int, default=64)
    cli = p.parse_args()
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, matched = [], []
    for seed in cli.seeds:
        reports = {}
        for mode in ("joint", "head_only", "sft"):
            path = out / f"{mode}_{seed}.json"
            options = ["--checkpoint", cli.checkpoint, "--mode", mode, "--seed", str(seed),
                       "--steps", str(cli.steps), "--eval-tasks", str(cli.eval_tasks),
                       "--eval-seed-start", "200000", "--output", str(path)]
            if mode == "sft":
                options.extend(["--replay-trajectories", str(out / f"joint_{seed}.json")])
            reports[mode] = run(parser().parse_args(options))
        context = fixed_context_evaluation(reports["joint"], reports["sft"], cli.checkpoint,
                    out / f"joint_{seed}.pt", out / f"sft_{seed}.pt")
        context["optimizer_seed"] = seed
        matched.append(context)
        get = lambda mode, key: reports[mode]["evaluation"]["summary"][key]["mean"]
        metrics = {
            "Frozen / initial": get("joint", "initial_reference_mse"),
            "Frozen / random budget-2": get("joint", "random_reference_mse"),
            "Head-only RL / own policy": get("head_only", "learned_posttrained_mse"),
            "Joint RL / own policy": get("joint", "learned_posttrained_mse"),
            "Frozen / joint policy contexts": get("joint", "learned_reference_mse"),
            "Matched SFT / joint policy contexts": context["sft_mse"],
            "Joint RL / initial": get("joint", "initial_posttrained_mse"),
            "Matched SFT / initial": get("sft", "initial_posttrained_mse"),
            "Joint RL / random budget-2": get("joint", "random_posttrained_mse"),
            "Matched SFT / random budget-2": get("sft", "random_posttrained_mse"),
        }
        all_rows.append({"optimizer_seed": seed, "mse": metrics,
            "joint_acquisitions": get("joint", "learned_acquisitions"),
            "head_only_acquisitions": get("head_only", "learned_acquisitions"),
            "training_forward_calls": {mode: sum(r["model_forward_calls"] for r in report["history"])
                                       for mode, report in reports.items()},
            "gradient_check": reports["joint"]["gradient_check"]})
    aggregate = {name: {"mean": float(np.mean([r["mse"][name] for r in all_rows])),
                         "optimizer_seed_std": float(np.std([r["mse"][name] for r in all_rows], ddof=1))
                         if len(all_rows) > 1 else 0.0}
                 for name in all_rows[0]["mse"]}
    summary = {"status": "preliminary_same_family_CPU_pilot_no_checkpoint_selection",
               "seeds": cli.seeds, "steps": cli.steps, "eval_tasks": cli.eval_tasks,
               "notes": ["All optimizer seeds reuse the same held-out tasks; they are not independent task replications.",
                         "SFT replays exactly the training actions of joint RL; its untrained action head is not evaluated here.",
                         "Equal optimizer steps and trajectories do not imply equal computation; forward calls are recorded.",
                         "Frozen baselines are identical across optimization seeds."],
               "aggregate": aggregate, "runs": all_rows, "matched_context_evaluation": matched}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"aggregate": aggregate, "summary": str(out / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
