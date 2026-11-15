"""Snapshot stored R2 screens and audit low-occupancy confusion arithmetic.

This reads six small manifests only. AP/AUROC come from stored screen histories;
they are not independently recomputed from event predictions. No checkpoint,
response, calibration or confirmation payload is opened.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def review(root):
    campaign = json.loads((root / "campaign.json").read_text())
    models = []
    checks = 0
    reference_counts = None
    screen_paths = set()
    for name in campaign["initial_models"]:
        source = root / "experiments" / name / "manifest.json"
        record = json.loads(source.read_text())
        config = record["config"]
        screen_paths.add(config["tune_screen"])
        history = []
        for row in record["history"]:
            domain = row["occupancy_below_100"]
            counts = domain["raw_counts"]
            if reference_counts is None:
                reference_counts = counts
            assert counts == reference_counts
            metrics = domain["metrics"]["T3"]
            confusion = metrics["confusion_counts"]
            denominators = [sum(cells) for cells in confusion]
            assert denominators == [counts[c] for c in (0, 1, 3)]
            assert metrics["raw_counts"] == denominators
            recalls = [confusion[c][c] / denominators[c] for c in range(3)]
            for expected, measured in zip(recalls, metrics["per_class"]):
                assert math.isclose(expected, measured["recall"], abs_tol=1e-12, rel_tol=0)
            assert math.isclose(sum(recalls) / 3, metrics["BA"], abs_tol=1e-12, rel_tol=0)
            checks += 7
            predicted_mu = sum(cells[1] for cells in confusion)
            mu = metrics["per_class"][1]
            history.append({
                "non_tau_exposures": row["non_tau_exposures"],
                "updates": row["updates"],
                "lr": row["lr"],
                "all_NLL3": row["metrics"]["T3"]["NLL"],
                "all_BA3": row["metrics"]["T3"]["BA"],
                "low_NLL3": metrics["NLL"],
                "low_BA3": metrics["BA"],
                "low_recall_e_mu_NC": recalls,
                "low_mu_AP": mu["AP"],
                "low_mu_AUROC": mu["AUROC"],
                "low_mu_efficiency_at_1pct_FPR": mu["eff_FPR_0.01"],
                "low_mu_predicted_fraction": predicted_mu / sum(denominators),
                "low_mu_precision": confusion[1][1] / predicted_mu if predicted_mu else None,
                "low_confusion_counts_e_mu_NC": confusion,
            })
        models.append({
            "name": name, "design": config["design"], "task": config["task"],
            "seed": config["seed"], "status_at_snapshot": record["status"],
            "source": str(source),
            "last_recorded_counters": record["counters"],
            "history": history,
        })
    assert len(screen_paths) == 1
    checks += 1
    non_tau_events = sum(reference_counts[c] for c in (0, 1, 3))
    return {
        "UTC": datetime.now(timezone.utc).isoformat(),
        "status": "descriptive_progress_snapshot_not_R2_acceptance",
        "scope": "Initial six models; fixed4k reused Tune screens; no new inference or fitting",
        "source_root": str(root), "tune_screen": next(iter(screen_paths)),
        "low_occupancy_definition": "accepted pixel+bar+tracker channels < 100; includes empty events",
        "low_raw_counts_e_mu_tau_NC": reference_counts,
        "common_non_tau_events": non_tau_events,
        "mu_AP_prevalence_reference": reference_counts[1] / non_tau_events,
        "ranking_metrics_source": "Stored history only; AP/AUROC not independently recomputed",
        "arithmetic_checks_passed": checks,
        "Cal_A_Cut_B_Confirm_payload_read": False,
        "models": models,
    }


def plot(report, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    colors = {"D0": "#5f7f95", "D1": "#d17b0f", "D2": "#288b69"}
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for model in report["models"]:
        rows = model["history"]
        x = [row["non_tau_exposures"] / 1e6 for row in rows]
        series = [[row["low_recall_e_mu_NC"][1] for row in rows],
                  [row["low_mu_AP"] for row in rows],
                  [row["low_mu_AUROC"] for row in rows]]
        for axis, values in zip(axes, series):
            axis.plot(x, values, "-" if model["task"] == "T3" else "--", marker=".",
                      color=colors[model["design"]], label=f'{model["design"]} {model["task"]}')
    axes[0].set_ylim(0, 1)
    axes[1].axhline(report["mu_AP_prevalence_reference"], color="gray", linestyle=":",
                   label="Muon prevalence")
    axes[2].axhline(0.5, color="gray", linestyle=":", label="Random ranking")
    for axis, title in zip(axes, ["Muon argmax recall", "Muon average precision", "Muon AUROC"]):
        axis.set(title=title, xlabel="Non-tau training exposures (million)", ylabel="Score")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("Occupancy < 100; fixed development events; descriptive training snapshot", fontsize=10)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(folder / f"r2_sparse_muon_progress.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    report = review(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if args.figures:
        plot(report, args.figures)
    print(json.dumps({"output": str(args.output), "models": len(report["models"]),
                      "arithmetic_checks_passed": report["arithmetic_checks_passed"]}))


if __name__ == "__main__":
    main()
