"""Review stored S6 summaries and development predictions without test inference.

Run in the existing LCG_108_cuda environment. This reads six 10k validation
prediction files, their histories and manifests. It never opens test responses,
test predictions or checkpoints, and does not fit a model or select a cut.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


KEY = ["run", "chunk", "row"]
NON_TAU = [0, 1, 3]


def metrics(label, probability):
    classes = probability.shape[1]
    confusion = confusion_matrix(label, probability.argmax(1), labels=range(classes))
    counts = confusion.sum(1)
    recalls = np.diag(confusion) / counts
    auc = [roc_auc_score(label == c, probability[:, c]) for c in range(classes)]
    ap = [average_precision_score(label == c, probability[:, c]) for c in range(classes)]
    return {
        "events": len(label),
        "class_counts": counts.tolist(),
        "balanced_accuracy": float(recalls.mean()),
        "NLL": float(-np.log(probability[np.arange(len(label)), label].clip(1e-15)).mean()),
        "macro_AUROC": float(np.mean(auc)),
        "macro_AP": float(np.mean(ap)),
        "class_AUROC": auc,
        "class_AP": ap,
        "confusion_counts": confusion.tolist(),
    }


def conditional_diagnostics(frame, probability):
    label = frame["label"].to_numpy()
    domains = {
        "all": np.ones(len(label), dtype=bool),
        "pixel_centered": frame["pixel_centered"].to_numpy(),
        "large_scintillator": frame["large_scintillator"].to_numpy(),
    }
    result = {}
    for name, domain in domains.items():
        selected = domain & (label != 2)
        three = probability[selected][:, NON_TAU]
        three /= three.sum(1, keepdims=True)
        mapped = np.searchsorted(NON_TAU, label[selected])
        result[name] = {
            "four_class": metrics(label[domain], probability[domain]),
            "conditional_three_class": metrics(mapped, three),
        }
    tau = probability[label == 2][:, NON_TAU].argmax(1)
    result["tau_forced_assignments_when_tau_output_removed"] = np.bincount(tau, minlength=3).tolist()
    return result


def review(repository):
    candidates = json.loads((repository / "roadmap/configs/s6_candidates.json").read_text())
    summary = json.loads((repository / "roadmap/results/s6_summary.json").read_text())
    primary = summary["balanced_models"][summary["primary"]]
    confusion = np.asarray(primary["confusion_raw_counts"])
    expected = np.asarray(summary["samples"]["test_class_counts"])
    assert np.array_equal(confusion.sum(1), expected)
    recomputed_ba = float((np.diag(confusion) / expected).mean())
    assert abs(recomputed_ba - primary["metrics"]["balanced_accuracy"]["value"]) < 1e-12

    output = {
        "UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc descriptive review of previously reused validation; no new training or test-score access",
        "conditional_three_class_definition": "Remove true tau events; renormalize p[e,mu,NC]. Four-class weights remain fixed; this is not three-class retraining.",
        "probability_scope": "Raw saved probabilities normalized in float64; no temperature or prior fit",
        "domain_weighting": "Raw within-domain mixture, with class counts; balanced accuracy always averages class recalls",
        "S6_summary_checks": {
            "test_events_from_confusion": int(confusion.sum()),
            "balanced_accuracy_from_confusion": recomputed_ba,
            "reported_balanced_accuracy": primary["metrics"]["balanced_accuracy"]["value"],
        },
        "training": [],
        "development_diagnostics": {},
    }
    reference = None
    probabilities = {}
    columns = KEY + ["label", "split", "pixel_centered", "large_scintillator"] + [f"p{c}" for c in range(4)]
    for model in candidates["models"]:
        if model["role"] not in ["primary_seed", "supervised_control"]:
            continue
        assert model["config"]["architecture_revision"] == 3
        assert model["config"]["empty_response_bypass"]
        path = Path(model["validation_predictions"])
        frame = pq.read_table(path, columns=columns).to_pandas().sort_values(KEY).reset_index(drop=True)
        assert set(frame["split"]) == {"validation"}
        identity = frame[KEY + ["label", "pixel_centered", "large_scintillator"]]
        if reference is None:
            reference = identity
        else:
            assert reference.equals(identity)
        probability = frame[[f"p{c}" for c in range(4)]].to_numpy(dtype="float64")
        assert np.isfinite(probability).all() and (probability >= 0).all()
        probability /= probability.sum(1, keepdims=True)
        probabilities[model["name"]] = probability
        output["development_diagnostics"][model["name"]] = conditional_diagnostics(frame, probability)
        manifest = json.loads(Path(model["manifest"]).read_text())
        history = json.loads((path.parent / "history.json").read_text())
        timing = manifest["timing_seconds"]
        output["training"].append({
            "name": model["name"],
            "prediction_path": str(path),
            "parameters": manifest["parameters"],
            "train_events": manifest["train_events"],
            "best_epoch": manifest["best_epoch"],
            "epochs": len(history),
            "validation_NLL": [row["validation"]["NLL"] for row in history],
            "last_epoch_NLL_change": history[-1]["validation"]["NLL"] - history[-2]["validation"]["NLL"],
            "timing_seconds": timing,
            "total_seconds": manifest["seconds"],
            "recorded_data_wait_fraction_of_total": timing["data_wait"] / manifest["seconds"],
            "manifest_local_blocks_text": manifest.get("local_blocks"),
            "actual_window_pixels": model["config"]["window_pixels"],
            "manifest_cross_plane_flag": manifest.get("cross_plane_message_passing"),
        })
    for ensemble in candidates["ensembles"]:
        probability = np.mean([probabilities[name] for name in ensemble["members"]], axis=0)
        output["development_diagnostics"][ensemble["name"]] = conditional_diagnostics(frame, probability)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "models": len(result["training"]), "summary_checks": result["S6_summary_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
