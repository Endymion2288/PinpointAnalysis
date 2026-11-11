"""Read a bounded ROOT sample for roadmap evidence; never rewrite input data.

Requires numpy, awkward and uproot. The default reads three geometry/header
samples per run, scalar truth from chunk 000, and 32 hit entries from chunk 000.
It does not measure the total event count of the full production.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


RUN_LABELS = {10081: "CC_nue", 10082: "CC_numu", 10083: "CC_nutau", 10084: "NC"}
HIT_FIELDS = {
    "Hits/pixelHits": ["event_id", "layerID", "rowID", "colID", "edep"],
    "Hits/scintHits": [
        "event_id", "layerID", "panelID", "rowID", "colID", "isHorizontal",
        "edep", "nPE", "hitTime",
    ],
    "Hits/faserHits": ["event_id", "trackerID", "x", "y", "z"],
}


def quantiles(values):
    array = np.asarray(values)
    return dict(zip(["min", "p50", "p95", "max"], np.quantile(array, [0, .5, .95, 1]).tolist())) if array.size else None


def frequencies(values):
    unique, counts = np.unique(ak.to_numpy(values), return_counts=True)
    return {str(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}


def geometry_summary(record):
    result = {}
    for key, value in record.items():
        array = np.asarray(value)
        if array.ndim == 0:
            result[key] = array.item()
        else:
            result[key] = {
                "length": len(array), "first": array[:3].tolist(), "last": array[-3:].tolist(),
                "step_median": float(np.median(np.diff(array))) if len(array) > 1 else None,
            }
    return result


def inspect_run(base, run, sample_events):
    files = sorted(base.joinpath(str(run)).glob("*.root"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    chunk_ids = [int(p.stem.rsplit("_", 1)[1]) for p in files]
    result = {
        "nominal_label": RUN_LABELS[run], "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "chunk_id_min": min(chunk_ids), "chunk_id_max": max(chunk_ids),
        "missing_chunk_ids": sorted(set(range(min(chunk_ids), max(chunk_ids) + 1)) - set(chunk_ids)),
        "headers": [],
    }
    reference_geometry = None
    for index in sorted(set([0, len(files) // 2, len(files) - 1])):
        with uproot.open(files[index]) as root:
            geom = ak.to_list(root["geometry"].arrays(library="ak"))[0]
            if reference_geometry is None:
                reference_geometry = geom
                result["geometry"] = geometry_summary(geom)
            result["headers"].append({
                "file": files[index].name,
                "entries": {key: root[key].num_entries for key in ["event", *HIT_FIELDS]},
                "geometry_matches_first": geom == reference_geometry,
                "tree_cycles": root.keys(),
                "branches": {key: root[key].keys() for key in ["event", *HIT_FIELDS]},
            })

    with uproot.open(files[0]) as root:
        fields = ["evtID", "vtxID", "initPDG", "fslPDG", "intType", "processName", "initE", "initX", "initY", "initZ", "weight"]
        truth = root["event"].arrays(fields, library="ak")
        result["truth_sample_file"] = files[0].name
        result["truth_rows"] = len(truth)
        result["truth_counts"] = {key: frequencies(truth[key]) for key in ["initPDG", "fslPDG", "intType", "processName", "weight", "vtxID"]}
        result["truth_ranges_raw_units"] = {key: quantiles(ak.to_numpy(truth[key])) for key in ["initE", "initX", "initY", "initZ"]}
        expected_nu = {10081: 12, 10082: 14, 10083: 16}
        if run == 10084:
            consistent = (truth["intType"] == 3) & (abs(truth["fslPDG"]) == abs(truth["initPDG"]))
        else:
            nu = expected_nu[run]
            consistent = (truth["intType"] == 2) & (abs(truth["initPDG"]) == nu) & (abs(truth["fslPDG"]) == nu - 1)
        result["label_metadata_disagreements"] = ak.to_list(truth[~consistent][["evtID", "vtxID", "initPDG", "fslPDG", "intType", "processName"]])

        result["hits"] = {}
        for tree, fields in HIT_FIELDS.items():
            available = [key for key in fields if key in root[tree].keys()]
            hits = root[tree].arrays(available, entry_stop=sample_events, library="ak")
            count_field = "trackerID" if tree == "Hits/faserHits" else "layerID"
            counts = ak.to_numpy(ak.num(hits[count_field], axis=1))
            ids = ak.to_numpy(hits["event_id"])
            # A row-alignment diagnostic only; no automatic event-ID repair.
            mismatched = ids != ak.to_numpy(truth["evtID"][:len(ids)])
            stats = {
                "sample_entries": len(hits), "record_count": int(counts.sum()),
                "records_per_event": quantiles(counts), "zero_record_entries": int((counts == 0).sum()),
                "row_event_id_disagreements": int(mismatched.sum()),
                "id_disagreement_examples": [
                    {"row": int(i), "event_evtID": int(truth["evtID"][i]), "hit_event_id": int(ids[i]), "records": int(counts[i])}
                    for i in np.flatnonzero(mismatched)[:8]
                ],
            }
            if tree == "Hits/pixelHits":
                stats["edep_raw_MeV"] = quantiles(ak.to_numpy(ak.flatten(hits["edep"])))
            if tree == "Hits/scintHits":
                channel_fields = ["layerID", "panelID", "rowID", "colID", "isHorizontal"]
                channel_counts = []
                for event in hits:
                    channels = np.array([ak.to_numpy(event[key]) for key in channel_fields]).T
                    channel_counts.append(len(np.unique(channels, axis=0)))
                stats["unique_channel_count"] = sum(channel_counts)
                stats["channels_per_event"] = quantiles(channel_counts)
                for key in ["nPE", "hitTime"]:
                    if key in available:
                        stats[key] = {
                            "range": quantiles(ak.to_numpy(ak.flatten(hits[key]))),
                            "negative_records": int(ak.sum(hits[key] < 0)),
                            "zero_records": int(ak.sum(hits[key] == 0)),
                        }
            result["hits"][tree] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/eos/project-f/fasersim-bonn/public/pinpoint/data"))
    parser.add_argument("--runs", type=int, nargs="+", default=list(RUN_LABELS))
    parser.add_argument("--sample-events", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True, help="JSON output under the task results directory")
    args = parser.parse_args()
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "input": str(args.input),
        "scope": "directory sizes; three headers/geometries per run; chunk 000 scalar truth; bounded hit sample",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "uproot": uproot.__version__, "awkward": ak.__version__},
        "runs": {str(run): inspect_run(args.input, run, args.sample_events) for run in args.runs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
    for run, info in report["runs"].items():
        print(run, info["file_count"], info["bytes"], "bytes", "label disagreements", len(info["label_metadata_disagreements"]))


if __name__ == "__main__":
    main()
