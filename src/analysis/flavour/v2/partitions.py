"""Build family-disjoint v2 manifests from accepted metadata and research use.

This is a batch metadata reduction, not another production certification.
No detector response or Confirm score/target payload is read.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.inventory import COMPRESSION, write_json


KEY = ["run", "chunk", "row"]
FAMILY = ["family_hi", "family_lo"]
ROLES = {"fit_or_SSL": 1, "validation_or_probe": 2, "calibration": 4,
         "published_test": 8, "pilot_or_manual": 16, "prepared_reserve": 32}
SPLIT_ROLE = {"train": 1, "validation": 2, "calibration": 4, "test": 8}


def row_keys(frame):
    """Lossless integer packing for the existing accepted run/chunk/row range."""
    values = [frame[c].to_numpy(dtype="uint64") for c in KEY]
    if any(np.any(v >= limit) for v, limit in zip(values, [2**16, 2**28, 2**20])):
        raise ValueError("row identity exceeds the declared packing range")
    return values[0] << np.uint64(48) | values[1] << np.uint64(20) | values[2]


def family_index(frame):
    return pd.MultiIndex.from_frame(frame[FAMILY])


def equal_identity_rows(left, right):
    """Compare integer columns separately; mixed signed/uint64 arrays lose bits."""
    left = left.sort_values(KEY)
    right = right.sort_values(KEY)
    return len(left) == len(right) and all(np.array_equal(left[column].to_numpy(), right[column].to_numpy())
                                          for column in KEY+FAMILY)


def discover_sources(base):
    """Corpus closure: all prior stage manifests/catalogs plus S0 manual data.

    S3 and S5 use the S2/S4 corpora. Resolve symlinks to avoid duplicate I/O;
    include smoke corpora and every S6 original-event perturbation manifest.
    Generator hard scatters have their own namespace, recorded separately.
    """
    paths = set()
    for stage in sorted(base.glob("s[0-6]*")):
        for folder in [stage, stage / "smoke", stage / "extension", stage / "h100"]:
            for name in ["manifest.parquet", "catalog.parquet", "calibration_catalog.parquet",
                         "test_catalog.parquet", "transport_manifest.parquet", "variants_manifest.parquet"]:
                path = folder / name
                if path.exists():
                    paths.add(path.resolve())
    for kind in ["shards", "smoke"]:
        for path in (base / "s0_s1_2026-11-11" / kind).glob("*/provenance.parquet"):
            paths.add(path.resolve())
    for path in [(base/'s6_2026-11-14/transport/manifest.parquet'),
                 *sorted((base/'s6_2026-11-14/audit').glob('smoke_*/provenance.parquet'))]:
        if path.exists():
            paths.add(path.resolve())
    return sorted(paths)


def collect_usage(frame, sources):
    """Return row masks; every family touched by any role is excluded later."""
    index = pd.Index(row_keys(frame))
    if not index.is_unique:
        raise ValueError("accepted observation keys are not unique")
    masks = np.zeros(len(frame), dtype="uint8")
    records = []
    for path in sources:
        columns = pq.read_schema(path).names
        if not set(KEY).issubset(columns):
            raise ValueError(f"historical identity table lacks row keys: {path}")
        selected_columns = KEY + [c for c in FAMILY + ["split"] if c in columns]
        old = pd.read_parquet(path, columns=selected_columns)
        positions = index.get_indexer(row_keys(old))
        if np.any(positions < 0):
            raise ValueError(f"historical accepted rows absent from inventory: {path}")
        if set(FAMILY).issubset(old):
            for column in FAMILY:
                if not np.array_equal(frame[column].to_numpy()[positions], old[column].to_numpy()):
                    raise ValueError(f"historical/accepted family mismatch: {path}")
        role = (old["split"].map(SPLIT_ROLE).fillna(32).to_numpy(dtype="uint8")
                if "split" in old else np.full(len(old), ROLES["pilot_or_manual"], dtype="uint8"))
        np.bitwise_or.at(masks, positions, role)
        records.append({"path": str(path), "events": len(old),
                        "roles": {name: int(np.count_nonzero(role & bit)) for name, bit in ROLES.items()}})
    # Old calibration/test are public or fitted as whole partitions, including
    # any future-discovered response replica not explicitly in a corpus table.
    for name in ["calibration", "test"]:
        masks[frame.old_split.eq(name)] |= SPLIT_ROLE[name]
    return masks, records


def select_families(frame, count, rng, *, chunk_order=False):
    """Choose one canonical row per family, retaining every response's grouping.

    Evaluation uses seeded event permutations. Training first shuffles source
    chunks, then rows, giving dense ROOT reads without sorting on physics.
    """
    if chunk_order:
        chunks = frame[KEY[:2]].drop_duplicates()
        chunks = chunks.iloc[rng.permutation(len(chunks))].copy()
        chunks["chunk_rank"] = np.arange(len(chunks))
        ordered = frame.merge(chunks, on=KEY[:2], validate="many_to_one")
        ordered["random_rank"] = rng.random(len(ordered))
        ordered = ordered.sort_values(["chunk_rank", "random_rank"])
        ordered = ordered.drop(columns=["chunk_rank", "random_rank"])
    else:
        ordered = frame.iloc[rng.permutation(len(frame))]
    ordered = ordered.drop_duplicates(FAMILY)
    if len(ordered) < count:
        raise ValueError(f"need {count} unused families, only {len(ordered)} available")
    return ordered.iloc[:count].copy()


def make_partitions(frame, masks, seed, train_sizes, reserve_per_class):
    family_codes, families = pd.factorize(family_index(frame), sort=False)
    family_masks = np.zeros(len(families), dtype="uint8")
    np.bitwise_or.at(family_masks, family_codes, masks)
    used = family_masks[family_codes] != 0
    eligible = frame.usable_join & frame.label_status.eq("consistent")
    unused = frame.loc[eligible & ~used & frame.old_split.isin(["train", "validation"])].copy()
    # Cross-label families cannot be allocated as independent balanced rows.
    # Keep their whole family out of this round, with an explicit count.
    canonical = unused.drop_duplicates(FAMILY + ["label"])
    ambiguous = family_index(canonical).duplicated(keep=False)
    bad_families = family_index(canonical.loc[ambiguous])
    unused = unused.loc[~family_index(unused).isin(bad_families)]
    rng = np.random.default_rng(seed)
    partitions, availability = {}, []
    for label in range(4):
        pool = unused.loc[unused.label.eq(label)]
        availability.append({"label": label, "unused_rows": len(pool),
                             "unused_families": len(pool.drop_duplicates(FAMILY))})
        for name, per_class in reserve_per_class.items():
            picked = select_families(pool, per_class, rng)
            partitions.setdefault(name, []).append(picked)
            pool = pool.loc[~family_index(pool).isin(family_index(picked))]
        maximum = select_families(pool.loc[pool.old_split.eq("train")], max(train_sizes)//4,
                                  rng, chunk_order=True)
        maximum["class_train_rank"] = np.arange(len(maximum), dtype="int32")
        for size in train_sizes:
            partitions.setdefault(f"train_{size}", []).append(maximum.iloc[:size//4].copy())
    result = {name: pd.concat(rows, ignore_index=True).sort_values(KEY).reset_index(drop=True)
              for name, rows in partitions.items()}
    summary = {"availability": availability, "ambiguous_unused_families": len(bad_families.unique()),
               "historically_used_rows": int(used.sum()), "historically_used_families": int((family_masks != 0).sum()),
               "accepted_inventory_rows": len(frame), "accepted_inventory_families": len(families)}
    ledger = pd.DataFrame({"family_hi": families.get_level_values(0), "family_lo": families.get_level_values(1),
                           "usage_mask": family_masks})
    ledger = ledger.loc[ledger.usage_mask.ne(0)].reset_index(drop=True)
    return result, ledger, summary


def audit_partitions(partitions, ledger):
    train = sorted([n for n in partitions if n.startswith("train_")], key=lambda n: int(n.split("_")[1]))
    used = family_index(ledger)
    checks = {}
    for name, rows in partitions.items():
        checks[f"{name}_families_unique"] = not family_index(rows).has_duplicates
        checks[f"{name}_not_historical"] = not family_index(rows).isin(used).any()
        checks[f"{name}_balanced"] = set(rows.label.unique()) == set(range(4)) and rows.groupby("label").size().nunique() == 1
    distinct = [train[-1]] + [n for n in partitions if not n.startswith("train_")]
    for i, name in enumerate(distinct):
        for other in distinct[i+1:]:
            checks[f"{name}_disjoint_{other}"] = not family_index(partitions[name]).isin(family_index(partitions[other])).any()
    for small, large in zip(train[:-1], train[1:]):
        checks[f"{small}_nested_in_{large}"] = family_index(partitions[small]).isin(family_index(partitions[large])).all()
    if not all(checks.values()):
        raise ValueError(f"partition audit failed: {[k for k,v in checks.items() if not v]}")
    return {k: bool(v) for k, v in checks.items()}


def build(args):
    if (args.root/'partitions.json').exists():
        raise ValueError('frozen partitions already exist; use a new output root for a new allocation')
    started = time.perf_counter()
    pa.set_cpu_count(args.cpus)
    pa.set_io_thread_count(args.cpus)
    config = json.loads(args.config.read_text())
    columns = KEY + FAMILY + ["evtID", "label", "label_status", "usable_join", "split", "geometry_id",
                              "n_pixel", "n_bar", "n_tracker"]
    frame = pd.read_parquet(config["inventory"], columns=columns).rename(columns={"split": "old_split"})
    sources = discover_sources(Path(config["historical_root"]))
    masks, evidence = collect_usage(frame, sources)
    partitions, ledger, summary = make_partitions(frame, masks, config["seed"], config["train_sizes"], config["reserve_per_class"])
    checks = audit_partitions(partitions, ledger)
    folder = args.root / "manifests"
    sealed = args.root / "sealed"
    folder.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(mode=0o700, exist_ok=True)
    sealed.chmod(0o700)
    ledger.to_parquet(folder / "usage_ledger.parquet", index=False, compression=COMPRESSION)
    outputs, counts = {}, {}
    for name, rows in partitions.items():
        rows["split"] = "train" if name.startswith("train_") else name
        columns_out = KEY + FAMILY + ["evtID", "label", "geometry_id", "split", "old_split", "n_pixel", "n_bar", "n_tracker"]
        if "class_train_rank" in rows:
            columns_out.append("class_train_rank")
        path = (sealed / "confirm_rows.parquet") if name == "confirm" else folder / f"{name}.parquet"
        rows[columns_out].to_parquet(path, index=False, compression=COMPRESSION)
        if name == "confirm":
            path.chmod(0o600)
            identities = rows[FAMILY].copy()
            identities.insert(0, "confirm_index", np.arange(len(identities)))
            identities.to_parquet(folder / "confirm.parquet", index=False, compression=COMPRESSION)
        outputs[name] = {"manifest": str(folder / "confirm.parquet" if name == "confirm" else path),
                         "row_label_access": str(path), "events": len(rows)}
        counts[name] = {str(k): int(v) for k, v in rows.groupby("label").size().items()}
    plan = {"status": "frozen", "stage": "R0", "UTC": datetime.now(timezone.utc).isoformat(),
            "config": config, "manifests": outputs, "counts": counts, "summary": summary,
            "historical_sources": evidence, "usage_roles": ROLES, "checks": checks,
            "train_order": "seeded chunk permutation then within-chunk random rows; nested per-class prefixes",
            "confirm_access": "workflow blind: public identities only; sealed rows/labels reserved for R7; no Confirm responses read",
            "inventory_only_events": "schema/count inventory does not constitute model development use",
            "historical_initializations": "all selected v2 partitions exclude the entire previous fit/SSL/validation/calibration/test corpus closure",
            "seconds": time.perf_counter()-started,
            "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    write_json(args.root / "partitions.json", plan)
    print(json.dumps({"root": str(args.root), "counts": counts, "summary": summary, "checks": len(checks)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=4)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
