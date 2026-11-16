"""Neutrino energy spectrum truncation diagnostics and matched-energy reweighting (W3-01).

Raw catalog.parquet / ROOT event.initE is stored in MeV (nu_tau max = 2,311,800 MeV
= 2.3118 TeV; e/mu/NC up to 5,041,200 MeV = 5.0412 TeV). All slice keys and bin edges
in this module operate explicitly in GeV after converting energy_nu_GeV = initE / 1000.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import FAMILY, KEY


COMMON_ENERGY_MAX_GEV = 2300.0
FULL_ENERGY_MIN_GEV = 10.0
FULL_ENERGY_MAX_GEV = 5500.0


def compute_energy_match_weights(
    energy_gev: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 20,
    e_min_gev: float = FULL_ENERGY_MIN_GEV,
    e_max_common_gev: float = COMMON_ENERGY_MAX_GEV,
) -> tuple[np.ndarray, dict]:
    """Compute per-event weights matching all 4 classes to min_c f_c(b) within <= 2.30 TeV."""
    edges = np.geomspace(e_min_gev, e_max_common_gev, n_bins + 1)
    in_common = (energy_gev >= e_min_gev) & (energy_gev <= e_max_common_gev)
    bin_idx = np.clip(np.digitize(energy_gev, edges) - 1, 0, n_bins - 1)

    counts = np.zeros((4, n_bins), dtype=np.int64)
    densities = np.zeros((4, n_bins), dtype=np.float64)
    for c in range(4):
        mask = (labels == c) & in_common
        cnt = np.bincount(bin_idx[mask], minlength=n_bins)[:n_bins]
        counts[c] = cnt
        tot = max(int(cnt.sum()), 1)
        densities[c] = cnt / float(tot)

    min_env = densities.min(axis=0)
    weights = np.zeros(len(energy_gev), dtype=np.float32)
    for c in range(4):
        mask = (labels == c) & in_common
        ratio = np.divide(
            min_env,
            densities[c],
            out=np.zeros(n_bins, dtype=np.float64),
            where=densities[c] > 0,
        )
        w_c = ratio[bin_idx[mask]]
        if w_c.sum() > 0:
            w_c = w_c * (len(w_c) / w_c.sum())
        weights[mask] = w_c.astype(np.float32)

    info = {
        'edges_GeV': edges.tolist(),
        'common_max_GeV': e_max_common_gev,
        'class_counts_in_common': counts.tolist(),
        'min_envelope_mass': float(min_env.sum()),
        'effective_fraction_by_class': [
            float((weights[labels == c] > 0).mean()) if (labels == c).any() else 0.0
            for c in range(4)
        ],
    }
    return weights, info


def run_w3_01_spectrum(
    catalog_path: Path,
    manifests_root: Path,
    output_dir: Path,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = KEY + FAMILY + ['evtID', 'label', 'split', 'initE', 'pixel_centered', 'large_scintillator']
    cat = pd.read_parquet(catalog_path, columns=cols)
    cat['energy_nu_GeV'] = cat['initE'].astype(np.float64) / 1000.0

    tau_max = float(cat.loc[cat['label'] == 2, 'energy_nu_GeV'].max())
    if not (2000.0 <= tau_max <= 2500.0):
        raise ValueError(f'unexpected nu_tau max energy {tau_max} GeV (expected ~2311.8 GeV)')

    train = cat[cat['split'] == 'train'].sort_values(KEY).reset_index(drop=True)
    tune = cat[cat['split'] == 'validation'].sort_values(KEY).reset_index(drop=True)

    # Full 20-bin diagnostic over [10, 5500] GeV
    full_edges = np.geomspace(FULL_ENERGY_MIN_GEV, FULL_ENERGY_MAX_GEV, 21)
    train_bin = np.clip(np.digitize(train['energy_nu_GeV'].to_numpy(), full_edges) - 1, 0, 19)
    full_hist = {
        str(c): np.bincount(train_bin[train['label'].to_numpy() == c], minlength=20).tolist()
        for c in range(4)
    }

    # Train300k & Train100k matched weights (<= 2.30 TeV)
    w300, w300_info = compute_energy_match_weights(
        train['energy_nu_GeV'].to_numpy(),
        train['label'].to_numpy(),
    )
    train_weights = train[KEY + FAMILY + ['label', 'energy_nu_GeV']].copy()
    train_weights['energy_match_weight'] = w300

    m100 = pd.read_parquet(manifests_root / 'train_100000.parquet', columns=KEY)
    m100_idx = pd.MultiIndex.from_frame(m100[KEY])
    in_100k = pd.MultiIndex.from_frame(train[KEY]).isin(m100_idx)
    train100 = train.loc[in_100k].reset_index(drop=True)
    w100, w100_info = compute_energy_match_weights(
        train100['energy_nu_GeV'].to_numpy(),
        train100['label'].to_numpy(),
    )
    train100_weights = train100[KEY + FAMILY + ['label', 'energy_nu_GeV']].copy()
    train100_weights['energy_match_weight'] = w100

    # Tune slices
    tune_common = tune[tune['energy_nu_GeV'] <= COMMON_ENERGY_MAX_GEV].copy()
    tune_highe_nontau = tune[(tune['energy_nu_GeV'] > COMMON_ENERGY_MAX_GEV) & (tune['label'] != 2)].copy()

    train_weights.to_parquet(output_dir / 'train300k_spectrum_weights.parquet', index=False, compression=COMPRESSION)
    train100_weights.to_parquet(output_dir / 'train100k_spectrum_weights.parquet', index=False, compression=COMPRESSION)
    tune_common[KEY + FAMILY + ['label', 'energy_nu_GeV', 'pixel_centered', 'large_scintillator']].to_parquet(
        output_dir / 'tune_E_le_2p3TeV_keys.parquet', index=False, compression=COMPRESSION
    )
    tune_highe_nontau[KEY + FAMILY + ['label', 'energy_nu_GeV', 'pixel_centered', 'large_scintillator']].to_parquet(
        output_dir / 'highE_gt_2p3TeV_non_tau_keys.parquet', index=False, compression=COMPRESSION
    )

    summary = {
        'status': 'complete',
        'seconds': time.perf_counter() - started,
        'common_energy_max_GeV': COMMON_ENERGY_MAX_GEV,
        'train300k_energy_stats_GeV': {
            str(c): {
                'count': int((train['label'] == c).sum()),
                'min': float(train.loc[train['label'] == c, 'energy_nu_GeV'].min()),
                'median': float(train.loc[train['label'] == c, 'energy_nu_GeV'].median()),
                'max': float(train.loc[train['label'] == c, 'energy_nu_GeV'].max()),
                'frac_le_2p3TeV': float((train.loc[train['label'] == c, 'energy_nu_GeV'] <= COMMON_ENERGY_MAX_GEV).mean()),
            }
            for c in range(4)
        },
        'tune40k_energy_stats_GeV': {
            str(c): {
                'count': int((tune['label'] == c).sum()),
                'min': float(tune.loc[tune['label'] == c, 'energy_nu_GeV'].min()),
                'median': float(tune.loc[tune['label'] == c, 'energy_nu_GeV'].median()),
                'max': float(tune.loc[tune['label'] == c, 'energy_nu_GeV'].max()),
                'count_le_2p3TeV': int(((tune['label'] == c) & (tune['energy_nu_GeV'] <= COMMON_ENERGY_MAX_GEV)).sum()),
                'count_gt_2p3TeV': int(((tune['label'] == c) & (tune['energy_nu_GeV'] > COMMON_ENERGY_MAX_GEV)).sum()),
            }
            for c in range(4)
        },
        'tune_E_le_2p3TeV_total': int(len(tune_common)),
        'highE_gt_2p3TeV_non_tau_total': int(len(tune_highe_nontau)),
        'full_20bin_edges_GeV': full_edges.tolist(),
        'train300k_full_20bin_counts': full_hist,
        'train300k_common_reweight': w300_info,
        'train100k_common_reweight': w100_info,
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(output_dir / 'spectrum_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--catalog',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data/catalog.parquet'),
    )
    parser.add_argument(
        '--manifests-root',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/manifests'),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01'),
    )
    args = parser.parse_args()
    summary = run_w3_01_spectrum(args.catalog, args.manifests_root, args.output)
    print(json.dumps({
        'status': summary['status'],
        'tune_E_le_2p3TeV_total': summary['tune_E_le_2p3TeV_total'],
        'highE_gt_2p3TeV_non_tau_total': summary['highE_gt_2p3TeV_non_tau_total'],
        'seconds': summary['seconds'],
    }, indent=2))


if __name__ == '__main__':
    main()
