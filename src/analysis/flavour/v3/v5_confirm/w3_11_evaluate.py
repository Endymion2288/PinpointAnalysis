"""Evaluate W3-11 3-seed (17, 42, 2026) stability on Tune40k and prepare V5 freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.v2_pinpointformer.evaluate import load_tune_reference_frame
from analysis.flavour.v3.v4_tau.ensemble import extract_member_arrays, fit_oof_domain_ensemble
from analysis.flavour.v3.v4_tau.evaluate import (
    _build_slices,
    _eval_member_df,
    _primary_tag_from_manifest,
    paired_block_bootstrap_delta,
)
from analysis.flavour.v3.v5_confirm import (
    DEFAULT_V3_ROOT,
    ROLE_ORDER,
    SEEDS,
    experiment_dir_for,
)
from analysis.flavour.v3.v5_confirm.freeze import (
    build_v5_freeze_manifest,
    save_v5_freeze_manifests,
)


def _load_tune_predictions(exp_dir: Path) -> tuple[str, pd.DataFrame]:
    tag = _primary_tag_from_manifest(exp_dir)
    pq_path = exp_dir / ('best_predictions.parquet' if tag == 'best' else 'predictions.parquet')
    if not pq_path.exists():
        pq_path = exp_dir / 'predictions.parquet'
        tag = 'ema'
    return tag, pd.read_parquet(pq_path)


def _average_member_frames(frames: list[pd.DataFrame], ref_df: pd.DataFrame) -> pd.DataFrame:
    """Average predicted class probabilities (and charm_prob) across aligned frames."""
    aligned = [extract_member_arrays(df, ref_df) for df in frames]
    mean_probs = np.mean([a['probs'] for a in aligned], axis=0)
    mean_probs = mean_probs / np.maximum(mean_probs.sum(axis=-1, keepdims=True), 1e-12)
    out = ref_df[KEY].copy().reset_index(drop=True)
    for c in range(4):
        out[f'p_{c}'] = mean_probs[:, c].astype(np.float32)
    charms = [a['charm'] for a in aligned if a['charm'] is not None]
    if charms:
        out['charm_prob'] = np.mean(charms, axis=0).astype(np.float32)
    return out


def _aggregate_seed_stats(per_seed_evals: dict[int, dict]) -> dict:
    """Compute mean +/- std (ddof=1) across seeds for key slices and OOF tau metrics."""
    summary: dict[str, dict] = {}
    slice_names = ('All', 'D_pix_contained', 'D_scint_contained', 'Truth_pixel_centered', 'Truth_large_scintillator', 'Tune_E_le_2p3TeV')
    metric_names = ('nll', 'BA4', 'cond_BA3', 'tau_at_1pct_fpr', 'tau_at_0p1pct_fpr', 'auroc_tau')
    for sname in slice_names:
        s_dict = {}
        for mname in metric_names:
            vals = [per_seed_evals[s]['slices'][sname].get(mname) for s in SEEDS]
            arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
            if len(arr):
                s_dict[mname] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    'per_seed': {str(s): float(per_seed_evals[s]['slices'][sname].get(mname, float('nan'))) for s in SEEDS},
                }
        summary[sname] = s_dict

    reg_dict = {}
    for key in ('reg_tau_1pct_fpr', 'reg_tau_0p1pct_fpr', 'np_1pct_all', 'np_0p1pct_all', 'np_1pct_scint', 'asimov_z0_oof_18bin_multidomain'):
        vals = [per_seed_evals[s]['regional_calibration'].get(key) for s in SEEDS]
        arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
        if len(arr):
            reg_dict[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                'per_seed': {str(s): float(per_seed_evals[s]['regional_calibration'].get(key, float('nan'))) for s in SEEDS},
            }
    summary['regional_calibration'] = reg_dict
    return summary


def evaluate_w3_11(
    v3_root: Path = DEFAULT_V3_ROOT,
    *,
    n_boot: int = 400,
    write_freeze: bool = True,
) -> dict:
    """Evaluate all 15 Train300k checkpoints (3 seeds x 5 roles) and OOF ensembles on Tune40k."""
    w3_11_dir = v3_root / 'W3-11'
    w3_12_dir = v3_root / 'W3-12'
    pred_dir = w3_11_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    w3_12_dir.mkdir(parents=True, exist_ok=True)

    ref_df = load_tune_reference_frame()
    slices = _build_slices(ref_df)
    fam_codes, _ = pd.factorize(pd.MultiIndex.from_frame(ref_df[FAMILY]))
    folds, fold_summary = grouped_folds(
        ref_df[KEY + FAMILY + ['label']].reset_index(drop=True), folds=5, seed=20261117
    )
    labels = ref_df['label'].to_numpy(dtype=np.int64)

    member_frames: dict[str, dict[int, pd.DataFrame]] = {role: {} for role in ROLE_ORDER}
    per_checkpoint: dict[str, dict] = {}
    role_seed_evals: dict[str, dict[int, dict]] = {role: {} for role in ROLE_ORDER}

    for role in ROLE_ORDER:
        for seed in SEEDS:
            exp_dir = experiment_dir_for(v3_root, role, seed)
            tag, df = _load_tune_predictions(exp_dir)
            ev, _, _ = _eval_member_df(df, ref_df, slices, fam_codes, n_boot=n_boot)
            ev['checkpoint_tag'] = tag
            ev['experiment_dir'] = str(exp_dir)
            member_frames[role][seed] = df
            per_checkpoint[f'{role}_s{seed}'] = ev
            role_seed_evals[role][seed] = ev

    role_seed_summary = {role: _aggregate_seed_stats(role_seed_evals[role]) for role in ROLE_ORDER}

    # Seed-averaged standalone models on Tune40k
    role_3seed_frames: dict[str, pd.DataFrame] = {}
    role_3seed_evals: dict[str, dict] = {}
    for role in ROLE_ORDER:
        avg_df = _average_member_frames([member_frames[role][s] for s in SEEDS], ref_df)
        out_pq = pred_dir / f'{role}_3seed_avg.parquet'
        avg_df.to_parquet(out_pq, index=False, compression=COMPRESSION)
        ev, _, _ = _eval_member_df(avg_df, ref_df, slices, fam_codes, n_boot=n_boot)
        role_3seed_frames[role] = avg_df
        role_3seed_evals[f'{role}_3seed_avg'] = ev

    def _strip_ens(res: dict) -> dict:
        return {k: v for k, v in res.items() if k not in ('oof_probs', 's_tau', 's_raw')}

    def _extract_ens_metric(sl: dict, mname: str):
        if mname == 'np_1pct_tau_eff':
            return sl.get('np_1pct', {}).get('tau_eff')
        if mname == 'np_0p1pct_tau_eff':
            return sl.get('np_0p1pct', {}).get('tau_eff')
        return sl.get(mname)

    # Per-seed OOF Dual-Regime 5-role Ensemble + 3-seed stability
    per_seed_ensembles: dict[str, dict] = {}
    seed_ens_probs: dict[int, np.ndarray] = {}
    seed_ens_stau: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        m_dict = {role: member_frames[role][seed] for role in ROLE_ORDER}
        res = fit_oof_domain_ensemble(
            m_dict,
            ref_df,
            slices,
            n_folds=5,
            folds=folds,
            save_path=pred_dir / f'ens_v4_dual_regime_s{seed}.parquet',
        )
        epq = pd.read_parquet(pred_dir / f'ens_v4_dual_regime_s{seed}.parquet')
        seed_ens_probs[seed] = epq[['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(dtype=np.float64)
        seed_ens_stau[seed] = epq['s_tau'].to_numpy(dtype=np.float64)
        per_seed_ensembles[f'ens_v4_dual_regime_s{seed}'] = _strip_ens(res)

    ens_seed_stability = {}
    for sname in ('All', 'D_pix_contained', 'D_scint_contained', 'Truth_pixel_centered', 'Truth_large_scintillator', 'Tune_E_le_2p3TeV'):
        ens_seed_stability[sname] = {}
        for mname in ('nll', 'BA4', 'cond_BA3', 'within_slice_S_tau_1pct', 'within_slice_S_tau_0p1pct', 'np_1pct_tau_eff', 'np_0p1pct_tau_eff'):
            raw_vals = {
                str(s): _extract_ens_metric(per_seed_ensembles[f'ens_v4_dual_regime_s{s}']['slices'][sname], mname)
                for s in SEEDS
            }
            arr = np.array([v for v in raw_vals.values() if v is not None and np.isfinite(v)], dtype=np.float64)
            if len(arr):
                ens_seed_stability[sname][mname] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    'per_seed': {k: float(v) if v is not None else float('nan') for k, v in raw_vals.items()},
                }

    z0_vals = [per_seed_ensembles[f'ens_v4_dual_regime_s{s}']['asimov_z0_oof_18bin_multidomain'] for s in SEEDS]
    ens_seed_stability['asimov_z0_oof_18bin_multidomain'] = {
        'mean': float(np.mean(z0_vals)),
        'std': float(np.std(z0_vals, ddof=1)),
        'per_seed': {str(s): float(z0_vals[i]) for i, s in enumerate(SEEDS)},
    }

    # 3-Seed Flagship Dual-Regime OOF Ensemble combining all 5 seed-averaged roles (15 checkpoints)
    ens_3seed_res = _strip_ens(
        fit_oof_domain_ensemble(
            role_3seed_frames,
            ref_df,
            slices,
            n_folds=5,
            folds=folds,
            save_path=pred_dir / 'ens_v5_3seed_dual_regime.parquet',
        )
    )
    ens3_df = pd.read_parquet(pred_dir / 'ens_v5_3seed_dual_regime.parquet')
    ens3_probs = ens3_df[['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(dtype=np.float64)
    ens3_stau = ens3_df['s_tau'].to_numpy(dtype=np.float64)

    paired_vs_s42 = {
        sname: paired_block_bootstrap_delta(
            labels, ens3_probs, seed_ens_probs[42], fam_codes, slices[sname], s_new=ens3_stau, s_base=seed_ens_stau[42], n_boot=n_boot, seed=42
        )
        for sname in ('All', 'D_scint_contained', 'D_pix_contained', 'Truth_pixel_centered', 'Tune_E_le_2p3TeV')
    }

    # Check quantitative stability gate: no seed > 0.03 degradation in All BA4 vs seed=42
    max_ba4_drop = 0.0
    for role in ROLE_ORDER:
        ba4_42 = per_checkpoint[f'{role}_s42']['slices']['All']['BA4']
        for seed in (17, 2026):
            max_ba4_drop = max(max_ba4_drop, float(ba4_42 - per_checkpoint[f'{role}_s{seed}']['slices']['All']['BA4']))
    if max_ba4_drop > 0.03:
        raise RuntimeError(f'W3-11 stability check failed: max All BA4 drop vs seed=42 is {max_ba4_drop:.4f} > 0.03')

    out = {
        'status': 'complete',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'split': 'Tune40k',
        'sealed_payloads_touched': False,
        'seeds': list(SEEDS),
        'roles': list(ROLE_ORDER),
        'folds': fold_summary,
        'per_checkpoint': per_checkpoint,
        'role_3seed_stability': role_seed_summary,
        'role_3seed_averaged_models': role_3seed_evals,
        'per_seed_ensembles': per_seed_ensembles,
        'ensemble_3seed_stability': ens_seed_stability,
        'ens_v5_3seed_dual_regime': ens_3seed_res,
        'paired_delta_ens_v5_vs_ens_s42': paired_vs_s42,
        'max_all_ba4_drop_vs_seed42': max_ba4_drop,
    }
    summary_path = w3_11_dir / 'w3_11_multiseed_results.json'
    write_json(summary_path, out)

    if write_freeze:
        repo_root = Path(__file__).resolve().parents[5]
        freeze = build_v5_freeze_manifest(
            v3_root=v3_root,
            repo_root=repo_root,
            w3_11_summary_path=summary_path,
            pre_unsealing_git_commit=None,
        )
        save_v5_freeze_manifests(
            freeze,
            eos_freeze_path=w3_12_dir / 'analysis_freeze_v5.json',
            repo_freeze_path=repo_root / 'roadmap/v3/v5_freeze_manifest.json',
        )
    return out


def stamp_pre_unsealing_commit(commit: str, v3_root: Path = DEFAULT_V3_ROOT) -> dict:
    """Write the verified pre-unsealing Git commit SHA into the EOS V5 freeze JSON."""
    eos_freeze = v3_root / 'W3-12' / 'analysis_freeze_v5.json'
    data = json.loads(eos_freeze.read_text())
    data['pre_unsealing_git_commit'] = str(commit).strip()
    data['freeze_stamped_UTC'] = datetime.now(timezone.utc).isoformat()
    write_json(eos_freeze, data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument('--n-boot', type=int, default=400)
    parser.add_argument('--no-freeze', action='store_true')
    parser.add_argument('--stamp-commit', default=None, help='Stamp pre-unsealing Git commit into EOS freeze JSON only')
    args = parser.parse_args()

    if args.stamp_commit:
        stamped = stamp_pre_unsealing_commit(args.stamp_commit, v3_root=args.v3_root)
        print(json.dumps({'status': stamped['status'], 'pre_unsealing_git_commit': stamped['pre_unsealing_git_commit']}, indent=2))
        return

    res = evaluate_w3_11(v3_root=args.v3_root, n_boot=args.n_boot, write_freeze=not args.no_freeze)
    e5 = res['ens_v5_3seed_dual_regime']['slices']
    compact = {
        'status': res['status'],
        'max_all_ba4_drop_vs_seed42': res['max_all_ba4_drop_vs_seed42'],
        'ens_v5_All': e5['All'],
        'ens_v5_D_scint_contained': e5['D_scint_contained'],
        'ens_v5_Truth_pixel_centered': e5['Truth_pixel_centered'],
        'ens_v5_asimov_z0_18bin': res['ens_v5_3seed_dual_regime']['asimov_z0_oof_18bin_multidomain'],
    }
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
