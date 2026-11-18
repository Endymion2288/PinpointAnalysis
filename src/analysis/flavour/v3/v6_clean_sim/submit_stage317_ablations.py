"""Launch Stage 317 controlled 3-seed GPU ablations (Exp-1, Exp-3, Exp-5) across Kubeflow & HTCondor.

Strictly operates on Train300k / Tune40k (R2/data) with ZERO access to Cal-A / Cut-B / Confirm200k.
Each GPU allocation runs 3 seeds {17, 42, 2026} concurrently (--parallel 3) sharing local NVMe input cache.

Groups (6 x 3 seeds = 18 models, identical 300k x 24p, bs=128, lr=8e-4, min_lr=1e-5, wd=0.03 [0 on 1D], ema=0.9997):
  1. grp_exp1_pf_v2_ctrl      : PF_V2 anchored (unbiased vertex loss P2-2, 0-WD-1D P2-3, dual-ch legacy for Exp-1)
  2. grp_exp1_pf_v3_ev4_ctrl  : PF_V3 Event4Head (FixedWeightEventLoss, enable_dense_gn2=False, dual-ch legacy for Exp-1)
  3. grp_exp1_pf_v3_gn2_ctrl  : PF_V3 GN2 Dense (FixedWeightEventLoss, enable_dense_gn2=True, dual-ch legacy for Exp-1)
  4. grp_exp3_pf_pe_only      : PF_V3 Event4Head under physical PE-only readout (PE>=2.5, SiPM N_pix=14400, y_mirror=False)
  5. grp_exp5_pf_pe_nomad_phys: PF_V4 Event4Head + calibrated NOMAD (100705/|s_y|, no DRO pAUC) under PE-only readout
  6. grp_exp3_b1pp_pe_only    : B1++ ZAlignedStereoNet under physical PE-only readout (PE>=2.5, SiPM N_pix=14400, y_mirror=False)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.flavour.inventory import write_json
from analysis.flavour.v3.common.submit import submit_job


SEEDS = (17, 42, 2026)
R2_DATA_ROOT = '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data'
TUNE_SCREEN = '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/data/tune_screen.parquet'
SIDECAR_ROOT = '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/truth_sidecar'
DEFAULT_STAGE317_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/stage317_ablations')


def _common_pf_cfg(name: str, seed: int, *, pe_only: bool) -> dict:
    cfg = {
        'name': name,
        'seed': int(seed),
        'task': 'T4',
        'data_root': R2_DATA_ROOT,
        'tune_screen_path': TUNE_SCREEN,
        'sidecar_root': SIDECAR_ROOT,
        'quantize_xy_mm': None,
        'dim': 192,
        'heads': 8,
        'n_local_layers': 4,
        'n_global_layers': 4,
        'drop_path': 0.10,
        'passes': 24,
        'batch_size': 128,
        'learning_rate': 8.0e-4,
        'min_lr': 1.0e-5,
        'weight_decay': 0.03,
        'warmup_fraction': 0.05,
        'ema_decay': 0.9997,
        'bf16': True,
        'screen_every_passes': 4,
        'cache_workers': 3,
        'loader_workers': 0,
        'pe_only': bool(pe_only),
    }
    if pe_only:
        cfg.update({
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': True,
            'y_mirror': False,
        })
    else:
        cfg['y_mirror'] = True
    return cfg


def build_stage317_configs(root: Path) -> dict[str, dict]:
    cfg_dir = root / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict] = {}

    # 1. Exp-1a: PF_V2 anchored (same 300k x 24p schedule)
    g1_paths = []
    for s in SEEDS:
        c = _common_pf_cfg(f'pf_ctrl_v2_anchored_s{s}', s, pe_only=False)
        c.update({
            'use_vertex_anchor': True,
            'vertex_loss_weight': 0.15,
            'legacy_vertex_bias': False,
            'train_module': 'analysis.flavour.v3.v2_pinpointformer.train',
        })
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        g1_paths.append(p)
    groups['s317-exp1-pfv2'] = {
        'configs': g1_paths,
        'share': '--share-pf-inputs',
        'train_module': 'analysis.flavour.v3.v2_pinpointformer.train',
        'cluster_sidecar': False,
        'backend': 'condor',
        'gpu_mem_mb': 38000,
    }

    # 2. Exp-1b: PF_V3 Event4Head (no GN2 dense heads, FixedWeightEventLoss)
    g2_paths = []
    for s in SEEDS:
        c = _common_pf_cfg(f'pf_ctrl_v3_event4head_s{s}', s, pe_only=False)
        c.update({
            'enable_dense_gn2': False,
            'load_cluster_sidecar': False,
            'use_fixed_event_loss': True,
            'train_module': 'analysis.flavour.v3.v3_multitask.train',
        })
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        g2_paths.append(p)
    groups['s317-exp1-ev4'] = {
        'configs': g2_paths,
        'share': '--share-pf-inputs',
        'train_module': 'analysis.flavour.v3.v3_multitask.train',
        'cluster_sidecar': False,
        'backend': 'condor',
        'gpu_mem_mb': 38000,
    }

    # 3. Exp-1c: PF_V3 GN2 Dense (with GN2 dense heads, FixedWeightEventLoss)
    g3_paths = []
    for s in SEEDS:
        c = _common_pf_cfg(f'pf_ctrl_v3_gn2_fixed_s{s}', s, pe_only=False)
        c.update({
            'enable_dense_gn2': True,
            'load_cluster_sidecar': True,
            'use_fixed_event_loss': True,
            'train_module': 'analysis.flavour.v3.v3_multitask.train',
        })
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        g3_paths.append(p)
    groups['s317-exp1-gn2'] = {
        'configs': g3_paths,
        'share': '--share-pf-inputs',
        'train_module': 'analysis.flavour.v3.v3_multitask.train',
        'cluster_sidecar': True,
        'backend': 'kubeflow',
        'flavor': 'h100-47gb',
    }

    # 4. Exp-3a: PF_V3 Event4Head under physical single-channel PE readout (PE >= 2.5 + SiPM sat)
    g4_paths = []
    for s in SEEDS:
        c = _common_pf_cfg(f'pf_pe_only_ev4_s{s}', s, pe_only=True)
        c.update({
            'enable_dense_gn2': False,
            'load_cluster_sidecar': False,
            'use_fixed_event_loss': True,
            'train_module': 'analysis.flavour.v3.v3_multitask.train',
        })
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        g4_paths.append(p)
    groups['s317-exp3-pfpe'] = {
        'configs': g4_paths,
        'share': '--share-pf-inputs',
        'train_module': 'analysis.flavour.v3.v3_multitask.train',
        'cluster_sidecar': False,
        'backend': 'condor',
        'gpu_mem_mb': 38000,
    }

    # 5. Exp-5: PF_V4 Event4Head + Physically Calibrated NOMAD (100705/|s_y|, no DRO pAUC) under PE-only
    g5_paths = []
    for s in SEEDS:
        c = _common_pf_cfg(f'pf_pe_only_nomad_phys_s{s}', s, pe_only=True)
        c.update({
            'model_family': 'pf_v4',
            'enable_dense_gn2': False,
            'load_cluster_sidecar': False,
            'use_fixed_event_loss': True,
            'enable_nomad': True,
            'legacy_nomad_scale': False,
            'enable_pauc': False,
            'nomad_init_scale': 1.0,
            'train_module': 'analysis.flavour.v3.v4_tau.train',
        })
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        g5_paths.append(p)
    groups['s317-exp5-nomad'] = {
        'configs': g5_paths,
        'share': '--share-pf-inputs',
        'train_module': 'analysis.flavour.v3.v4_tau.train',
        'cluster_sidecar': False,
        'backend': 'condor',
        'gpu_mem_mb': 38000,
    }

    # 6. Exp-3b: B1++ ZAlignedStereoNet under physical single-channel PE readout (100% matched to W3-11 b1pp_300k_24p)
    g6_paths = []
    for s in SEEDS:
        c = {
            'name': f'b1pp_pe_only_s{s}',
            'seed': int(s),
            'task': 'T4',
            'data_root': R2_DATA_ROOT,
            'tune_screen_path': TUNE_SCREEN,
            'dim': 160,
            'n_long_layers': 4,
            'heads': 8,
            'drop_path': 0.10,
            'channel_dropout': 0.03,
            'passes': 24,
            'batch_size': 256,
            'learning_rate': 0.0011,
            'min_lr': 1.0e-5,
            'weight_decay': 0.03,
            'warmup_fraction': 0.05,
            'ema_decay': 0.999,
            'bf16': True,
            'screen_every_passes': 4,
            'cache_workers': 3,
            'loader_workers': 0,
            'pe_only': True,
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': True,
            'y_mirror': False,
            'train_module': 'analysis.flavour.v3.v1_b1pp.train',
        }
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        g6_paths.append(p)
    groups['s317-exp3-b1pe'] = {
        'configs': g6_paths,
        'share': '--share-b1pp-inputs',
        'train_module': 'analysis.flavour.v3.v1_b1pp.train',
        'cluster_sidecar': False,
        'backend': 'condor',
        'gpu_mem_mb': 38000,
    }

    return groups


def submit_all(root: Path, dry_run: bool = False) -> dict:
    groups = build_stage317_configs(root)
    submissions = {}
    for gname, spec in groups.items():
        cmd = [
            'analysis.flavour.v3.common.group',
            '--root', str(root),
            '--name', gname,
            '--parallel', '3',
            spec['share'],
            '--cache-workers', '3',
            '--train-module', spec['train_module'],
        ]
        if spec['cluster_sidecar']:
            cmd.extend(['--stage-cluster-sidecar', f'{SIDECAR_ROOT}/cluster_sidecar'])
        cmd.append('--configs')
        cmd.extend(str(p) for p in spec['configs'])

        if dry_run:
            submissions[gname] = {'backend': spec['backend'], 'command': cmd}
            continue

        rec_file = root / 'runtime' / f'{gname}.submission.json'
        if rec_file.exists():
            submissions[gname] = json.loads(rec_file.read_text())
            continue

        if spec['backend'] == 'kubeflow':
            res = submit_job(
                root=root,
                name=gname,
                stage='stage317',
                command=cmd,
                backend='kubeflow',
                flavor=spec['flavor'],
                gpu=True,
                deadline_seconds=28800,
            )
        else:
            res = submit_job(
                root=root,
                name=gname,
                stage='stage317',
                command=cmd,
                backend='condor',
                cpus=4,
                memory='13GB',
                disk='18GB',
                gpu=True,
                min_gpu_memory_mb=14500,
                min_capability=7.5,
                job_flavour='workday',
            )
        submissions[gname] = res

    write_json(root / 'stage317_submissions.json', submissions)
    return submissions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_STAGE317_ROOT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    out = submit_all(args.root, dry_run=args.dry_run)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
