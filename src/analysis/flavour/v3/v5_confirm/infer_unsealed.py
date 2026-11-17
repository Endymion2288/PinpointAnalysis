"""Post-freeze batched shared-representation GPU inference, Probes A/B, and 14-condition stress evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

from analysis.flavour.datasets import CachedNativeDataset
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.s6.conditions import Condition
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import device_batch, init_worker, shutdown_loader
from analysis.flavour.v2.r1.tasks import probabilities
from analysis.flavour.v3.common.datasets import V3EvalDataset
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.common.probes import compute_probe_a, compute_probe_b, run_model_with_entropy
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import collate_b1pp, extract_b1pp_item, static_plane_layout
from analysis.flavour.v3.v1_b1pp.train import _configure_runtime
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import make_pf_loader, move_batch_to_device
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v3_multitask.train import _pf_v3_predict
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4, ZAlignedStereoNetV4
from analysis.flavour.v3.v5_confirm import DEFAULT_S6_ROOT, DEFAULT_V3_ROOT
from analysis.flavour.v3.v5_confirm.freeze import require_v5_freeze


def _build_model_from_freeze_entry(entry: dict, geometry: dict, layout: dict | None, device: torch.device):
    cfg = dict(entry['config'])
    role = entry['role']
    state = torch.load(entry['checkpoint_path'], map_location='cpu', weights_only=False)
    if role == 'pf_v3_gn2':
        model = PinpointFormerMultitask(geometry, cfg)
    elif role in ('pf_v4_nomad_only', 'pf_v4_nomad_pauc'):
        model = PinpointFormerV4(geometry, cfg)
    elif role == 'b1pp_v3':
        model = ZAlignedStereoNet(layout, cfg)
    elif role == 'b1pp_v4_nomad_pauc':
        model = ZAlignedStereoNetV4(layout, cfg, geometry=geometry)
    else:
        raise ValueError(f'Unknown freeze role: {role}')
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval()


class _StressVariantDataset(Dataset):
    """Apply all 14 Condition transforms paired to Confirm200k stress events on demand."""

    def __init__(
        self,
        data_root: Path,
        stress_manifest: pd.DataFrame,
        variants: list[dict],
        geometry: dict,
        bounds: dict,
        hardware_map: dict,
        family: str,
        staged_dir: Path | None = None,
    ) -> None:
        self.geometry = geometry
        self.bounds = bounds
        self.family = family
        self.variants = variants
        self.transforms = [Condition(geometry, v['parameters'], hardware_map) for v in variants]

        paths = sorted(p for p in (data_root / 'shards').iterdir() if (p / 'observations.parquet').exists())
        native_paths = [staged_dir / p.name for p in paths] if staged_dir else paths
        native = CachedNativeDataset(native_paths, max_shards=1, target_columns=('label',))

        # Map exact KEY -> native index and ideal tracker row once in RAM (8,000 events is compact)
        meta_rows = []
        for si, sh_path in enumerate(paths):
            pr = pd.read_parquet(sh_path / 'provenance.parquet', columns=KEY + FAMILY + ['split', 'label'])
            pr['_native_idx'] = native.boundaries[si] + np.arange(len(pr), dtype=np.int64)
            pr['_shard_idx'] = si
            pr['_local_idx'] = np.arange(len(pr), dtype=np.int64)
            meta_rows.append(pr)
        all_prov = pd.concat(meta_rows, ignore_index=True)
        pos = pd.MultiIndex.from_frame(all_prov[KEY]).get_indexer(pd.MultiIndex.from_frame(stress_manifest[KEY]))
        if (pos < 0).any():
            raise ValueError('stress_manifest rows missing from confirm shards')
        picked = all_prov.iloc[pos].reset_index(drop=True)

        recs = picked.to_dict('records')
        n_picked = len(recs)
        base_obs: list[dict | None] = [None] * n_picked
        base_ideal: list[dict | None] = [None] * n_picked
        order = np.argsort(picked['_native_idx'].to_numpy(dtype=np.int64), kind='mergesort')
        cur_s_idx = -1
        cur_ideal: list[dict] = []
        for pos_i in order:
            rec = recs[int(pos_i)]
            n_idx = int(rec['_native_idx'])
            s_idx = int(rec['_shard_idx'])
            l_idx = int(rec['_local_idx'])
            if s_idx != cur_s_idx:
                ideal_pq = native_paths[s_idx] / 'ideal_tracker.parquet'
                if not ideal_pq.exists():
                    ideal_pq = paths[s_idx] / 'ideal_tracker.parquet'
                cur_ideal = pq.read_table(ideal_pq).to_pylist()
                cur_s_idx = s_idx
            base_obs[int(pos_i)] = native[n_idx]['observations']
            base_ideal[int(pos_i)] = cur_ideal[l_idx]
        native.close()
        self.base_obs = base_obs  # type: ignore[assignment]
        self.base_ideal = base_ideal  # type: ignore[assignment]
        self.base_meta = recs

        self.meta_out = picked.loc[picked.index.repeat(len(variants))].reset_index(drop=True)
        self.meta_out['variant'] = np.tile([v['name'] for v in variants], len(picked))

    def __len__(self) -> int:
        return len(self.base_obs) * len(self.variants)

    def __getitem__(self, idx: int) -> dict:
        ev_idx = idx // len(self.transforms)
        v_idx = idx % len(self.transforms)
        meta = self.base_meta[ev_idx]
        changed = self.transforms[v_idx](self.base_obs[ev_idx], meta, self.base_ideal[ev_idx])
        dom = compute_observable_containment(changed, self.geometry, self.bounds)
        label = int(meta['label'])
        if self.family == 'pf':
            item = extract_pf_item(idx, changed, label, dom, self.geometry, self.bounds, {'quantize_xy_mm': None})
        else:
            item = extract_b1pp_item(idx, changed, label, dom, self.geometry, self.bounds, {})
        item['obs_domain_id'] = int(dom['obs_domain_id'])
        return item


def _eval_pf_models_once(
    loader: DataLoader,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    use_bf16: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    n_total = len(loader.dataset)
    probs_out = {name: np.zeros((n_total, 4), dtype=np.float32) for name in models}
    charm_out = {name: np.zeros(n_total, dtype=np.float32) for name in models}
    obs_dom_out = np.zeros(n_total, dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            idx = batch['index'].numpy()
            obs_dom_out[idx] = batch['obs_domain_id'].numpy()
            db = move_batch_to_device(batch, device)
            for name, model in models.items():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    pr = _pf_v3_predict(model, db)
                probs_out[name][idx] = pr['probs'].cpu().numpy()
                charm_out[name][idx] = pr['charm_prob'].cpu().numpy()
    return probs_out, charm_out, obs_dom_out


def _eval_b1pp_models_once(
    loader: DataLoader,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    use_bf16: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_total = len(loader.dataset)
    probs_out = {name: np.zeros((n_total, 4), dtype=np.float32) for name in models}
    obs_dom_out = np.zeros(n_total, dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            idx = batch['index'].numpy()
            obs_dom_out[idx] = batch['obs_domain_id'].numpy()
            gpu = device_batch(batch, str(device))
            for name, model in models.items():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(gpu).float()
                probs_out[name][idx] = probabilities(logits, 'T4').cpu().numpy()
    return probs_out, obs_dom_out


def _run_pf_probes(
    confirm_dir: Path,
    staged_root: Path,
    stress_manifest_path: Path,
    stress_df: pd.DataFrame,
    models: dict[str, torch.nn.Module],
    geometry: dict,
    bounds: dict,
    device: torch.device,
    use_bf16: bool,
    workers: int,
    nominal_stress_pf: pd.DataFrame,
) -> dict:
    """Run Confirm200k Probe A, Probe B (1mm quantization), and leave-one-modality-out sensitivity."""
    old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
    os.environ['PINPOINT_V3_SHARED_STAGE'] = str(staged_root)
    try:
        # 1) Probe A (entropy & tau@1% across occupancy) using pf_v4_nomad_pauc_s42 and pf_v3_gn2_s42
        ds_nat = V3EvalDataset(
            confirm_dir,
            split='confirm',
            config={'quantize_xy_mm': None, 'key_filter_path': str(stress_manifest_path), 'cache_shards': 2},
            item_transform=extract_pf_item,
        )
        ld_nat = make_pf_loader(ds_nat, batch_size=96, shuffle=False, workers=workers, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
        probe_a = {}
        try:
            labels = ds_nat.metadata['label'].to_numpy(dtype=np.int64)
            fam_codes, _ = pd.factorize(pd.MultiIndex.from_frame(ds_nat.metadata[FAMILY]))
            for mname in ('pf_v3_gn2_s42', 'pf_v4_nomad_pauc_s42'):
                ent_res = run_model_with_entropy(models[mname], ld_nat, device, use_bf16=use_bf16)
                probe_a[mname] = compute_probe_a(
                    ent_res['probs'],
                    labels,
                    ent_res['raw_pixel_hits'],
                    ent_res['mean_entropy'],
                    learned_scales=ent_res['learned_scales'],
                    layer_entropy=ent_res['layer_entropy'],
                )
        finally:
            shutdown_loader(ld_nat)
            ds_nat.native.close()

        # 2) Probe B (1 mm pixel quantization sensitivity) on Truth_pixel_centered inside stress_manifest
        ds_q1 = V3EvalDataset(
            confirm_dir,
            split='confirm',
            config={'quantize_xy_mm': 1.0, 'key_filter_path': str(stress_manifest_path), 'cache_shards': 2},
            item_transform=extract_pf_item,
        )
        ld_q1 = make_pf_loader(ds_q1, batch_size=96, shuffle=False, workers=workers, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
        try:
            q1_probs, _, _ = _eval_pf_models_once(ld_q1, {k: models[k] for k in ('pf_v3_gn2_s42', 'pf_v4_nomad_pauc_s42')}, device, use_bf16)
        finally:
            shutdown_loader(ld_q1)
            ds_q1.native.close()

        pos_pix = pd.MultiIndex.from_frame(ds_q1.metadata[KEY]).get_indexer(pd.MultiIndex.from_frame(stress_df[KEY]))
        if (pos_pix < 0).any():
            raise ValueError('stress_df keys missing in Probe B alignment')
        pix_cen_mask = stress_df['pixel_centered'].to_numpy(dtype=bool)
        pos_nom = pd.MultiIndex.from_frame(nominal_stress_pf[KEY]).get_indexer(pd.MultiIndex.from_frame(stress_df[KEY]))
        labels_s = stress_df['label'].to_numpy(dtype=np.int64)
        fam_s, _ = pd.factorize(pd.MultiIndex.from_frame(stress_df[FAMILY]))
        probe_b = {}
        for mname in ('pf_v3_gn2_s42', 'pf_v4_nomad_pauc_s42'):
            pn = nominal_stress_pf.iloc[pos_nom][[f'{mname}__p{c}' for c in range(4)]].to_numpy(dtype=np.float64)
            pq_m = q1_probs[mname][pos_pix]
            probe_b[mname] = compute_probe_b(labels_s, pn, pq_m, pix_cen_mask, family_keys=fam_s, n_boot=400)

        # 3) Leave-one-modality-out delta-NLL
        mod_drops = {}
        base_nll = float(-np.log(np.clip(
            nominal_stress_pf.iloc[pos_nom][[f'pf_v4_nomad_pauc_s42__p{c}' for c in range(4)]].to_numpy(dtype=np.float64)[np.arange(len(labels_s)), labels_s],
            1e-8, 1.0,
        )).mean())
        for mod in ('pixel', 'bar', 'tracker'):
            ds_d = V3EvalDataset(
                confirm_dir,
                split='confirm',
                config={'drop_modality': mod, 'key_filter_path': str(stress_manifest_path), 'cache_shards': 2},
                item_transform=extract_pf_item,
            )
            ld_d = make_pf_loader(ds_d, batch_size=96, shuffle=False, workers=workers, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
            try:
                dp, _, _ = _eval_pf_models_once(ld_d, {'pf_v4_nomad_pauc_s42': models['pf_v4_nomad_pauc_s42']}, device, use_bf16)
            finally:
                shutdown_loader(ld_d)
                ds_d.native.close()
            pos_d = pd.MultiIndex.from_frame(ds_d.metadata[KEY]).get_indexer(pd.MultiIndex.from_frame(stress_df[KEY]))
            if (pos_d < 0).any():
                raise ValueError(f'stress_df keys missing in modality drop {mod}')
            dnll = float(-np.log(np.clip(dp['pf_v4_nomad_pauc_s42'][pos_d][np.arange(len(labels_s)), labels_s], 1e-8, 1.0)).mean())
            mod_drops[mod] = {'nll': dnll, 'delta_nll': float(dnll - base_nll)}

        return {'probe_a': probe_a, 'probe_b': probe_b, 'modality_ablation_pf_v4_s42': {'nominal_nll': base_nll, 'drops': mod_drops}}
    finally:
        if old_stage is None:
            os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
        else:
            os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage


def _run_b1pp_probes(
    confirm_dir: Path,
    staged_root: Path,
    stress_manifest_path: Path,
    stress_df: pd.DataFrame,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    use_bf16: bool,
    workers: int,
    nominal_stress_b1: pd.DataFrame,
) -> dict:
    """Compute B1++-V4 leave-one-modality-out sensitivity on Confirm200k."""
    old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
    os.environ['PINPOINT_V3_SHARED_STAGE'] = str(staged_root)
    try:
        pos_nom = pd.MultiIndex.from_frame(nominal_stress_b1[KEY]).get_indexer(pd.MultiIndex.from_frame(stress_df[KEY]))
        labels_s = stress_df['label'].to_numpy(dtype=np.int64)
        base_p = nominal_stress_b1.iloc[pos_nom][[f'b1pp_v4_nomad_pauc_s42__p{c}' for c in range(4)]].to_numpy(dtype=np.float64)
        base_nll = float(-np.log(np.clip(base_p[np.arange(len(labels_s)), labels_s], 1e-8, 1.0)).mean())
        drops = {}
        for mod in ('pixel', 'bar', 'tracker'):
            ds_d = V3EvalDataset(
                confirm_dir,
                split='confirm',
                config={'drop_modality': mod, 'key_filter_path': str(stress_manifest_path), 'cache_shards': 2},
                item_transform=extract_b1pp_item,
            )
            ld_d = DataLoader(
                ds_d,
                batch_sampler=BatchSampler(SequentialSampler(ds_d), batch_size=128, drop_last=False),
                num_workers=workers,
                collate_fn=collate_b1pp,
                pin_memory=True,
                **(dict(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2) if workers > 0 else {}),
            )
            try:
                dp, _ = _eval_b1pp_models_once(ld_d, {'b1pp_v4_nomad_pauc_s42': models['b1pp_v4_nomad_pauc_s42']}, device, use_bf16)
            finally:
                shutdown_loader(ld_d)
                ds_d.native.close()
            pos_d = pd.MultiIndex.from_frame(ds_d.metadata[KEY]).get_indexer(pd.MultiIndex.from_frame(stress_df[KEY]))
            dnll = float(-np.log(np.clip(dp['b1pp_v4_nomad_pauc_s42'][pos_d][np.arange(len(labels_s)), labels_s], 1e-8, 1.0)).mean())
            drops[mod] = {'nll': dnll, 'delta_nll': float(dnll - base_nll)}
        return {'modality_ablation_b1pp_v4_s42': {'nominal_nll': base_nll, 'drops': drops}}
    finally:
        if old_stage is None:
            os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
        else:
            os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage


def run_family_inference(
    *,
    family: str,
    v3_root: Path = DEFAULT_V3_ROOT,
    s6_root: Path = DEFAULT_S6_ROOT,
    splits: tuple[str, ...] = ('cal_a', 'cut_b', 'confirm'),
    workers: int = 4,
    batch_size: int | None = None,
    freeze_path: Path | None = None,
) -> dict:
    """Run shared-representation inference across all frozen checkpoints of one family ('pf' or 'b1pp')."""
    _configure_runtime()
    w3_12 = v3_root / 'W3-12'
    if freeze_path is None:
        freeze_path = w3_12 / 'analysis_freeze_v5.json'
    freeze = require_v5_freeze(freeze_path, verify_git_commit=True, verify_checkpoints=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_bf16 = device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8
    plan_obj = json.loads((w3_12 / 'preparation_plan.json').read_text())
    geometry = json.loads(Path(plan_obj['geometry']).read_text())
    bounds = detector_bounds(geometry)
    layout = static_plane_layout(geometry, bounds) if family == 'b1pp' else None
    if layout is not None:
        layout['geometry'] = geometry

    entries = [m for m in freeze['members'] if m['family'] == family]
    models = {m['member_id']: _build_model_from_freeze_entry(m, geometry, layout, device) for m in entries}
    bs = int(batch_size or (128 if family == 'b1pp' else 96))

    pred_root = w3_12 / 'predictions'
    parent_scratch = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(parent_scratch).mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {'family': family, 'models': list(models.keys()), 'splits': {}}

    with tempfile.TemporaryDirectory(prefix=f'v5-infer-{family}-', dir=parent_scratch) as tmp_dir:
        tmp = Path(tmp_dir)
        for split in splits:
            split_dir = w3_12 / f'data_{split}'
            out_dir = pred_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            exp_events = int(plan_obj['splits'][split]['events'])
            todo = [
                mid for mid in models
                if not (out_dir / f'{mid}.parquet').exists() or pq.read_metadata(out_dir / f'{mid}.parquet').num_rows != exp_events
            ]
            if not todo and split != 'confirm':
                report['splits'][split] = {'events': exp_events, 'evaluated_models': 0, 'seconds': 0.0, 'cached': True}
                continue
            shard_paths = sorted(p for p in (split_dir / 'shards').iterdir() if (p / 'observations.parquet').exists())
            stage = stage_shards(shard_paths, tmp / f'stage_{split}', workers=min(4, max(1, workers)))
            staged_dir = Path(stage['local_shard_root'])
            old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
            os.environ['PINPOINT_V3_SHARED_STAGE'] = str(staged_dir)
            t0 = time.perf_counter()
            try:
                if todo:
                    sub_models = {mid: models[mid] for mid in todo}
                    ds = V3EvalDataset(
                        split_dir,
                        split=split,
                        config={'quantize_xy_mm': None, 'cache_shards': 2},
                        item_transform=extract_pf_item if family == 'pf' else extract_b1pp_item,
                    )
                    if family == 'pf':
                        ld = make_pf_loader(ds, batch_size=bs, shuffle=False, workers=workers, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
                        try:
                            probs_map, charm_map, _ = _eval_pf_models_once(ld, sub_models, device, use_bf16)
                        finally:
                            shutdown_loader(ld)
                            ds.native.close()
                    else:
                        ld = DataLoader(
                            ds,
                            batch_sampler=BatchSampler(SequentialSampler(ds), batch_size=bs, drop_last=False),
                            num_workers=workers,
                            collate_fn=collate_b1pp,
                            pin_memory=True,
                            **(dict(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2) if workers > 0 else {}),
                        )
                        try:
                            probs_map, _ = _eval_b1pp_models_once(ld, sub_models, device, use_bf16)
                            charm_map = {}
                        finally:
                            shutdown_loader(ld)
                            ds.native.close()

                    base_meta = ds.metadata[KEY].copy().reset_index(drop=True)
                    for mid in todo:
                        df_m = base_meta.copy()
                        pm = probs_map[mid]
                        for c in range(4):
                            df_m[f'p_{c}'] = pm[:, c]
                        if mid in charm_map:
                            df_m['charm_prob'] = charm_map[mid]
                        df_m.to_parquet(out_dir / f'{mid}.parquet', index=False, compression=COMPRESSION)

                report['splits'][split] = {'events': exp_events, 'evaluated_models': len(todo), 'seconds': float(time.perf_counter() - t0)}

                # Confirm-only paired 14-variant stress & Probes A/B
                if split == 'confirm':
                    stress_manifest_path = w3_12 / 'stress_manifest.parquet'
                    stress_df = pd.read_parquet(stress_manifest_path).sort_values(KEY).reset_index(drop=True)
                    stress_out_pq = pred_root / f'stress_{family}.parquet'
                    if not stress_out_pq.exists():
                        cond_cfg = json.loads((s6_root / 'conditions.json').read_text())
                        hw_map = json.loads((s6_root / 'audit' / 'bar_hardware_map.json').read_text())
                        var_ds = _StressVariantDataset(
                            split_dir,
                            stress_df,
                            cond_cfg['variants'],
                            geometry,
                            bounds,
                            hw_map,
                            family=family,
                            staged_dir=staged_dir,
                        )
                        if family == 'pf':
                            vld = make_pf_loader(var_ds, batch_size=bs, shuffle=False, workers=workers, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
                            try:
                                v_probs, v_charm, v_dom = _eval_pf_models_once(vld, models, device, use_bf16)
                            finally:
                                shutdown_loader(vld)
                        else:
                            vld = DataLoader(
                                var_ds,
                                batch_sampler=BatchSampler(SequentialSampler(var_ds), batch_size=bs, drop_last=False),
                                num_workers=workers,
                                collate_fn=collate_b1pp,
                                pin_memory=True,
                                **(dict(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2) if workers > 0 else {}),
                            )
                            try:
                                v_probs, v_dom = _eval_b1pp_models_once(vld, models, device, use_bf16)
                                v_charm = {}
                            finally:
                                shutdown_loader(vld)
                        st_out = var_ds.meta_out[KEY + ['label', 'variant']].copy().reset_index(drop=True)
                        st_out['obs_domain_id'] = v_dom.astype(np.int64)
                        for mid, pm in v_probs.items():
                            for c in range(4):
                                st_out[f'{mid}__p{c}'] = pm[:, c]
                            if mid in v_charm:
                                st_out[f'{mid}__charm'] = v_charm[mid]
                        st_out.to_parquet(stress_out_pq, index=False, compression=COMPRESSION)
                    else:
                        st_out = pd.read_parquet(stress_out_pq)

                    nom_df = st_out[st_out['variant'] == 'nominal'].reset_index(drop=True)
                    shard_ordered_keys_pq = tmp / 'stress_keys_shard_order.parquet'
                    if 'shard' in stress_df.columns:
                        stress_df_shard_order = stress_df.sort_values(['shard'] + KEY).reset_index(drop=True)
                    else:
                        prov_rows = []
                        for si, sp in enumerate(shard_paths):
                            pr = pd.read_parquet(sp / 'provenance.parquet', columns=KEY)
                            pr['_si'] = si
                            prov_rows.append(pr)
                        ap = pd.concat(prov_rows, ignore_index=True)
                        mpos = pd.MultiIndex.from_frame(ap[KEY]).get_indexer(pd.MultiIndex.from_frame(stress_df[KEY]))
                        stress_df_shard_order = stress_df.iloc[np.argsort(mpos, kind='mergesort')].reset_index(drop=True)
                    stress_df_shard_order[KEY].to_parquet(shard_ordered_keys_pq, index=False, compression=COMPRESSION)
                    if family == 'pf':
                        probes = _run_pf_probes(
                            split_dir, staged_dir, shard_ordered_keys_pq, stress_df, models, geometry, bounds, device, use_bf16, workers, nom_df
                        )
                        write_json(w3_12 / 'probes_confirm_pf.json', probes)
                    else:
                        probes = _run_b1pp_probes(
                            split_dir, staged_dir, shard_ordered_keys_pq, stress_df, models, device, use_bf16, workers, nom_df
                        )
                        write_json(w3_12 / 'probes_confirm_b1pp.json', probes)
            finally:
                if old_stage is None:
                    os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
                else:
                    os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage
                shutil.rmtree(staged_dir, ignore_errors=True)

    report['UTC_finished'] = datetime.now(timezone.utc).isoformat()
    report['peak_GPU_MiB'] = float(torch.cuda.max_memory_allocated() / 2**20) if device.type == 'cuda' else 0.0
    report['peak_RSS_MiB'] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    write_json(w3_12 / f'infer_{family}_summary.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('pf', 'b1pp'), required=True)
    parser.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument('--s6-root', type=Path, default=DEFAULT_S6_ROOT)
    parser.add_argument('--splits', nargs='+', default=['cal_a', 'cut_b', 'confirm'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--freeze-path', type=Path, default=None)
    args = parser.parse_args()
    rep = run_family_inference(
        family=args.family,
        v3_root=args.v3_root,
        s6_root=args.s6_root,
        splits=tuple(args.splits),
        workers=args.workers,
        batch_size=args.batch_size,
        freeze_path=args.freeze_path,
    )
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
