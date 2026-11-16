"""Strictly isolated V3 datasets: V3EvalDataset has zero access to truth sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from analysis.flavour.datasets import CachedNativeDataset
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r1.tasks import NON_TAU
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds


# Provenance columns allowed in V3EvalDataset (identity + split only, zero truth kinematics/vertex).
EVAL_PROVENANCE_COLUMNS = tuple(KEY + FAMILY + ['split', 'label'])


class V3EvalDataset(Dataset):
    """Evaluation and pure-observation dataset; signature never accepts truth sidecars."""

    def __init__(
        self,
        obs_root: Path | str,
        split: str = 'validation',
        config: dict | None = None,
        cache_dir: Path | str | None = None,
        item_transform=None,
    ) -> None:
        self.root = Path(obs_root)
        self.split = split
        self.config = dict(config or {})
        self.item_transform = item_transform

        plan = json.loads((self.root / 'plan.json').read_text())
        self.geometry = json.loads(Path(plan['geometry']).read_text())
        self.bounds = detector_bounds(self.geometry)

        self.paths = sorted(
            p for p in (self.root / 'shards').iterdir()
            if (p / 'observations.parquet').exists()
        )
        staged = os.environ.get('PINPOINT_V3_SHARED_STAGE') or os.environ.get('PINPOINT_NATIVE_SHARD_ROOT')
        native_paths = [Path(staged) / p.name for p in self.paths] if staged else self.paths
        self.native = CachedNativeDataset(
            native_paths,
            cache_dir=cache_dir,
            max_shards=int(self.config.get('cache_shards', 2)),
        )

        indexes: list[int] = []
        meta_frames: list[pd.DataFrame] = []
        for i, path in enumerate(self.paths):
            frame = pd.read_parquet(path / 'provenance.parquet', columns=list(EVAL_PROVENANCE_COLUMNS))
            keep = frame['split'].eq(split)
            idx = np.flatnonzero(keep.to_numpy())
            if len(idx):
                indexes.extend((self.native.boundaries[i] + idx).tolist())
                meta_frames.append(frame.iloc[idx])

        self.indexes = np.asarray(indexes, dtype=np.int64)
        self.metadata = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame(columns=list(EVAL_PROVENANCE_COLUMNS))

        # Optional deterministic subsetting for Train100k or T3 non-tau training
        mask = np.ones(len(self.metadata), dtype=bool)
        if split == 'train' and self.config.get('manifest_subset'):
            sub_df = pd.read_parquet(self.config['manifest_subset'], columns=KEY)
            sub_idx = pd.MultiIndex.from_frame(sub_df[KEY])
            mask &= pd.MultiIndex.from_frame(self.metadata[KEY]).isin(sub_idx)
        if split == 'train' and self.config.get('task') == 'T3':
            mask &= self.metadata['label'].isin(NON_TAU).to_numpy()
        if not mask.all():
            self.indexes = self.indexes[mask]
            self.metadata = self.metadata.loc[mask].reset_index(drop=True)

        # Optional deterministic subset by exact (run, chunk, row) keys (e.g. Tune screen)
        if self.config.get('key_filter_path'):
            kf = pd.read_parquet(self.config['key_filter_path'], columns=KEY)
            pos = pd.MultiIndex.from_frame(self.metadata[KEY]).get_indexer(
                pd.MultiIndex.from_frame(kf[KEY])
            )
            if (pos < 0).any():
                raise ValueError('requested key_filter rows absent from V3EvalDataset')
            self.indexes = self.indexes[pos]
            self.metadata = self.metadata.iloc[pos].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.indexes)

    def _raw_observation(self, index: int) -> tuple[dict, int]:
        event = self.native[int(self.indexes[index])]
        obs = event['observations']
        drop_mod = self.config.get('drop_modality')
        if drop_mod:
            obs = dict(obs)
            key = {'tracker': 'tracker_hypothesis'}.get(drop_mod, drop_mod)
            obs[key] = {k: v[:0] for k, v in obs[key].items()}
        return obs, int(event['targets']['label'])

    def __getitem__(self, index: int) -> dict:
        obs, label = self._raw_observation(index)
        dom = compute_observable_containment(obs, self.geometry, self.bounds)
        if self.item_transform is not None:
            return self.item_transform(index, obs, label, dom, self.geometry, self.bounds, self.config)
        return {
            'index': int(index),
            'label': label,
            'c_obs': dom['c_obs'],
            'obs_domain_id': dom['obs_domain_id'],
            'cal_start': dom['cal_start'],
            'observation': obs,
        }


class V3TrainDataset(V3EvalDataset):
    """Training dataset coupling pure observations with optional offline truth sidecars."""

    def __init__(
        self,
        obs_root: Path | str,
        sidecar_root: Path | str | None,
        split: str = 'train',
        config: dict | None = None,
        cache_dir: Path | str | None = None,
        item_transform=None,
    ) -> None:
        if split != 'train':
            raise ValueError('V3TrainDataset is restricted exclusively to the train split')
        super().__init__(obs_root=obs_root, split=split, config=config, cache_dir=cache_dir, item_transform=item_transform)
        self.sidecar_root = Path(sidecar_root) if sidecar_root is not None else None
        self.event_sidecar: pd.DataFrame | None = None
        self.sample_weights: np.ndarray | None = None
        self.cluster_memmap: dict | None = None

        if self.sidecar_root is not None and (self.sidecar_root / 'event_sidecar.parquet').exists():
            sc_df = pd.read_parquet(self.sidecar_root / 'event_sidecar.parquet')
            pos = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(
                pd.MultiIndex.from_frame(self.metadata[KEY])
            )
            if (pos < 0).any():
                raise ValueError('train rows missing from truth_sidecar event_sidecar.parquet')
            self.event_sidecar = sc_df.iloc[pos].reset_index(drop=True)

        if self.config.get('sample_weight_path'):
            w_df = pd.read_parquet(self.config['sample_weight_path'], columns=KEY + ['energy_match_weight'])
            w_pos = pd.MultiIndex.from_frame(w_df[KEY]).get_indexer(
                pd.MultiIndex.from_frame(self.metadata[KEY])
            )
            if (w_pos < 0).any():
                raise ValueError('train rows missing from sample_weight_path')
            self.sample_weights = w_df['energy_match_weight'].to_numpy(dtype=np.float32)[w_pos]

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        if self.event_sidecar is not None:
            row = self.event_sidecar.iloc[index]
            item['label_7'] = int(row['label_7'])
            item['tau_branch_valid'] = int(row['tau_branch_valid'])
            item['charm_flag'] = int(row['charm_flag'])
            item['charm_valid'] = int(row['charm_valid'])
            item['vertex_xyz_mm'] = np.array(
                [row['vertex_x_mm'], row['vertex_y_mm'], row['vertex_z_mm']], dtype=np.float32
            )
            item['vertex_valid'] = int(row['vertex_valid'])
        if self.sample_weights is not None:
            item['sample_weight'] = float(self.sample_weights[index])
        return item
