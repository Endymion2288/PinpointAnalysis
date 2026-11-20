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


class V3ObservationDataset(Dataset):
    """Strictly observation-only dataset for label-free deployment & mock-readout replay (Stage 322-11).

    Reads ONLY `observations.parquet` + `geometry.json`. Never opens `targets.parquet`,
    never reads any `label` column, and never touches truth sidecars.
    """

    def __init__(
        self,
        obs_paths: Path | str | list[Path | str],
        geometry_json: Path | str,
        config: dict | None = None,
        item_transform=None,
    ) -> None:
        self.config = dict(config or {})
        self.item_transform = item_transform
        self.geometry = json.loads(Path(geometry_json).read_text())
        self.bounds = detector_bounds(self.geometry)

        if isinstance(obs_paths, (str, Path)):
            p = Path(obs_paths)
            if p.is_file():
                files = [p]
            elif (p / 'observations.parquet').is_file():
                files = [p / 'observations.parquet']
            elif (p / 'shards').is_dir():
                files = sorted(
                    d / 'observations.parquet'
                    for d in (p / 'shards').iterdir()
                    if (d / 'observations.parquet').is_file()
                )
            else:
                files = sorted(p.rglob('observations.parquet'))
        else:
            files = [Path(x) for x in obs_paths]
        if not files:
            raise FileNotFoundError(f'No observations.parquet found in {obs_paths}')

        self.files = files
        self.lengths = [int(pq.ParquetFile(f).metadata.num_rows) for f in self.files]
        self.boundaries = np.cumsum([0] + self.lengths)
        self._cache_idx: int | None = None
        self._cache_rows: list[dict] | None = None

    def __len__(self) -> int:
        return int(self.boundaries[-1])

    def _get_obs(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = int(np.searchsorted(self.boundaries[1:], index, side='right'))
        local_idx = int(index - self.boundaries[shard_idx])
        if self._cache_idx != shard_idx:
            self._cache_rows = pq.read_table(self.files[shard_idx]).to_pylist()
            self._cache_idx = shard_idx
        assert self._cache_rows is not None
        return self._cache_rows[local_idx]

    def __getitem__(self, index: int) -> dict:
        obs = self._get_obs(index)
        dom = compute_observable_containment(obs, self.geometry, self.bounds, config=self.config)
        if self.item_transform is not None:
            return self.item_transform(index, obs, None, dom, self.geometry, self.bounds, self.config)
        return {
            'index': int(index),
            'c_obs': dom['c_obs'],
            'obs_domain_id': dom['obs_domain_id'],
            'cal_start': dom['cal_start'],
            'observation': obs,
        }


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

        all_paths = sorted(
            p for p in (self.root / 'shards').iterdir()
            if (p / 'observations.parquet').exists()
        )
        active_paths: list[Path] = []
        active_idx_lists: list[np.ndarray] = []
        meta_frames: list[pd.DataFrame] = []
        for path in all_paths:
            frame = pd.read_parquet(path / 'provenance.parquet', columns=list(EVAL_PROVENANCE_COLUMNS))
            keep = frame['split'].eq(split)
            idx = np.flatnonzero(keep.to_numpy())
            if len(idx):
                active_paths.append(path)
                active_idx_lists.append(idx)
                meta_frames.append(frame.iloc[idx])

        self.paths = active_paths if active_paths else all_paths[:1]
        staged = os.environ.get('PINPOINT_V3_SHARED_STAGE') or os.environ.get('PINPOINT_NATIVE_SHARD_ROOT')
        native_paths = [Path(staged) / p.name for p in self.paths] if staged else self.paths
        self.native = CachedNativeDataset(
            native_paths,
            cache_dir=cache_dir,
            max_shards=int(self.config.get('cache_shards', 2)),
        )

        indexes: list[int] = []
        for i, idx in enumerate(active_idx_lists):
            indexes.extend((self.native.boundaries[i] + idx).tolist())

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
        dom = compute_observable_containment(obs, self.geometry, self.bounds, config=self.config)
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
        self.cluster_sidecar_dir: Path | None = None
        self.cluster_memmap: dict | None = None

        if self.sidecar_root is not None and (self.sidecar_root / 'event_sidecar.parquet').exists():
            sc_df = pd.read_parquet(self.sidecar_root / 'event_sidecar.parquet')
            pos = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(
                pd.MultiIndex.from_frame(self.metadata[KEY])
            )
            if (pos < 0).any():
                raise ValueError('train rows missing from truth_sidecar event_sidecar.parquet')
            self.event_sidecar = sc_df.iloc[pos].reset_index(drop=True)

        if self.config.get('load_cluster_sidecar', False):
            local_sc = os.environ.get('PINPOINT_V3_LOCAL_CLUSTER_SIDECAR')
            if local_sc and Path(local_sc).exists():
                self.cluster_sidecar_dir = Path(local_sc)
            elif self.sidecar_root is not None and (self.sidecar_root / 'cluster_sidecar').exists():
                self.cluster_sidecar_dir = self.sidecar_root / 'cluster_sidecar'

        if self.config.get('sample_weight_path'):
            w_df = pd.read_parquet(self.config['sample_weight_path'], columns=KEY + ['energy_match_weight'])
            w_pos = pd.MultiIndex.from_frame(w_df[KEY]).get_indexer(
                pd.MultiIndex.from_frame(self.metadata[KEY])
            )
            if (w_pos < 0).any():
                raise ValueError('train rows missing from sample_weight_path')
            self.sample_weights = w_df['energy_match_weight'].to_numpy(dtype=np.float32)[w_pos]

    def _get_cluster_memmaps(self) -> dict | None:
        if self.cluster_sidecar_dir is None:
            return None
        if self.cluster_memmap is None:
            pix_off = np.memmap(self.cluster_sidecar_dir / 'pix_offsets.bin', dtype='<i8', mode='r')
            bar_off = np.memmap(self.cluster_sidecar_dir / 'bar_offsets.bin', dtype='<i8', mode='r')
            n_pix = int(pix_off[-1])
            n_bar = int(bar_off[-1])
            self.cluster_memmap = {
                'pix_off': pix_off,
                'bar_off': bar_off,
                'first_cid': np.memmap(
                    self.cluster_sidecar_dir / 'cluster_first_channel_id.bin', dtype='<u8', mode='r', shape=(n_pix,)
                ),
                'origin': np.memmap(
                    self.cluster_sidecar_dir / 'cluster_origin.bin', dtype='|i1', mode='r', shape=(n_pix,)
                ),
                'ancestry': np.memmap(
                    self.cluster_sidecar_dir / 'cluster_ancestry_id.bin', dtype='<i4', mode='r', shape=(n_pix, 2)
                ),
                'pix_rel': np.memmap(
                    self.cluster_sidecar_dir / 'pixel_rel_target.bin', dtype='<f2', mode='r', shape=(n_pix, 3)
                ),
                'bar_rel': np.memmap(
                    self.cluster_sidecar_dir / 'bar_rel_target.bin', dtype='<f2', mode='r', shape=(n_bar, 3)
                ),
            }
        return self.cluster_memmap

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
            item['vis_kinematics'] = np.array(
                [row['vis_pz_log1p'], row['vis_pt_log1p'], row['vis_cos_dphi'], row['vis_sin_dphi']],
                dtype=np.float32,
            )
            item['kin_valid'] = int(row['kin_valid'])
            mm = self._get_cluster_memmaps()
            if mm is not None:
                si = int(row['sidecar_index'])
                ps, pe = int(mm['pix_off'][si]), int(mm['pix_off'][si + 1])
                bs, be = int(mm['bar_off'][si]), int(mm['bar_off'][si + 1])
                item['sc_first_cid'] = np.array(mm['first_cid'][ps:pe], copy=True)
                item['sc_origin'] = np.array(mm['origin'][ps:pe], copy=True)
                item['sc_ancestry'] = np.array(mm['ancestry'][ps:pe], copy=True)
                item['sc_pix_rel'] = np.array(mm['pix_rel'][ps:pe], copy=True)
                item['sc_bar_rel'] = np.array(mm['bar_rel'][bs:be], copy=True)
        if self.sample_weights is not None:
            item['sample_weight'] = float(self.sample_weights[index])
        return item
