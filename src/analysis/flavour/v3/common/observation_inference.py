"""Label-free observation-only inference entrypoint for frozen PINPOINT models (Stage 322-11).

Reads strictly `observations.parquet` + `geometry.json` (zero `targets.parquet`, zero `label`,
zero `truth_sidecar`), runs frozen FP32 checkpoints for any model family (`b1pp`, `pf_multitask`,
`pf_v4`), and returns calibrated/raw probabilities, domains, and reconstructed PV predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.flavour.v2.profile import device_batch
from analysis.flavour.v2.r1.tasks import probabilities
from analysis.flavour.v3.common.datasets import V3ObservationDataset
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import (
    collate_b1pp,
    extract_b1pp_item,
    static_plane_layout,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4


def infer_observation_only(
    obs_paths: Path | str | list[Path | str],
    geometry_json: Path | str,
    checkpoint_path: Path | str,
    config: dict,
    *,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Run deterministic FP32 inference on an observation-only source with zero labels."""
    dev = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    geom = json.loads(Path(geometry_json).read_text())
    cfg = dict(config)
    train_mod = str(cfg.get('train_module', ''))
    is_b1pp = ('v1_b1pp' in train_mod) or (cfg.get('model_family') == 'b1pp')

    ds = V3ObservationDataset(
        obs_paths=obs_paths,
        geometry_json=geometry_json,
        config=cfg,
        item_transform=extract_b1pp_item if is_b1pp else extract_pf_item,
    )
    state = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    sd = state.get('model', state)

    if is_b1pp:
        layout = static_plane_layout(geom)
        model: torch.nn.Module = ZAlignedStereoNet(layout, cfg).to(dev)
    elif cfg.get('model_family') == 'pf_v4':
        model = PinpointFormerV4(geom, cfg).to(dev)
    else:
        model = PinpointFormerMultitask(geom, cfg).to(dev)
    model.load_state_dict(sd)
    model.eval()

    rows_probs: list[np.ndarray] = []
    rows_logits: list[np.ndarray] = []
    rows_dom: list[np.ndarray] = []
    rows_idx: list[np.ndarray] = []
    rows_vhat: list[np.ndarray] = []
    rows_charm: list[np.ndarray] = []

    n = len(ds)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            items = [ds[i] for i in range(start, min(n, start + batch_size))]
            if is_b1pp:
                batch = collate_b1pp(items, mirror_rng=None)
                if 'label' in batch:
                    raise AssertionError('Observation-only batch must not contain label')
                gpu = device_batch(batch, dev)
                logits = model(gpu).float()
                probs = probabilities(logits, cfg.get('task', 'T4')).float()
                rows_logits.append(logits.cpu().numpy())
                rows_probs.append(probs.cpu().numpy())
            else:
                batch = collate_pf(items, mirror_rng=None)
                if 'label' in batch:
                    raise AssertionError('Observation-only batch must not contain label')
                gpu = move_batch_to_device(batch, dev)
                out = model(gpu, grad_blend=1.0, return_aux=False)
                rows_logits.append(out['logits'].float().cpu().numpy())
                rows_probs.append(out['probs4'].float().cpu().numpy())
                rows_vhat.append(out['v_hat'].float().cpu().numpy())
                rows_charm.append(out['charm_prob'].float().cpu().numpy())
            rows_dom.append(batch['obs_domain_id'].numpy())
            rows_idx.append(batch['index'].numpy())

    probs_np = np.concatenate(rows_probs, axis=0) if rows_probs else np.empty((0, 4), dtype=np.float32)
    dom_np = np.concatenate(rows_dom, axis=0) if rows_dom else np.empty(0, dtype=np.int64)
    idx_np = np.concatenate(rows_idx, axis=0) if rows_idx else np.empty(0, dtype=np.int64)
    out_dict: dict[str, np.ndarray] = {
        'index': idx_np,
        'obs_domain_id': dom_np,
        'p0': probs_np[:, 0],
        'p1': probs_np[:, 1],
        'p2': probs_np[:, 2],
        'p3': probs_np[:, 3],
    }
    if rows_vhat:
        vh = np.concatenate(rows_vhat, axis=0)
        out_dict['v_hat_x_mm'] = vh[:, 0]
        out_dict['v_hat_y_mm'] = vh[:, 1]
        out_dict['v_hat_z_mm'] = vh[:, 2]
    if rows_charm:
        out_dict['charm_prob'] = np.concatenate(rows_charm, axis=0)
    return pd.DataFrame(out_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    df = infer_observation_only(
        args.observations,
        args.geometry,
        args.checkpoint,
        cfg,
        batch_size=args.batch_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(json.dumps({'events': int(len(df)), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
