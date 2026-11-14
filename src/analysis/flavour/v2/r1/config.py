"""Build R1 recipes from the accepted revision-3 geometry and MAE control."""

import json
from pathlib import Path

from analysis.flavour.v2.r1.data import image_geometry


def recipe(r0, family, task='T4', seed=42, compile_local=False):
    reference = json.loads((Path(r0)/'configs/s6_reference.json').read_text())['models']
    name = 's4_empty_mae3_n100000_s42' if family == 'B4-MAE' else 's4_empty_scratch_n100000_s42'
    config = next(row['config'].copy() for row in reference if row['name'] == name)
    config.pop('name', None)
    if family != 'B4-MAE':
        config.pop('pretrained_checkpoint', None)
        config['initialization'] = 'scratch'
    config.update(family=family, task=task, seed=seed, train_domain='all',
        cache_shards=2, bucket_size=0, batch_size=8, workers=2, compile_local=compile_local,
        local_chunk_blocks=2048, weight_decay=.01, clip_grad_norm=1.,
        modalities=['pixel', 'bar', 'tracker'], precision='BF16_encoder_FP32_head')
    if family in ['B2', 'B3']:
        config.update(dim=64, dynamic=False, grouping_mm=4.)
        config['local_chunk_blocks'] = 512 if family == 'B2' else 1024
    if family == 'B2':
        config.update(compile_local=False,
            local_execution_policy='eager: compiled BF16 local backward produces nonfinite gradients on the fixed real profile batch; eager BF16 and compiled FP32 pass')
    if family == 'B1':
        frozen = json.loads((Path(r0)/'partitions.json').read_text())
        geometry = json.loads(Path(frozen['config']['geometry']).read_text())
        config.update(image=image_geometry(geometry), cnn_width=16)
    return config
