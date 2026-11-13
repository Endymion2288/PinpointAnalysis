"""Train S3 revision 2 only after its real-train internal-health pilot passes."""
import argparse
import json
from pathlib import Path

import torch
import pandas as pd

from analysis.flavour.s2.train import run
from analysis.flavour.s2.readback import compatible_gpu_name
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import build_model
from analysis.flavour.s3.pilot_repair import optimizer_for


def architecture_metadata(config):
    if config['architecture'] == 'part':
        return {'normalization': 'original_ParT_PreLN_body_original_head_same_repair_optimizer_schedule',
                'fusion_bottleneck': 'original_ParT_local_blocks_event_attention_mean_max',
                'local_blocks': 'same_plane_4mm_then_canonical_chunks32_no_event_truncation'}
    return {'normalization': 'fixed_physical_input_scales_PeriLN_LayerScale_QK_LN_independent_fusion_branches_FP32_head',
            'fusion_bottleneck': ('adaptive_regions_plus_all_local_readout' if config.get('readout_skip') else
                                  'adaptive_regions_without_direct_readout' if config.get('compression') == 'multiscale' else
                                  'normalized_fixed_query_control'),
            'local_blocks': 'same_module_16pixel_windows_then_canonical_chunks32_no_event_truncation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text()); config = campaign['experiments'][args.task]
    gate_path = args.root / 'pilot' / config['required_pilot'] / 'gate.json'
    gate = json.loads(gate_path.read_text())
    if not gate['gate_passed']:
        raise RuntimeError('real-train health/memorization pilot must pass before full fitting')
    for key in ['architecture','dim','compression','module_queries','latents','readout_skip',
                'compile_local','local_chunk_blocks','local_execution_version',
                'pitch_mm','modalities','region_blocks','window_pixels']:
        if gate['config'].get(key)!=config.get(key):
            raise RuntimeError(f'pilot execution/architecture mismatch for {key}')
    stress_dir = args.root / 'stress' / config['required_pilot']
    stress = json.loads((stress_dir / 'summary.json').read_text())
    stress_config = stress['model_configs'][0]
    for key in ['architecture','dim','compression','readout_skip','compile_local','local_chunk_blocks',
                'local_execution_version','pitch_mm','modalities','region_blocks','window_pixels']:
        if stress_config.get(key)!=config.get(key):
            raise RuntimeError(f'full-event stress execution/representation mismatch for {key}')
    measured = pd.read_parquet(stress_dir / 'measurements.parquet')
    active_backward = measured[measured['mode'].eq('backward') & measured.represented_nodes.gt(1)]
    if not measured.finite.all() or len(active_backward) < 1 or not active_backward.input_gradient_squared_norm.gt(0).all():
        raise RuntimeError('complete-event stress gradients/activations must pass before fitting')
    if args.device.startswith('cuda'):
        actual = torch.cuda.get_device_name()
        expected = campaign['hardware'].get('allowed_GPUs',campaign['hardware']['GPU'])
        assert compatible_gpu_name(actual,expected)
        assert torch.__version__ == campaign['hardware']['torch']
        assert torch.cuda.is_bf16_supported(including_emulation=False)
    run(args, dataset_class=PlaneDataset, model_factory=build_model, collate_fn=collate_planes,
        health_factory=TrainingHealth, optimizer_factory=optimizer_for,
        manifest_extra={'stage': 'S3_revision2', 'architecture_revision': 2,
                        'pilot_gate': str(gate_path), 'pilot_gate_passed': True,
                        'full_event_stress_gate': str(stress_dir/'summary.json'),
                        'internal_health_required': True,
                        'local_encoder_execution': 'torch.compile_dynamic' if config.get('compile_local') else 'eager',
                        'local_chunk_blocks': config.get('local_chunk_blocks',256),
                        'health_forward_execution': 'identical_training_encoder_hooks_outside_compiled_features',
                        **architecture_metadata(config),
                        'module_coordinate_relations': 'pixel_bar_z_tracker_station_metadata_not_physical_flight_distance',
                        'schema_whitelist': 'current_observation_18_features_Fourier_hardware_ID_counts_no_truth'})


if __name__ == '__main__':
    main()
