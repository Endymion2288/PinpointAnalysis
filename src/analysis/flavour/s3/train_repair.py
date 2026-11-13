"""Train S3 revision 2 only after its real-train internal-health pilot passes."""
import argparse
import json
from pathlib import Path

import torch

from analysis.flavour.s2.train import run
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import build_model
from analysis.flavour.s3.pilot_repair import optimizer_for


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
    if args.device.startswith('cuda'):
        actual = torch.cuda.get_device_name()
        expected = campaign['hardware']['GPU']
        assert actual == expected or (expected == 'NVIDIA H100 NVL' and actual.startswith('NVIDIA H100 NVL MIG '))
        assert torch.__version__ == campaign['hardware']['torch']
        assert torch.cuda.is_bf16_supported(including_emulation=False)
    run(args, dataset_class=PlaneDataset, model_factory=build_model, collate_fn=collate_planes,
        health_factory=TrainingHealth, optimizer_factory=optimizer_for,
        manifest_extra={'stage': 'S3_revision2', 'architecture_revision': 2,
                        'pilot_gate': str(gate_path), 'pilot_gate_passed': True,
                        'internal_health_required': True,
                        'normalization': 'fixed_physical_input_scales_PeriLN_LayerScale_QK_LN_independent_fusion_branches_FP32_head',
                        'fusion_bottleneck': 'adaptive_regions_plus_all_local_readout' if config.get('readout_skip') else 'normalized_fixed_query_control',
                        'module_coordinate_relations': 'pixel_bar_z_tracker_station_metadata_not_physical_flight_distance',
                        'schema_whitelist': 'current_observation_18_features_Fourier_hardware_ID_counts_no_truth'})


if __name__ == '__main__':
    main()
