"""S3 fitting through the established atomic, restartable S2 training loop."""
import argparse
import json
from pathlib import Path

import torch

from analysis.flavour.s2.train import run
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.models import PlaneModuleClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--task', required=True, type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    config = campaign['experiments'][args.task]
    if args.device.startswith('cuda'):
        assert torch.cuda.get_device_name() == campaign['hardware']['GPU']
        assert torch.__version__ == campaign['hardware']['torch']
        assert torch.cuda.is_bf16_supported(including_emulation=False)
    run(args, dataset_class=PlaneDataset, model_factory=PlaneModuleClassifier,
        collate_fn=collate_planes,
        manifest_extra={'stage': 'S3', 'local_blocks': 'same_plane_pitch_multiple_windows_canonical_chunks32_complete_coverage',
                        'module_layout': 'pixel_ordered_true_layer_flags_bar_stored_layer_tracker_station',
                        'fusion_bottleneck': f"{config.get('module_queries', 4)}_learned_queries_per_physical_module_and_detector_then_declared_fusion",
                        'module_coordinate_relations': 'known_z_module_and_detector_only_no_invented_bar_point',
                        'schema_whitelist': '18_observation_features_local_Fourier_xy_hardware_module_ID_no_truth'})


if __name__ == '__main__':
    main()
