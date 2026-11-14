"""Export representation/model facts from live objects and accepted geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.representation import (
    MAX_CHANNELS_PER_BLOCK, NODE_FEATURES, SPATIAL_WINDOW_MM,
    TRACKER_STATION_PROXY_MM, XYZ_SCALES_MM,
)
from analysis.flavour.s3.dataset import DEFAULT_WINDOW_PIXELS, TRACKER_WINDOW_MM


OBSERVATION_INPUTS = {
    'pixel': ['channel_id', 'xyz_mm'],
    'bar': ['bounds_mm', 'horizontal', 'edep_proxy_MeV', 'pe_proxy', 'panel', 'layer'],
    'tracker_hypothesis': ['station', 'xy_mm', 'sigma_xy_mm'],
}


def representation_metadata(model, geometry, config, *, plane_windows, observation_schema=None):
    encoder = getattr(model, 'encoder', model)
    pitch = [float(np.median(np.diff(geometry[key]))) for key in ['pixel_Xpos', 'pixel_Ypos']]
    window_pixels = int(config.get('window_pixels', DEFAULT_WINDOW_PIXELS)) if plane_windows else 0
    width = [window_pixels * p for p in pitch] if window_pixels else [SPATIAL_WINDOW_MM]*2
    context = getattr(encoder, 'context', [])
    multiscale = getattr(encoder, 'multiscale', False)
    global_blocks = getattr(encoder, 'global_blocks', [])
    generic_cross_plane = bool(getattr(encoder, 'cross_plane', False) and len(global_blocks))
    generic_geometry = bool(generic_cross_plane and
        (getattr(encoder, 'architecture', None) == 'particlenet'
         or any(getattr(layer, 'pair', None) is not None for layer in global_blocks)))
    latents = getattr(encoder, 'latents', None)
    queries = getattr(encoder, 'module_queries', None)
    # A latent cross-attention path mixes planes but is distinct from a
    # geometry-aware relation layer. Preserve both facts explicitly.
    result = {
        'metadata_revision': 2,
        'transform': 'analysis.flavour.s3.dataset.plane_tokens' if plane_windows else 'analysis.flavour.s2.representation.token_inputs',
        'model_class': type(model).__module__ + '.' + type(model).__name__,
        'encoder_class': type(encoder).__module__ + '.' + type(encoder).__name__,
        'parameters': sum(p.numel() for p in model.parameters()),
        'encoder_parameters': sum(p.numel() for p in encoder.parameters()),
        'node_features': list(NODE_FEATURES), 'node_feature_count': len(NODE_FEATURES),
        'xyz_divisors_mm': list(XYZ_SCALES_MM),
        'pixel_pitch_mm': pitch, 'pixel_rebin_pitch_mm': float(config.get('pitch_mm', 0.)),
        'pixel_window_pixels': window_pixels if plane_windows else None,
        'pixel_window_mm': width, 'window_origin': 'absolute xy origin',
        'max_channels_per_block': MAX_CHANNELS_PER_BLOCK,
        'block_rule': 'module/type + same-plane window then canonical chunks' if plane_windows else 'same-plane 4mm window then canonical chunks',
        'grouping_by_modality': ({
            'pixel': 'hardware module and pixel plane, floor(absolute xy / pixel_window_mm)',
            'bar': 'stored hardware layer and bar z',
            'tracker': 'station and floor(absolute xy / tracker_window_mm)',
        } if plane_windows else {
            'pixel': 'pixel z, measured view and floor(absolute xy / 4 mm)',
            'bar': 'stored panel', 'tracker': 'station and floor(absolute xy / 16 mm)'}),
        'within_group_order': 'canonical xy order; split into at most 32 channels',
        'local_residual_windows_mm': {'pixel': width, 'bar': [SPATIAL_WINDOW_MM]*2,
                                      'tracker': [SPATIAL_WINDOW_MM]*2},
        'event_channel_truncation': False,
        'region_rule': 'canonical module/type block order in fixed-size groups' if multiscale else None,
        'blocks_per_region': int(config.get('region_blocks', 32)) if multiscale else None,
        'region_geometry': 'mean scaled z, detector one-hot and response; xy geometry columns are zero' if multiscale else None,
        'queries_per_region_or_module': int(queries.shape[0]) if queries is not None else None,
        'global_latents': int(latents.shape[0]) if latents is not None else None,
        'local_attention_or_edge_layers': len(getattr(encoder, 'local', [])),
        'explicit_geometry_context_layers': len(context),
        'generic_cross_plane_global_layers': len(global_blocks) if generic_cross_plane else 0,
        'global_latent_attention_layers': len(global_blocks) if latents is not None else 0,
        'explicit_cross_plane_geometry': bool(len(context) or generic_geometry),
        'direct_local_readout': bool(getattr(encoder, 'direct', False)),
        'empty_response_bypass': bool(getattr(encoder, 'empty_response_bypass', False)),
        'modalities': list(config.get('modalities', ['pixel', 'bar'])),
        'tracker_window_mm': TRACKER_WINDOW_MM if plane_windows else None,
        'tracker_z': {'definition': 'station index times proxy scale; not flight distance',
                      'station_proxy_mm': TRACKER_STATION_PROXY_MM},
        'used_observation_fields': {('tracker_hypothesis' if name == 'tracker' else name):
            OBSERVATION_INPUTS['tracker_hypothesis' if name == 'tracker' else name]
            for name in config.get('modalities', ['pixel', 'bar'])},
        'observations_schema': str(observation_schema) if observation_schema is not None else None,
        'target_or_provenance_features': [],
    }
    result['geometry_fields_used'] = ['pixel_Xpos', 'pixel_Ypos', 'pixel_Zpos'] + (['layer_is_pixel'] if plane_windows else [])
    fields = ['nodes', 'local_mask', 'event_mask', 'event_index']
    if queries is not None:
        prefix = 'region' if multiscale else 'module'
        fields += [prefix + suffix for suffix in ['_mask', '_group', '_position', '_id', '_detector', '_geometry', '_event']]
        if getattr(encoder, 'direct', False):
            fields.append('block_detector')
    else:
        fields.append('block_index')
    result['model_input_tensors'] = fields
    result['additional_model_inputs'] = ({'objects': {
        'features': 31, 'source': 'analysis.flavour.s5.objects.object_features',
        'definition': 'four log counts; seven feature 10/50/90% quantiles; vertex presence/xyz/log dca/log sigma_z',
        'standardization': 'selected Train mean/scale', 'truth_used': False}}
        if config.get('use_objects') else {})
    result['scope'] = 'raw observation encoder; additional wrapper inputs are listed separately'
    result['label_and_index_usage'] = 'batch label is supervised loss only; index is output alignment only; neither enters model forward'
    if observation_schema is not None:
        for group, fields in result['used_observation_fields'].items():
            available = {field.name for field in observation_schema.field(group).type}
            if not set(fields).issubset(available):
                raise ValueError(f'observation whitelist missing fields in {group}')
    return result


def main():
    from analysis.flavour.s3.models_repair import build_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--name', default='s4_empty_scratch_n100000_s42')
    args = parser.parse_args()
    partition = json.loads((args.root/'partitions.json').read_text())
    geometry = json.loads(Path(partition['config']['geometry']).read_text())
    config = next(row['config'] for row in json.loads(args.candidates.read_text())['models'] if row['name']==args.name)
    model = build_model({**config, 'compile_local': False})
    path = next((args.root/'profile_data/shards').glob('*/observations.parquet'))
    result = representation_metadata(model, geometry, config, plane_windows=True, observation_schema=pq.read_schema(path))
    result.update(reference_model=args.name, structural_instantiation='eager; no checkpoint, response inference or compilation')
    write_json(args.root/'representation_metadata.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
