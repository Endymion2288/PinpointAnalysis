"""Read back the R0 delivery without opening sealed confirmation rows or labels."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import FAMILY, KEY, equal_identity_rows, family_index


def load(root, name):
    return json.loads((root/name).read_text())


def readback(root):
    started = time.perf_counter()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    plan = load(root, 'partitions.json')
    checks = {}
    ledger = pd.read_parquet(root/'manifests/usage_ledger.parquet')
    used = family_index(ledger)
    partitions, counts = {}, {}
    for name, entry in plan['manifests'].items():
        # Confirm only exposes family keys. Do not access row_label_access here.
        frame = pd.read_parquet(entry['manifest'])
        families = family_index(frame)
        partitions[name] = families
        checks[name+'_events'] = len(frame) == entry['events']
        checks[name+'_family_unique'] = families.is_unique
        checks[name+'_historical_disjoint'] = not families.isin(used).any()
        if name == 'confirm':
            checks['confirm_public_identity_only'] = set(frame) == {'confirm_index', *FAMILY}
            counts[name] = {'events': len(frame), 'class_counts': 'sealed; R0 construction counts only'}
        else:
            actual = {str(key): int(value) for key, value in frame.groupby('label').size().items()}
            checks[name+'_class_counts'] = actual == plan['counts'][name]
            counts[name] = {'events': len(frame), 'class_counts': actual}
    independent = ['train_1000000', 'tune', 'cal_a', 'cut_b', 'confirm']
    for i, name in enumerate(independent):
        for other in independent[i+1:]:
            checks[f'{name}_disjoint_{other}'] = not partitions[name].isin(partitions[other]).any()
    for small, large in [('train_100000', 'train_300000'), ('train_300000', 'train_1000000')]:
        checks[small+'_nested_in_'+large] = partitions[small].isin(partitions[large]).all()
    source = pd.read_parquet(root/'manifests/usage_sources.parquet')
    checks['usage_source_membership_complete'] = (len(source) == len(ledger) and
        np.array_equal(source.ledger_row, np.arange(len(ledger))) and source.source_mask.ne(0).all())
    closure = load(root, 'usage_closure.json')
    checks['history_closure'] = closure['status'] == 'complete' and all(closure['checks'].values())
    checks['external_generator_vertex_pool_excluded'] = closure['checks']['external_vertex_donors_excluded']
    checks['external_generator_run_namespace_disjoint'] = closure['external_generator']['accepted_run_namespace_disjoint']
    checks['confirm_sealed_permissions'] = ((root/'sealed').stat().st_mode & 0o777 == 0o700 and
        (root/'sealed/confirm_rows.parquet').stat().st_mode & 0o777 == 0o600)
    checks['original_33_partition_gates'] = len(plan['checks']) == 33 and all(plan['checks'].values())
    metadata = load(root, 'representation_metadata.json')
    checks['actual_representation_metadata'] = (metadata['parameters'] == 897992 and
        metadata['pixel_window_pixels'] == 16 and np.allclose(metadata['pixel_window_mm'], [.3328, .3648]) and
        metadata['explicit_geometry_context_layers'] == 0 and not metadata['explicit_cross_plane_geometry'] and
        metadata['global_latents'] == 64 and metadata['queries_per_region_or_module'] == 2 and
        metadata['global_latent_attention_layers'] == 2 and len(metadata['node_features']) == 18)
    checks['input_whitelist'] = (metadata['target_or_provenance_features'] == [] and
        set(metadata['used_observation_fields']) == {'pixel', 'bar', 'tracker_hypothesis'} and
        not {'label', 'index'}.intersection(metadata['model_input_tensors']))
    prepare = load(root, 'preparation.json')
    checks['prepared_train_inputs'] = (prepare['status'] == 'complete' and prepare['profile_events'] == 10000 and
        prepare['micro_events'] == 256 and prepare['exact_event_family_readback'] and prepare['bar_PE_conserved'])
    checks['converted_keys_and_uint64_family_exact'] = equal_identity_rows(
        pd.read_parquet(root/'profile_data/catalog.parquet', columns=KEY+FAMILY),
        pd.read_parquet(root/'profile_data/manifest.parquet', columns=KEY+FAMILY))
    for prefix, folder, file in [('stream', 'profile_data', 'catalog.parquet'), ('micro', 'micro_data', 'catalog.parquet')]:
        catalog = pd.read_parquet(root/folder/file)
        checks[prefix+'_is_new_Train'] = family_index(catalog).isin(partitions['train_100000']).all()
        checks[prefix+'_balanced'] = set(catalog.label.unique()) == set(range(4)) and catalog.groupby('label').size().nunique() == 1
    fit = load(root, 'fit/fit32_a2.json')
    checks['real_32_learning_and_health'] = (fit['status'] == 'complete' and all(fit['checks'].values()) and
                                            fit['loaded_event_keys_and_labels_match_selection'])
    fit_events = pd.read_parquet(root/'fit/fit32_a2_events.parquet')
    micro = pd.read_parquet(root/'micro_data/catalog.parquet')
    checks['fit_selection_keys_and_classes'] = (len(fit_events) == 32 and
        fit_events.groupby('label').size().to_dict() == {i: 8 for i in range(4)} and
        pd.MultiIndex.from_frame(fit_events[KEY]).isin(pd.MultiIndex.from_frame(micro[KEY])).all())
    final_cache = load(root, 'profiles/final_cache_readback.json')
    checks['final_cache_format_keys_and_tensors'] = (final_cache['status'] == 'complete' and
        final_cache['cache_key']['cache_format_revision'] == 2 and final_cache['event_keys_checked'] == 256 and
        final_cache['equivalence']['all_collated_tensors_bitwise_equal'] and final_cache['temporary_scratch_removed'])
    profiles = {}
    for mode, file_prefix, events, minimum_cases in [('micro', 'micro_a2', 256, 7),
            ('stream', 'stream_a1', 10000, 7), ('stream_compiled', 'stream_compiled_a1', 10000, 3)]:
        profile = load(root, f'profiles/{file_prefix}.json')
        checks[mode+'_profile_complete'] = profile['status'] == 'complete' and len(profile['cases']) >= minimum_cases
        checks[mode+'_profile_equivalence'] = (profile['equivalence']['events'] == events and
            profile['equivalence']['all_collated_tensors_bitwise_equal'] and
            profile['packed_cache_build']['channel_coverage_and_response_masks_equal'])
        # GPU index_add reductions can differ by floating-point rounding even
        # for bitwise-identical inputs. Require a numerical output readback.
        checks[mode+'_GPU_logits_readback'] = (profile['equivalence']['GPU_logits_compared_events'] == 2 and
            profile['equivalence']['GPU_logits_max_absolute_difference'] <= 3e-6)
        checks[mode+'_profile_event_coverage'] = all(row['events'] == events and row['exact_event_coverage']
                                                   for case in profile['cases'] for row in case['passes'])
        checks[mode+'_profile_cache_cleanup'] = profile.get('temporary_scratch_removed', False)
        if mode == 'stream_compiled':
            checks['compiled_encoder_gradients_positive'] = all(
                case['warmup_gradient_squared_norms'].get('input.0.weight', 0.) > 0 and
                case['warmup_gradient_squared_norms'].get('latent_read.q.weight', 0.) > 0
                for case in profile['cases'] if case['training'])
        table = pd.read_parquet(root/f'profiles/{file_prefix}_batches.parquet')
        scenarios = []
        for case in profile['cases']:
            cold, warm = case['passes'][0], case['passes'][-1]
            scenarios.append({'scenario': case['scenario'], 'workers': case['workers'], 'batch_size': case['batch_size'],
                'training': case['training'], 'cold_seconds': cold['seconds'], 'warm_seconds': warm['seconds'],
                'warm_events_per_second': warm['events_per_second'],
                # The first profiler revision logged iterator startup separately.
                # Include it explicitly in the end-to-end data-wait fraction.
                'cold_data_wait_fraction_including_startup': (cold['iterator_startup_seconds']+cold['seconds_by_stage']['wait'])/cold['seconds'],
                'warm_data_wait_fraction_including_startup': (warm['iterator_startup_seconds']+warm['seconds_by_stage']['wait'])/warm['seconds'],
                'steady_data_wait_fraction': warm['steady_data_wait_fraction'],
                'warm_stage_seconds': warm['seconds_by_stage'], 'warm_worker_seconds': warm['worker_seconds'],
                'warm_collate_worker_seconds': warm['collate_worker_seconds'],
                'channel_padding_utilization': warm['channel_padding_utilization'],
                'event_block_padding_utilization': warm['event_block_padding_utilization'],
                'peak_memory': case['peak_memory']})
            if 'warmup_and_compile_seconds' in case:
                scenarios[-1]['warmup_and_compile_seconds'] = case['warmup_and_compile_seconds']
        subset = table[table['pass'].eq(table['pass'].max()) & ~table.scenario.str.startswith('validation')].copy()
        subset['channels_per_event'] = subset.represented_channels/subset.events
        subset['compute_ms_per_event'] = subset.compute_seconds/subset.events*1000
        latency = []
        for scenario, group in subset.groupby('scenario', sort=True):
            group['occupancy_quartile'] = pd.qcut(group.channels_per_event.rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            for band, block in group.groupby('occupancy_quartile', observed=True):
                latency.append({'scenario': scenario, 'batch_occupancy_quartile': str(band),
                                'events': int(block.events.sum()), 'mean_channels_per_event': float(block.represented_channels.sum()/block.events.sum()),
                                'compute_ms_per_event': float(block.compute_seconds.sum()/block.events.sum()*1000)})
        profiles[mode] = {'GPU': profile['GPU'], 'precision': profile['precision'], 'cases': scenarios,
                          'batch_occupancy_latency': latency, 'cache_build': profile['packed_cache_build'],
                          'staging': profile['staging'], 'campaign_seconds': profile['seconds']}
    checks = {key: bool(value) for key, value in checks.items()}
    result = {'status': 'complete' if all(checks.values()) else 'failed', 'stage': 'R0',
              'UTC': datetime.now(timezone.utc).isoformat(), 'checks': checks, 'passed': sum(checks.values()),
              'partition_counts': counts, 'historically_used_families': len(ledger),
              'Confirm_sealed_rows_labels_or_responses_read': False,
              'profiles': profiles, 'fit': {key: fit[key] for key in ['final', 'steps', 'checks', 'seconds']},
              'limits': ['training/inference profiles use Train events and lr=0; no new validation quality measured',
                         'eager and compiled measurements are grouped by actual allocated GPU; no cross-device causal speedup claim',
                         'compiled native versus packed_bucket changes both cache and sampler; cases share the compiler disk cache',
                         'cold pass includes worker startup after allocation-local staging; it is not a fresh EOS cache measurement',
                         '10k events selected for dense source-chunk reads; not the full inventory occupancy distribution',
                         'two passes per scenario; no independent runtime confidence interval',
                         'RSS aggregate can double count shared libraries/mappings'],
              'seconds': time.perf_counter()-started,
              'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    write_json(root/'acceptance.json', result)
    if not all(checks.values()):
        raise RuntimeError('R0 delivery failed: '+str([key for key, value in checks.items() if not value]))
    print(json.dumps({'status': result['status'], 'passed': result['passed']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    readback(parser.parse_args().root)


if __name__ == '__main__':
    main()
