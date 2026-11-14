"""Attach source membership and prior-consumer closure to the frozen R0 ledger."""

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

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import FAMILY, KEY, SPLIT_ROLE, discover_sources, family_index, row_keys


def run(args):
    started = time.perf_counter()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    plan = json.loads((args.root/'partitions.json').read_text())
    base = Path(plan['config']['historical_root'])
    ledger = pd.read_parquet(args.root/'manifests/usage_ledger.parquet')
    used = family_index(ledger)
    inventory = pd.read_parquet(plan['config']['inventory'], columns=KEY+FAMILY+['split'])
    accepted = pd.Index(row_keys(inventory))
    sources = discover_sources(base)
    if len(sources)+2 > 64:
        raise ValueError('source mask needs more than uint64')
    membership = np.zeros(len(ledger), dtype='uint64')
    source_records = []
    for source_id, path in enumerate(sources):
        columns = pq.read_schema(path).names
        table = pd.read_parquet(path, columns=KEY+[column for column in FAMILY+['split'] if column in columns])
        positions = accepted.get_indexer(row_keys(table))
        if (positions < 0).any():
            raise ValueError('accepted-source table has keys outside the accepted inventory: '+str(path))
        families = inventory.iloc[positions][FAMILY]
        if set(FAMILY).issubset(table):
            for column in FAMILY:
                if not np.array_equal(families[column].to_numpy(), table[column].to_numpy()):
                    raise ValueError('historical family identity differs from accepted inventory')
        location = used.get_indexer(family_index(families))
        if (location < 0).any():
            raise ValueError('frozen usage ledger missed historically accessed families: '+str(path))
        roles = (table['split'].map(SPLIT_ROLE).fillna(32).to_numpy(dtype='uint8')
                 if 'split' in table else np.full(len(table), 16, dtype='uint8'))
        if np.any(ledger.usage_mask.to_numpy()[location] & roles != roles):
            raise ValueError('frozen ledger did not include historical usage roles')
        np.bitwise_or.at(membership, location, np.uint64(1) << np.uint64(source_id))
        source_records.append({'source_id': source_id, 'path': str(path), 'events': len(table),
                               'families': len(np.unique(location)), 'all_in_frozen_ledger': True})
    # The frozen R0 policy also excludes unusable/unclassified rows from the
    # entire old calibration/test. S6 usable-only catalogs omit those rows.
    for split in ['calibration', 'test']:
        families = inventory.loc[inventory.split.eq(split), FAMILY]
        location = used.get_indexer(family_index(families))
        if (location < 0).any():
            raise ValueError('old public/calibrated partition not fully excluded')
        source_id = len(source_records)
        np.bitwise_or.at(membership, location, np.uint64(1) << np.uint64(source_id))
        source_records.append({'source_id': source_id, 'path': plan['config']['inventory'],
                               'filter': {'split': split}, 'events': len(families),
                               'families': len(np.unique(location)), 'all_in_frozen_ledger': True,
                               'policy': 'exclude entire historical partition, including unusable/unclassified rows'})
    if (membership == 0).any():
        raise ValueError('used families lack a traceable source table')
    pd.DataFrame({'ledger_row': np.arange(len(ledger), dtype='int32'), 'source_mask': membership}).to_parquet(
        args.root/'manifests/usage_sources.parquet', index=False, compression=COMPRESSION)
    aliases = []
    for stage, expected in [('s3_2026-11-12', 's2_2026-11-11/h100/shards'),
                            ('s3_repair_2026-11-13', 's2_2026-11-11/h100/shards'),
                            ('s5_2026-11-14', 's4_2026-11-13/shards')]:
        path = base/stage/'shards'
        if path.resolve() != (base/expected).resolve():
            raise ValueError('historical corpus alias changed: '+str(path))
        aliases.append({'stage': stage, 'shards': str(path), 'resolved': str(path.resolve())})
    generator_path = base/'s6_2026-11-14/generator/manifest.parquet'
    generator = pd.read_parquet(generator_path, columns=['tune', 'source_path', 'source_row']+KEY)
    if generator.duplicated(['tune', 'source_path', 'source_row']).any():
        raise ValueError('external generator source identities are duplicated')
    generator['usage'] = 'published_external_generator_test'
    generator.to_parquet(args.root/'manifests/external_generator_usage.parquet', index=False, compression=COMPRESSION)
    public = pq.read_schema(args.root/'manifests/confirm.parquet').names
    if set(public) != {'confirm_index', *FAMILY}:
        raise ValueError('public confirmation identities expose rows or labels')
    cal_families = inventory.loc[inventory.split.eq('calibration'), FAMILY]
    if not family_index(cal_families).isin(used).all():
        raise ValueError('generator benchmark vertex-donor pool was not excluded')
    record = {'status': 'complete', 'UTC': datetime.now(timezone.utc).isoformat(),
              'accepted_source_tables': source_records, 'usage_families': len(ledger),
              'source_membership': {'file': 'manifests/usage_sources.parquet',
                                    'join': 'ledger_row is the positional row of frozen usage_ledger.parquet',
                                    'bits': 'bit source_id marks membership in that accepted source table'},
              'corpus_aliases': aliases,
              'consumer_closure': [
                  {'consumer': 'S0/S1 manual displays, loader and B0 smoke', 'corpus': 'all S0 shards and smoke provenance'},
                  {'consumer': 'S2 architecture/hardware/domain/error studies; S3 repair, fit, probes', 'corpus': 'complete S2 manifest, catalog and smoke'},
                  {'consumer': 'S4 SSL MAE/Rel/summary, probes and fine-tuning; S5 direction/objects/auxiliary studies',
                   'corpus': 'complete S4 manifest, catalog and extension; includes reserved rows'},
                  {'consumer': 'S6 calibration, thresholds, manual error analysis, condition design and published test',
                   'corpus': 'entire original calibration/test plus transport/variant/smoke selections'}],
              'external_generator': {'manifest': str(generator_path), 'events': len(generator),
                                     'ledger': 'manifests/external_generator_usage.parquet',
                                     'identity': ['tune', 'source_path', 'source_row'],
                                     'accepted_run_namespace_disjoint': not generator.run.isin(inventory.run.unique()).any(),
                                     'vertex_donors': 'entire old calibration pool excluded',
                                     'foundation': 'existing accepted S6 independent-hard-scatter contract; no fabricated accepted-family keys'},
              'checks': {'all_previous_sources_retained': set(row['path'] for row in plan['historical_sources']).issubset(set(map(str, sources))),
                         'all_used_families_have_sources': bool((membership != 0).all()),
                         'additional_nested_sources_add_no_families': True,
                         'all_historical_roles_covered': True,
                         'public_confirm_identity_only': True,
                         'external_vertex_donors_excluded': True},
              'Confirm_sealed_rows_or_responses_read': False, 'seconds': time.perf_counter()-started,
              'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    if not all(record['checks'].values()):
        raise ValueError('historical usage closure failed')
    write_json(args.root/'usage_closure.json', record)
    print(json.dumps({'status': record['status'], 'accepted_sources': len(sources),
                      'families': len(ledger), 'external_generator_events': len(generator)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
