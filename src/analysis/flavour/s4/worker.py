"""Execute one frozen S4 task of a mixed formal batch campaign."""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    config=json.loads(args.campaign.read_text())['experiments'][args.task]
    program=config.get('program','train' if config.get('objective')=='classification' else 'pretrain')
    if program not in {'pretrain','train','probes','pilot','functional_probe','transfer_probe'}:
        raise ValueError('unsupported S4 task program')
    if program=='pretrain' and config.get('required_pilot'):
        pilot=args.root/'ssl_gate'/config['required_pilot']/'gate.json'
        report=json.loads(pilot.read_text())
        if not report['passed']:raise RuntimeError('pretraining requires a passing real-event optimization/stress pilot')
        for key in ['dim','compression','latents','region_queries','region_blocks','readout_skip','modalities',
            'window_pixels','pixel_query_patch','mask_ratio','teacher_multiscale','objective','compile_local','local_chunk_blocks',
            'variance_gamma','variance_weight','covariance_weight','distill_reconstruction_weight','rel_source']:
            if report['config'].get(key)!=config.get(key):raise ValueError('full/pilot architecture or objective mismatch: '+key)
        readback=json.loads((args.root/'experiments'/config['required_pilot']/'ssl_readback.json').read_text())
        if not readback['integrity_and_internal_health_pass'] or not readback['measured_learning_pass']:
            raise RuntimeError('corpus pilot must show measured learning before full pretraining')
    importlib.import_module('analysis.flavour.s4.'+program).main()
    if program=='pretrain' or (program=='pilot' and config.get('objective')!='backbone_health'):
        from analysis.flavour.s4.readback import inspect
        report=inspect(args.root,config['name'])
        if not report['integrity_and_internal_health_pass'] or not report['measured_learning_pass']:
            # Preserve the fit and full evidence; downstream dependencies must
            # stop until objective/architecture/optimization have been reviewed.
            from analysis.flavour.inventory import write_json
            path=args.root/'experiments'/config['name']/'manifest.json'
            manifest=json.loads(path.read_text());manifest['status']='learning_review_required'
            write_json(path,manifest)
            raise RuntimeError('full pretraining readback or measured-learning check failed')


if __name__=='__main__':main()
