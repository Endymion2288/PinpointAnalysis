"""Release batch diagnostics only after actual model and evaluation dependencies finish."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def complete(path):
    return path.exists() and json.loads(path.read_text()).get('status')=='complete'


def submit(root,name,module,arguments,gpu=False,cpus=4,memory='12GB'):
    path=root/'runtime'/f'{name}.submission.json'
    if path.exists() and json.loads(path.read_text()).get('submitted'):return
    command=[sys.executable,'-m','analysis.flavour.v2.submit','--root',str(root),'--stage','R2',
        '--name',name,'--cpus',str(cpus),'--memory',memory,'--job-flavour','workday']
    if gpu:command+=['--gpu','--min-capability','8','--disk','16GB']
    subprocess.run(command+[module,'--root',str(root),*arguments],check=True)


def advance(root,submitter=submit):
    campaign=json.loads((root/'campaign.json').read_text())
    names=campaign['initial_models']+campaign.get('extension_models',[])
    groups={}
    for name in names:
        config=json.loads((root/'configs'/f'{name}.json').read_text())
        groups.setdefault((config['family'],config['design']),[]).append(name)
    for (family,design),members in sorted(groups.items()):
        if not all(complete(root/'experiments'/name/'manifest.json') for name in members):continue
        if all(complete(root/'gap'/name/kind/'report.json') for name in members for kind in ['endpoint','best_screen']):continue
        attempt=f'gap_{family.lower()}_{design.lower()}_a1'
        submitter(root,'r2_'+attempt,'analysis.flavour.v2.r2.gap',
            ['run','--attempt',attempt,'--names',*members],gpu=True,cpus=4,memory='12GB')
    matrix_tag=f'm{len(names)}'
    if not all(complete(root/'experiments'/name/'manifest.json') for name in names):return False
    submitter(root,f'r2_evaluate_{matrix_tag}_a1','analysis.flavour.v2.r2.evaluate',
        ['--replicates','400','--workers','8'],cpus=8,memory='12GB')
    reports=[root/'evaluation'/name for name in ['results.json','independent_points.json','independent_core_intervals.json']]
    if not all(complete(path) for path in reports):return False
    result=json.loads(reports[0].read_text())
    expected={name+'_'+kind for name in names for kind in ['endpoint','best_screen']}
    if set(result['models'])!=expected:return False
    submitter(root,f'r2_figures_{matrix_tag}_a1','analysis.flavour.v2.r2.plots',[],cpus=2,memory='6GB')
    ready=all(complete(root/'gap'/name/kind/'report.json') for name in names for kind in ['endpoint','best_screen'])
    write_json(root/'runtime/finish_dependencies.json',{'status':'dependencies_ready' if ready else 'waiting_Train_gaps',
        'models':names,'scope':'batch delivery only; interpretation, conditional expansion, figures terminal check, accounting and acceptance remain'})
    return ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    started=time.monotonic()
    while True:
        if advance(args.root) or not args.watch:return
        if time.monotonic()-started>2*86400:raise TimeoutError('inspect unfinished scaling dependencies')
        time.sleep(30)


if __name__=='__main__':main()
