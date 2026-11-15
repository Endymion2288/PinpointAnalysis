"""Run the finite set of architecture-matched real learning gates in one claim."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--configs',type=Path,nargs='+',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    cases=[]
    for i,path in enumerate(args.configs):
        config=json.loads(path.read_text());attempt=args.attempt+f'_c{i}'
        tasks=['T4'] if config.get('native_global_ablation') else ['T3','T4']
        subprocess.run([sys.executable,'-m','analysis.flavour.v2.r3.fit','--root',str(args.root),
            '--config',str(path),'--attempt',attempt,'--tasks',*tasks],check=True)
        cases.append({'config':str(path),'attempt':attempt,'tasks':tasks,
            'summary':str(args.root/'fit'/attempt/'summary.json')})
    write_json(args.root/'fit'/args.attempt/'gates.json',{'status':'complete','cases':cases,
        'Cal_A_Cut_B_Confirm_payload_read':False})


if __name__=='__main__':main()
