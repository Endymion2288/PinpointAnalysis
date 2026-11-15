"""Run gates then one frozen R3 fit in a single obtained allocation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if not Path(config['preflight']).exists():
        subprocess.run([sys.executable,'-m','analysis.flavour.v2.r3.campaign','preflight',
            '--root',str(args.root),'--configs',str(args.config)],check=True)
    subprocess.run([sys.executable,'-m','analysis.flavour.v2.r3.smoke',
        '--root',str(args.root),'--config',str(args.config),'--attempt',args.attempt],check=True)
    subprocess.run([sys.executable,'-m','analysis.flavour.v2.r3.train',
        '--root',str(args.root),'--config',str(args.config)],check=True)


if __name__=='__main__':
    main()
