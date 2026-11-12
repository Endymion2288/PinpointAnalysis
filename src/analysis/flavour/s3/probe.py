"""Measure actual S3 GPU/container runtime before scientific fitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pyarrow
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    x = torch.ones((256, 256), device='cuda')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'GPU': torch.cuda.get_device_name(),
              'memory_bytes': torch.cuda.get_device_properties(0).total_memory,
              'BF16': torch.cuda.is_bf16_supported(including_emulation=False),
              'torch': torch.__version__, 'python': sys.version, 'pyarrow': pyarrow.__version__,
              'matmul_finite': bool(torch.isfinite(x @ x).all())}
    output = args.root / 'kubeflow/probe_environment.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
