#!/bin/bash
# Source after LCG108. Dependencies and compiler scratch stay on the worker.
PINPOINT_REPAIR_ROOT="$1"
PINPOINT_REPAIR_RUNTIME="${_CONDOR_SCRATCH_DIR:-/tmp}/pinpoint_s3r_runtime"
mkdir -p "$PINPOINT_REPAIR_RUNTIME"
if [ ! -f "$PINPOINT_REPAIR_RUNTIME/python/triton/__init__.py" ]; then
    python -m pip install --no-deps --no-cache-dir --quiet --target "$PINPOINT_REPAIR_RUNTIME/python" triton==3.3.1
fi
tar -xzf "$PINPOINT_REPAIR_ROOT/audit/compiler_headers.tar.gz" -C "$PINPOINT_REPAIR_RUNTIME"
tar -xzf "$PINPOINT_REPAIR_ROOT/audit/compiler_linker_files.tar.gz" -C "$PINPOINT_REPAIR_RUNTIME"
sed -i "s@/usr/lib64/libc_nonshared.a@$PINPOINT_REPAIR_RUNTIME/pinpoint_lib64/libc_nonshared.a@g" "$PINPOINT_REPAIR_RUNTIME/pinpoint_lib64/libc.so"
export PYTHONPATH="$PINPOINT_REPAIR_RUNTIME/python:${PYTHONPATH:-}"
export C_INCLUDE_PATH="$PINPOINT_REPAIR_RUNTIME/pinpoint_headers:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$PINPOINT_REPAIR_RUNTIME/pinpoint_headers:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$PINPOINT_REPAIR_RUNTIME/pinpoint_lib64:${LIBRARY_PATH:-}"
export TORCHINDUCTOR_CACHE_DIR="$PINPOINT_REPAIR_RUNTIME/inductor"
export TRITON_CACHE_DIR="$PINPOINT_REPAIR_RUNTIME/triton"
export TORCHINDUCTOR_COMPILE_THREADS=1
python - "$PINPOINT_REPAIR_ROOT" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import torch
import triton
output = Path(sys.argv[1]) / 'runtime'
output.mkdir(exist_ok=True)
record = {'utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
          'torch': torch.__version__, 'triton': triton.__version__,
          'local_encoder_compilation': 'torch.compile(dynamic=True), identical encoder in health samples; hooks outside compiled features',
          'compiler_assets': ['audit/compiler_headers.tar.gz', 'audit/compiler_linker_files.tar.gz'],
          'scope': 'execution dependencies; no detector or model data changes'}
(output / f'{socket.gethostname()}_{os.getpid()}.json').write_text(json.dumps(record, indent=2)+'\n')
PY
