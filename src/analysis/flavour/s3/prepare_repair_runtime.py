"""Stage small EL9 compiler headers/linker assets for the minimal GPU image."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import tarfile

from analysis.flavour.inventory import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root / 'audit'; output.mkdir(parents=True, exist_ok=True)
    base = Path('/usr/include')
    with tarfile.open(output / 'compiler_headers.tar.gz', 'w:gz') as tar:
        for path in sorted(base.glob('*.h')):
            tar.add(path, arcname='pinpoint_headers/' + path.name)
        for name in ['bits', 'sys', 'gnu', 'linux', 'asm', 'asm-generic']:
            if (base / name).exists():
                tar.add(base / name, arcname='pinpoint_headers/' + name)
    with tarfile.open(output / 'compiler_linker_files.tar.gz', 'w:gz', dereference=True) as tar:
        for name in ['crti.o', 'crtn.o', 'libc.so', 'libc_nonshared.a', 'libpthread_nonshared.a',
                     'libm.so', 'libmvec.so', 'libpthread.so']:
            path = Path('/usr/lib64') / name
            if path.exists():
                tar.add(path, arcname='pinpoint_lib64/' + name)
    write_json(output / 'compiler_assets.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'source_os': platform.freedesktop_os_release(),
        'files': {p.name: p.stat().st_size for p in output.glob('compiler_*.tar.gz')},
        'scope': 'public EL9 headers and linker support; runtime libraries supplied by worker image; no user or detector data'})


if __name__ == '__main__':
    main()
