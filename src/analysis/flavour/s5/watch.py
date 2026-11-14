"""Persist short-lived Kubernetes terminal states before platform Pod cleanup."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    directory = args.root/'runtime/backends'
    directory.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(['kubectl', '--request-timeout=0', 'get', 'pods', '-n', 'xcheng',
        '-l', 'job-name='+args.job, '--watch', '--output-watch-events', '-o', 'json'],
        stdout=subprocess.PIPE, text=True)
    buffer = ''
    decoder = json.JSONDecoder()
    try:
        for line in process.stdout:
            buffer += line
            while buffer.strip():
                try:
                    event, end = decoder.raw_decode(buffer.lstrip())
                except json.JSONDecodeError:
                    break
                buffer = buffer.lstrip()[end:]
                pod = event.get('object', event)
                if pod.get('kind') != 'Pod':
                    continue
                name = pod['metadata']['name']
                phase = pod.get('status', {}).get('phase')
                path = directory/(name+('.terminal.pod.json' if phase in ['Succeeded', 'Failed'] else '.live.pod.json'))
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(pod, indent=2)+'\n')
                temporary.replace(path)
                if phase in ['Succeeded', 'Failed']:
                    print(json.dumps({'pod': name, 'phase': phase, 'snapshot': str(path)}), flush=True)
                    return
        if process.wait() != 0:
            raise RuntimeError('Kubernetes watch ended before a terminal Pod; restart after checking CLI credentials')
    finally:
        process.terminate()


if __name__ == '__main__':
    main()
