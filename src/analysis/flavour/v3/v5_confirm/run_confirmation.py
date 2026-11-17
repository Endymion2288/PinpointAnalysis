"""Top-level W3-12 confirmation orchestrator (prepare -> infer -> Cal-A/Cut-B/Confirm -> plots)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.flavour.v3.v5_confirm import DEFAULT_S6_ROOT, DEFAULT_V2_ROOT, DEFAULT_V3_ROOT
from analysis.flavour.v3.v5_confirm.freeze import require_v5_freeze
from analysis.flavour.v3.v5_confirm.infer_unsealed import run_family_inference
from analysis.flavour.v3.v5_confirm.plots import generate_all_v5_figures
from analysis.flavour.v3.v5_confirm.prepare_unsealed import convert_range, finalize_unsealed, plan_unsealed
from analysis.flavour.v3.v5_confirm.significance import run_sequential_unsealing


def run_all_statistics_and_plots(*, v3_root: Path = DEFAULT_V3_ROOT, n_boot: int = 1000) -> dict:
    """Verify freeze, run strict Cal-A -> Cut-B -> Confirm200k evaluation, and emit figures."""
    w3_12 = v3_root / 'W3-12'
    require_v5_freeze(w3_12 / 'analysis_freeze_v5.json', verify_git_commit=True, verify_checkpoints=True)
    res = run_sequential_unsealing(v3_root=v3_root, n_boot=n_boot)
    figs = generate_all_v5_figures(v3_root=v3_root)
    return {'status': res['status'], 'figures': figs, 'events': res['confirm200k']['events']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'stage',
        choices=('plan', 'convert', 'finalize', 'infer-pf', 'infer-b1pp', 'evaluate', 'plots', 'all-stats'),
    )
    parser.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument('--v2-root', type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument('--s6-root', type=Path, default=DEFAULT_S6_ROOT)
    parser.add_argument('--task-start', type=int, default=0)
    parser.add_argument('--task-count', type=int, default=None)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--n-boot', type=int, default=1000)
    args = parser.parse_args()

    if args.stage == 'plan':
        out = plan_unsealed(v3_root=args.v3_root, v2_root=args.v2_root)
    elif args.stage == 'convert':
        out = convert_range(v3_root=args.v3_root, task_start=args.task_start, task_count=args.task_count, cpus=args.cpus)
    elif args.stage == 'finalize':
        out = finalize_unsealed(v3_root=args.v3_root)
    elif args.stage == 'infer-pf':
        out = run_family_inference(family='pf', v3_root=args.v3_root, s6_root=args.s6_root, workers=args.workers)
    elif args.stage == 'infer-b1pp':
        out = run_family_inference(family='b1pp', v3_root=args.v3_root, s6_root=args.s6_root, workers=args.workers)
    elif args.stage == 'evaluate':
        out = run_sequential_unsealing(v3_root=args.v3_root, n_boot=args.n_boot)
    elif args.stage == 'plots':
        out = {'figures': generate_all_v5_figures(v3_root=args.v3_root)}
    else:
        out = run_all_statistics_and_plots(v3_root=args.v3_root, n_boot=args.n_boot)
    print(json.dumps(out if isinstance(out, dict) else {'done': len(out)}, indent=2))


if __name__ == '__main__':
    main()
