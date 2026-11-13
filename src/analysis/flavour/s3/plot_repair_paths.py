"""Display fitted path reliance and token diversity without a collapse threshold."""
import argparse
import json
from pathlib import Path

import numpy as np


def spectrum_summary(records):
    """Summarize the recorded groups; this is not the full-event token rank."""
    groups = np.asarray([row['groups'] for row in records], dtype=float)
    ranks = np.asarray([row['mean_effective_rank'] for row in records])
    return {
        'recorded_batches': len(records), 'recorded_groups': int(groups.sum()),
        'queries_per_group': records[0]['queries'],
        'group_weighted_mean_effective_rank': float(np.average(ranks, weights=groups)),
        'batch_mean_effective_rank_range': [float(ranks.min()), float(ranks.max())],
        'group_weighted_query_variation_RMS': float(np.average(
            [row['within_group_query_variation_RMS'] for row in records], weights=groups)),
    }


def plot(root, names, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for name in names:
            path = root / 'experiments' / name / 'path_diagnostics.json'
            if not path.exists():
                continue  # The independently retrained ParT has no matching branches.
            report = json.loads(path.read_text())
            branches = [key for key in report['branch_permutation']
                        if key not in ['all_spatial', 'all_regional', 'all_local']]
            y = np.arange(len(branches))
            labels = [key.replace('_', ' ') for key in branches]
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
            for axis, metric, title in [
                (axes[0, 0], 'probe_delta_NLL', 'Branch permutation: NLL change'),
                (axes[0, 1], 'probability_change_RMS', 'Branch permutation: prediction change')]:
                values = [report['branch_permutation'][key][metric] for key in branches]
                axis.barh(y, values)
                axis.axvline(0, color='gray', linewidth=.8)
                axis.set(yticks=y, yticklabels=labels, xlabel=metric.replace('_', ' '), title=title)
                axis.invert_yaxis()
                axis.grid(axis='x', alpha=.2)
            values = [report['branch_between_event_variation_RMS'][key] for key in branches]
            axes[1, 0].barh(y, np.maximum(values, 1e-8))
            axes[1, 0].set(yticks=y, yticklabels=labels, xscale='log',
                           xlabel='Between-event branch variation RMS (zero shown at 1e-8)',
                           title='Input-dependent branch variation')
            axes[1, 0].invert_yaxis()
            spectra = {key: spectrum_summary(rows) for key, rows in report['token_spectrum'].items()}
            for i, (key, row) in enumerate(spectra.items()):
                rank = row['group_weighted_mean_effective_rank']
                axes[1, 1].barh(i, rank)
                axes[1, 1].scatter(row['queries_per_group'], i, color='black', marker='|', s=100)
                axes[1, 1].annotate(f"{rank:.4f} / {row['queries_per_group']} queries", (rank, i),
                                    xytext=(4, 4), textcoords='offset points', fontsize=8)
            axes[1, 1].set(yticks=range(len(spectra)), yticklabels=list(spectra), xscale='log',
                           xlabel='Uncentered squared-spectrum entropy rank; black tick = query count',
                           title='Recorded groups: effective rank and query count')
            axes[1, 1].invert_yaxis()
            controls = report['head_controls']
            nonlinear = controls['nonlinear_head_only']['probe_delta_NLL']
            linear = controls['linear_readout_only']['probe_delta_NLL']
            fig.suptitle(
                f"{name}: {report['probe_events']}-event CPU FP32 path diagnostic\n"
                f"Probe NLL {report['probe_NLL']:.5f}; head-only delta NLL: nonlinear {nonlinear:+.5f}, linear {linear:+.5f}\n"
                'Same-presence event permutation; occupancy-quantile probe, not population performance. '
                'Removal has no retraining.', fontsize=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.experiments, args.output)


if __name__ == '__main__':
    main()
