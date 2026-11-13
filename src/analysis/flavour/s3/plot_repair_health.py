"""Plot activation, gradient, score and regional-query behavior for every fit."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(args.output) as pdf:
        for name in args.experiments:
            directory = args.root / 'experiments' / name
            rows = json.loads((directory / 'training_health.json').read_text())['records']
            steps = np.array([r['step'] for r in rows])
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
            for key in ['local_pool', 'module_read_norm', 'head_input_norm', 'head.0', 'head.1']:
                values = [max((v['RMS'] for v in r['activations'].get(key, [])), default=np.nan) for r in rows]
                if np.isfinite(values).any():
                    axes[0, 0].plot(steps, values, label=key, linewidth=.8)
            axes[0, 0].set(xlabel='Optimizer update', ylabel='Activation RMS', yscale='log', title='Sampled layer scales')
            axes[0, 0].legend(fontsize=7)
            for key in ['input', 'local', 'local_pool', 'module_read', 'latent_read', 'head', 'linear_readout']:
                values = [r['gradient_squared_norms'].get(key, np.nan) for r in rows]
                if np.isfinite(values).any():
                    axes[0, 1].plot(steps, np.maximum(values, 1e-30), label=key, linewidth=.8)
            axes[0, 1].set(xlabel='Optimizer update', ylabel='Gradient squared norm before clipping',
                           yscale='log', title='Encoder and classifier connectivity')
            axes[0, 1].legend(fontsize=7)
            for c in range(4):
                axes[0, 2].plot(steps, [r['probability_std'][c] for r in rows], label=['CC nue', 'CC numu', 'CC nutau', 'NC'][c], linewidth=.8)
            axes[0, 2].set(xlabel='Optimizer update', ylabel='Within-batch probability SD', title='Sampled score variation')
            axes[0, 2].legend(fontsize=7)
            query = [r.get('query_diversity', {}).get('module_read', {}) for r in rows]
            axes[1, 0].plot(steps, [q.get('mean_off_diagonal_query_cosine', np.nan) for q in query], linewidth=.8)
            axes[1, 0].set(xlabel='Optimizer update', ylabel='Mean off-diagonal cosine', title='Queries within a module / region')
            second = axes[1, 0].twinx()
            second.plot(steps, [q.get('within_group_query_variation_RMS', np.nan) for q in query], color='C1', linewidth=.8)
            second.set_ylabel('Within-group query variation RMS', color='C1')
            for key in ['module_read', 'latent_read']:
                values = [r.get('sampled_attention', {}).get(key, {}).get('conditional_entropy_nats', np.nan) for r in rows]
                if np.isfinite(values).any():
                    axes[1, 1].plot(steps, values, label=key, linewidth=.8)
            axes[1, 1].set(xlabel='Optimizer update', ylabel='Conditional entropy (nats)',
                           title='First group; at most 8 queries / 128 keys')
            if axes[1, 1].lines:
                axes[1, 1].legend(fontsize=7)
            history_path = directory / 'history.json'
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            epochs = [r['epoch'] for r in history]
            axes[1, 2].plot(epochs, [r['train_loss'] for r in history], marker='.', label='Train')
            axes[1, 2].plot(epochs, [r['validation']['NLL'] for r in history], marker='.', label='Full validation')
            axes[1, 2].axhline(np.log(4), color='gray', linestyle='--', label='Uniform four-class NLL')
            axes[1, 2].set(xlabel='Epoch', ylabel='NLL', title='Learning and checkpoint selection')
            axes[1, 2].legend(fontsize=7)
            fig.suptitle(f'{name}\nHealth samples describe optimization; validation measures generalization', fontsize=10)
            pdf.savefig(fig); plt.close(fig)


if __name__ == '__main__':
    main()
