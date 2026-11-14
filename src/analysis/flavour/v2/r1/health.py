"""Observe branch activations outside compiled subgraphs without changing execution."""

import torch


class Activations:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.values = {}

    def start(self):
        self.values = {}
        names = (['backbone', 'tracker'] if hasattr(self.model, 'backbone') else
                 ['local_pool', 'module_read', 'latent_read', 'global_norm', 'summary'])
        for name in names:
            module = getattr(self.model, name)
            def capture(_, inputs, output, name=name):
                value = output.detach().float()
                self.values.setdefault(name, []).append(torch.stack([
                    value.square().mean(), value.abs().amax(), torch.isfinite(value).all().float()]))
            self.handles.append(module.register_forward_hook(capture))

    def finish(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        result = {}
        for name, values in self.values.items():
            value = torch.stack(values).cpu()
            result[name] = {'RMS': float(value[:, 0].mean().sqrt()),
                'maximum_absolute': float(value[:, 1].max()), 'finite': bool(value[:, 2].all()),
                'calls_including_checkpoint_recompute': len(values)}
        return result
