"""Sample internal training health; reject encoder disconnection, not low AP."""
import torch


class TrainingHealthError(RuntimeError):
    pass


class TrainingHealth:
    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.activations = {}
        self.handles = []
        self.dead_samples = 0
        modules = dict(model.named_modules())
        names = ['local_pool', 'module_read', 'module_read_norm', 'head_input_norm',
                 'head.0', 'head.1', 'head.2', 'head.3', 'global_norm']
        for name in names:
            if name in modules:
                self.handles.append(modules[name].register_forward_hook(self.hook(name)))

    def hook(self, name):
        def record(module, inputs, output):
            if self.enabled:
                value = output.detach().float()
                self.activations.setdefault(name, []).append(torch.stack([
                    value.square().mean().sqrt(), value.min(), value.max(),
                    (value == 0).float().mean(), (value < -5).float().mean(),
                    value.isfinite().float().mean()]))
        return record

    def start(self):
        self.enabled = True
        self.activations = {}

    def finish(self, logits, step, check=True):
        self.enabled = False
        names = ['RMS', 'min', 'max', 'zero_fraction', 'fraction_below_minus5', 'finite_fraction']
        activations = {key: [dict(zip(names, row.cpu().tolist())) for row in rows]
                       for key, rows in self.activations.items()}
        gradients = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                group = name.split('.')[0]
                gradients.setdefault(group, []).append(parameter.grad.detach().float().square().sum())
        gradients = {key: float(torch.stack(values).sum()) for key, values in gradients.items()}
        probability = logits.detach().float().softmax(-1)
        n_unique = len(torch.unique(probability, dim=0))
        encoder = sum(gradients.get(key, 0.) for key in ['input', 'local', 'local_pool'])
        self.dead_samples = self.dead_samples + 1 if encoder == 0 else 0
        report = {'step': step, 'activations': activations, 'gradient_squared_norms': gradients,
                  'unique_batch_probability_vectors': n_unique, 'batch_events': len(probability),
                  'probability_std': probability.std(0, unbiased=False).cpu().tolist(),
                  'consecutive_exact_zero_encoder_gradient_samples': self.dead_samples}
        reasons = []
        if not all(row['finite_fraction'] == 1 for rows in activations.values() for row in rows):
            reasons.append('nonfinite internal activation')
        if not all(torch.isfinite(torch.tensor(v)).item() for v in gradients.values()):
            reasons.append('nonfinite internal gradient')
        if self.dead_samples >= 3:
            reasons.append('three sampled backward passes have exactly zero gradient to all local encoders')
        report['healthy'] = not reasons
        report['failure_reasons'] = reasons
        # The caller writes the evidence before acting on this decision.
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()
