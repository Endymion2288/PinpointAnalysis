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
        self.informative_nodes = None
        self.token_stats = {}
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
                    value.isfinite().all().float()]))
                if value.ndim == 3 and value.shape[1] > 1 and name == 'module_read':
                    unit = torch.nn.functional.normalize(value, dim=-1)
                    cosine = unit @ unit.transpose(-1, -2)
                    count = value.shape[1]
                    eye = torch.eye(count, device=value.device, dtype=torch.bool)
                    self.token_stats[name] = {
                        'mean_off_diagonal_query_cosine': float(cosine[:, ~eye].mean()),
                        'within_group_query_variation_RMS': float(value.std(1, unbiased=False).square().mean().sqrt())}
        return record

    def start(self, batch=None):
        self.enabled = True
        self.activations = {}
        self.token_stats = {}
        self.informative_nodes = (int(((batch['nodes'][..., 15] > 0) & batch['local_mask']).sum())
                                  if batch is not None else None)
        for name, module in self.model.named_modules():
            if hasattr(module, 'qnorm') and not name.startswith('local'):
                module.record_health = True

    def finish(self, logits, step, check=True):
        self.enabled = False
        attention = {}
        for name, module in self.model.named_modules():
            if getattr(module, 'record_health', False):
                module.record_health = False
                if hasattr(module, 'health_probe'):
                    attention[name] = module.health_probe
        names = ['RMS', 'min', 'max', 'zero_fraction', 'fraction_below_minus5', 'all_finite']
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
        if self.informative_nodes != 0:
            self.dead_samples = self.dead_samples + 1 if encoder == 0 else 0
        report = {'step': step, 'activations': activations, 'gradient_squared_norms': gradients,
                  'query_diversity': self.token_stats, 'sampled_attention': attention,
                  'informative_nodes': self.informative_nodes,
                  'unique_batch_probability_vectors': n_unique, 'batch_events': len(probability),
                  'probability_std': probability.std(0, unbiased=False).cpu().tolist(),
                  'consecutive_exact_zero_encoder_gradient_samples': self.dead_samples}
        reasons = []
        if not all(row['all_finite'] == 1 for rows in activations.values() for row in rows):
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
