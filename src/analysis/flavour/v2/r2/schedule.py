"""Prescribed optimizer-update schedules and metadata-only sampler evidence."""

from collections import Counter
import math

import numpy as np


def learning_rate(step, total_updates, maximum, warmup=.05, floor=.05):
    progress = min(1., max(0., step/max(1, total_updates-1)))
    factor = (progress/warmup if progress<warmup else
              floor+(1-floor)*.5*(1+math.cos(math.pi*(progress-warmup)/(1-warmup))))
    return maximum*max(floor, factor)


def sampler_trace(sampler, labels, passes):
    """Replay every complete pass; tails and mixed windows may change U by pass."""
    labels = np.asarray(labels)
    sizes, classes, per_pass, windows = [], [], [], []
    exposure = np.zeros(len(labels), dtype='uint16')
    for epoch in range(passes):
        sampler.epoch = epoch
        updates, totals = 0, np.zeros(4, dtype='int64')
        for ix in sampler._windows():
            count = np.bincount(labels[ix], minlength=4)
            windows.append({'pass':epoch+1, 'events':len(ix), 'class_counts':count.tolist()})
        for batch in sampler:
            count = np.bincount(labels[batch], minlength=4)
            sizes.append(len(batch))
            classes.append(count)
            totals += count
            np.add.at(exposure, batch, 1)
            updates += 1
        per_pass.append({'pass':epoch+1, 'updates':updates, 'events':int(totals.sum()),
                         'class_counts':totals.tolist()})
    if not np.all(exposure==passes):
        raise ValueError('sampler does not expose every Train row exactly once per pass')
    sizes = np.asarray(sizes, dtype='int16')
    classes = np.asarray(classes, dtype='int16')
    proportions = classes/sizes[:, None]
    lag1 = []
    for c in range(4):
        values = proportions[:, c]
        lag1.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]) if values.std()>0 else None)
    report = {'total_updates':len(sizes), 'total_exposures':int(sizes.sum()),
        'non_tau_exposures':int(sizes.sum()-classes[:, 2].sum()),
        'unique_events':len(labels), 'passes':passes, 'per_pass':per_pass,
        'batch_size_distribution':dict(sorted(Counter(map(int, sizes)).items())),
        'single_class_batch_fraction':float((np.count_nonzero(classes, axis=1)==1).mean()),
        'class_proportion_lag1':lag1, 'terminal_100_batches_class_counts':classes[-100:].sum(0).tolist(),
        'minimum_row_exposures':int(exposure.min()), 'maximum_row_exposures':int(exposure.max()),
        'windows':windows}
    return report, sizes, classes
