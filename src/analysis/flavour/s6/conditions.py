"""Paired readout-proxy variations; field/material changes require transport."""
import numpy as np

from analysis.flavour.readout import modality_state, pixel_positions, tracker_measurements


def family_seed(meta):
    return int(meta['family_hi']) ^ int(meta['family_lo']) ^ 20261111


def stable_uniform(ids, salt=20261114):
    values = np.asarray(ids, dtype='uint64') ^ np.uint64(salt)
    values = (values ^ (values >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    values = (values ^ (values >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    values ^= values >> np.uint64(31)
    return (values >> np.uint64(11)).astype('float64')*2.**-53


def subset(fields, keep):
    return {key: np.asarray(value)[keep] for key, value in fields.items()}


class Condition:
    def __init__(self, geometry, parameters, hardware_map):
        self.parameters = parameters
        self.geometry = dict(geometry)
        self.shift = np.asarray(parameters.get('alignment_xy_mm', [0., 0.]))
        for axis, shift in zip('XY', self.shift):
            for key in [f'pixel_{axis}pos', f'scint_bar_{axis}center']:
                self.geometry[key] = (np.asarray(geometry[key])+shift).tolist()
        self.hardware = self.bar_hardware(hardware_map)

    def bar_hardware(self, mapping):
        g = self.geometry
        triples = np.asarray(mapping['triples_layer_panel_horizontal'], dtype='int64')
        nbar, npanel = len(g['scint_bar_Xcenter']), len(g['scint_Zpos'])
        layer, panel, horizontal = np.repeat(triples, nbar, axis=0).T
        bar = np.tile(np.arange(nbar), len(triples))
        ids = (layer*npanel+panel)*(2*nbar)+horizontal*nbar+bar
        x = np.where(horizontal, np.mean(g['scint_bar_Xcenter']), np.asarray(g['scint_bar_Xcenter'])[bar])
        y = np.where(horizontal, np.asarray(g['scint_bar_Ycenter'])[bar], np.mean(g['scint_bar_Ycenter']))
        hx = np.where(horizontal, g['scint_detector_width']/2, g['scint_bar_width']/2)
        hy = np.where(horizontal, g['scint_bar_height']/2, g['scint_detector_height']/2)
        result = {'channel_id': ids.astype('uint64'), 'layer': layer, 'panel': panel, 'horizontal': horizontal, 'bar': bar,
                  'bounds_mm': np.column_stack([x-hx, x+hx, y-hy, y+hy, np.asarray(g['scint_Zpos'])[panel]]).astype('float32')}
        return subset(result, np.argsort(ids))

    def __call__(self, original, meta, ideal=None):
        p = self.parameters
        seed = family_seed(meta)
        observation = {name: {k: np.asarray(v).copy() for k, v in original[name].items()}
                       for name in ['pixel', 'bar', 'tracker_hypothesis']}
        pixel, bars, tracker = [observation[n] for n in ['pixel', 'bar', 'tracker_hypothesis']]
        if self.shift.any():
            pixel['xyz_mm'][:, :2] += self.shift
            bars['bounds_mm'] += np.r_[self.shift[0], self.shift[0], self.shift[1], self.shift[1], 0.]
            tracker['xy_mm'] += self.shift
        if p.get('pixel_noise_mean', 0):
            rng = np.random.default_rng(seed ^ 18433)
            total = np.prod([len(self.geometry[f'pixel_{axis}pos']) for axis in 'XYZ'])
            ids = np.unique(np.r_[pixel['channel_id'], rng.integers(total, size=rng.poisson(p['pixel_noise_mean']))]).astype('uint64')
            pixel = {'channel_id': ids, 'xyz_mm': pixel_positions(ids, self.geometry).astype('float32')}
        if 'bar_gain' in p:
            bars['edep_proxy_MeV'] *= p['bar_gain']
            bars['pe_proxy'] = bars['pe_proxy']*p['bar_gain']
        if p.get('bar_noise_pe_mean_per_hardware_channel', 0):
            hw = self.hardware
            index = np.searchsorted(hw['channel_id'], bars['channel_id'])
            if not np.array_equal(hw['channel_id'][index], bars['channel_id']):
                raise ValueError('bar hardware map lacks an observed address')
            rng = np.random.default_rng(seed ^ 18434)
            noise = rng.poisson(p['bar_noise_pe_mean_per_hardware_channel'], len(hw['channel_id']))
            energy, pe, present = np.zeros(len(noise), dtype='float32'), noise.astype('float64'), noise > 0
            energy[index] = bars['edep_proxy_MeV']
            pe[index] += bars['pe_proxy']
            present[index] = True
            bars = subset(hw, present)
            bars.update(edep_proxy_MeV=energy[present], pe_proxy=pe[present])
        if 'bar_threshold_pe' in p:
            bars = subset(bars, bars['pe_proxy'] >= p['bar_threshold_pe'])
        for name, fields, salt in [('pixel', pixel, 18435), ('bar', bars, 18436)]:
            keep = np.ones(len(fields['channel_id']), dtype=bool)
            if name+'_survival' in p:
                keep &= np.random.default_rng(seed ^ salt).random(len(keep)) < p[name+'_survival']
            if name+'_dead_fraction' in p:
                keep &= stable_uniform(fields['channel_id'], salt) >= p[name+'_dead_fraction']
            if name == 'pixel' and 'pixel_dead_column_fraction_range' in p:
                nx = len(self.geometry['pixel_Xpos'])
                column = np.asarray(fields['channel_id'], dtype='int64') % nx/nx
                left, right = p['pixel_dead_column_fraction_range']
                keep &= ~((column >= left) & (column < right))
            if name == 'pixel':
                pixel = subset(fields, keep)
            else:
                bars = subset(fields, keep)
        if 'tracker_sigma_scale' in p:
            if ideal is None:
                raise ValueError('resolution variation requires ideal crossings')
            xyz = np.asarray(ideal['xyz_mm']).reshape(-1, 3)
            raw = {'trackerID': np.asarray(ideal['station']), **{axis: xyz[:, i] for i, axis in enumerate('xyz')}}
            scale = p['tracker_sigma_scale']
            _, tracker = tracker_measurements(raw, seed, sigma_x_mm=.58*scale, sigma_y_mm=.017*scale)
            tracker['xy_mm'] += self.shift
        if 'tracker_efficiency' in p:
            keep = np.random.default_rng(seed ^ 18437).random(len(tracker['station'])) < p['tracker_efficiency']/.99
            tracker = subset(tracker, keep)
        if 'drop_tracker_station' in p:
            tracker = subset(tracker, tracker['station'] != p['drop_tracker_station'])
        state = modality_state(pixel, bars, tracker)
        return {'pixel': pixel, 'bar': bars, 'tracker_hypothesis': tracker, **state}
