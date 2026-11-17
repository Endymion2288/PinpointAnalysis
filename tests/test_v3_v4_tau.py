"""Unit tests for Phase V4 NOMAD kinematics, DRO pAUC finetuning, OOF ensembles, and AST isolation."""

import ast
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common import datasets, domains
from analysis.flavour.v3.v1_b1pp.representation import collate_b1pp, extract_b1pp_item, static_plane_layout
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v4_tau import (
    DROPartialAUCLoss,
    NOMAD_FEAT_DIM,
    NomadTransverseExtractor,
    PinpointFormerV4,
    ZAlignedStereoNetV4,
    ensemble,
    fit_oof_domain_ensemble,
    models as v4_models,
    nomad_kinematics,
    pauc_finetune,
)
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet


GEOMETRY_PATH = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json')


def _make_synthetic_obs(geometry: dict, bounds: dict, *, acoplanar: bool = False, with_tracker: bool = True) -> dict:
    vp = np.asarray(bounds['vert_panels'], dtype=np.int64)
    hp = np.asarray(bounds['horiz_panels'], dtype=np.int64)
    sz = np.asarray(bounds['scint_Zpos'], dtype=np.float64)

    # Pick two doublet panels at stations s=2 and s=5
    doublet_pairs = []
    for pv in vp:
        diffs = np.abs(sz[hp] - sz[pv])
        j = int(np.argmin(diffs))
        if diffs[j] < 15.0:
            doublet_pairs.append((int(pv), int(hp[j])))

    panels, horizs, bars, edeps, pes = [], [], [], [], []
    # Upstream apex near center (bar 20, 20)
    for k, (pv, ph) in enumerate(doublet_pairs[2:8]):
        # Leading prong moving slightly +x, +y
        bx_lead = min(39, 20 + (k // 2))
        by_lead = min(39, 22 + (k // 2))
        panels.extend([pv, ph])
        horizs.extend([0, 1])
        bars.extend([bx_lead, by_lead])
        edeps.extend([85.0, 90.0])
        pes.extend([1200, 1300])

        # Recoil prong: either back-to-back (-x, -y) -> coplanar, or orthogonal (+y, -x) -> acoplanar
        if acoplanar:
            bx_had = max(0, 14 - k)
            by_had = min(39, 24 + k)
        else:
            bx_had = max(0, 19 - k)
            by_had = max(0, 18 - k)
        panels.extend([pv, ph])
        horizs.extend([0, 1])
        bars.extend([bx_had, by_had])
        edeps.extend([25.0, 25.0])
        pes.extend([350, 350])

    if with_tracker:
        # Three stations with clean sagitta
        st_arr = np.array([0, 1, 2], dtype=np.int8)
        xy_arr = np.array([[5.0, -7.0], [10.0, -5.0], [15.0, -2.5]], dtype=np.float32)
        sxy = np.full((3, 2), 0.1, dtype=np.float32)
    else:
        st_arr = np.empty(0, dtype=np.int8)
        xy_arr = np.empty((0, 2), dtype=np.float32)
        sxy = np.empty((0, 2), dtype=np.float32)

    return {
        'pixel': {
            'channel_id': np.array([10, 11], dtype=np.int64),
            'xyz_mm': np.array([[0.0, -12.0, -2981.995], [0.02, -12.0, -2981.995]], dtype=np.float32),
        },
        'bar': {
            'channel_id': np.arange(len(panels), dtype=np.int64),
            'layer': np.zeros(len(panels), dtype=np.int16),
            'panel': np.asarray(panels, dtype=np.int16),
            'horizontal': np.asarray(horizs, dtype=np.int8),
            'bar': np.asarray(bars, dtype=np.int16),
            'bounds_mm': np.zeros((len(panels), 5), dtype=np.float32),
            'edep_proxy_MeV': np.asarray(edeps, dtype=np.float32),
            'pe_proxy': np.asarray(pes, dtype=np.int64),
        },
        'tracker_hypothesis': {
            'station': st_arr,
            'xy_mm': xy_arr,
            'sigma_xy_mm': sxy,
        },
    }


def _empty_obs() -> dict:
    return {
        'pixel': {'channel_id': np.empty(0, dtype=np.int64), 'xyz_mm': np.empty((0, 3), dtype=np.float32)},
        'bar': {
            'channel_id': np.empty(0, dtype=np.int64),
            'layer': np.empty(0, dtype=np.int16),
            'panel': np.empty(0, dtype=np.int16),
            'horizontal': np.empty(0, dtype=np.int8),
            'bar': np.empty(0, dtype=np.int16),
            'bounds_mm': np.empty((0, 5), dtype=np.float32),
            'edep_proxy_MeV': np.empty(0, dtype=np.float32),
            'pe_proxy': np.empty(0, dtype=np.int64),
        },
        'tracker_hypothesis': {
            'station': np.empty(0, dtype=np.int8),
            'xy_mm': np.empty((0, 2), dtype=np.float32),
            'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
        },
    }


class TestV4Tau(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(GEOMETRY_PATH.read_text())
        cls.bounds = domains.detector_bounds(cls.geometry)
        cls.layout = static_plane_layout(cls.geometry, cls.bounds)
        cls.layout['geometry'] = cls.geometry

    def test_ast_zero_truth_and_sealed_literals(self) -> None:
        forbidden = (
            'truth_sidecar',
            'cluster_sidecar',
            'event_sidecar',
            'initX',
            'initY',
            'initZ',
            'initE',
            'Cal-A',
            'Cut-B',
            'Confirm200k',
            'sealed',
        )
        for mod in (nomad_kinematics, pauc_finetune, v4_models, ensemble):
            raw = Path(mod.__file__).read_text()
            for tok in forbidden:
                self.assertNotIn(tok, raw, msg=f'{mod.__name__} contains forbidden token {tok!r}')
            tree = ast.parse(raw)
            strs = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            for tok in forbidden:
                self.assertFalse(any(tok in s for s in strs))
        self.assertNotIn('sidecar_root', inspect.signature(datasets.V3EvalDataset.__init__).parameters)

    def test_nomad_b1pp_pf_equivalence_mirror_invariance_and_physics(self) -> None:
        extractor = NomadTransverseExtractor(self.geometry, self.bounds).eval()
        obs_cop = _make_synthetic_obs(self.geometry, self.bounds, acoplanar=False, with_tracker=False)
        obs_aco = _make_synthetic_obs(self.geometry, self.bounds, acoplanar=True, with_tracker=False)
        obs_trk = _make_synthetic_obs(self.geometry, self.bounds, acoplanar=True, with_tracker=True)
        obs_emp = _empty_obs()

        raw_list = [obs_cop, obs_aco, obs_trk, obs_emp]
        b1_items, pf_items = [], []
        for idx, ob in enumerate(raw_list):
            dom = domains.compute_observable_containment(ob, self.geometry, self.bounds)
            b1_items.append(extract_b1pp_item(idx, ob, idx % 4, dom, self.geometry, self.bounds, {}))
            pf_items.append(extract_pf_item(idx, ob, idx % 4, dom, self.geometry, self.bounds, {}))

        b1_batch = collate_b1pp(b1_items, mirror_rng=None)
        pf_batch = collate_pf(pf_items, mirror_rng=None, y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])

        f_b1 = extractor.from_b1pp(b1_batch)
        f_pf = extractor.from_pf(pf_batch)
        self.assertEqual(tuple(f_b1.shape), (4, NOMAD_FEAT_DIM))
        self.assertTrue(torch.allclose(f_b1, f_pf, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.all(f_pf[3] == 0.0))

        # Exact Y-mirror invariance on both B1++ and PF
        class _AlwaysFlipRng:
            def random(self, n: int) -> np.ndarray:
                return np.zeros(n, dtype=np.float64)

        b1_mir = collate_b1pp(b1_items, mirror_rng=_AlwaysFlipRng())
        pf_mir = collate_pf(pf_items, mirror_rng=_AlwaysFlipRng(), y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])
        self.assertTrue(torch.allclose(extractor.from_b1pp(b1_mir), f_b1, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(extractor.from_pf(pf_mir), f_pf, atol=1e-4, rtol=1e-4))

        # Physical sanity: acoplanar has higher acoplanarity and pt_miss than back-to-back coplanar
        self.assertGreater(float(f_pf[1, 1]), float(f_pf[0, 1]))
        self.assertGreater(float(f_pf[1, 3]), float(f_pf[0, 3]))
        self.assertNotEqual(float(f_pf[2, 8]), 0.0)

    def test_dro_pauc_gradients_and_empty_domain_exclusion(self) -> None:
        torch.manual_seed(7)
        crit = DROPartialAUCLoss(beta=0.01, margin=0.40)
        logits_act = torch.randn(32, 4, requires_grad=True)
        labels_act = torch.arange(32) % 4
        dom_act = torch.ones(32, dtype=torch.long)

        loss_a, _ = crit(torch.softmax(logits_act, dim=-1), labels_act, dom_act)
        self.assertTrue(bool(crit.tau_q_initialized.item()))
        loss_a.backward()
        self.assertTrue(torch.isfinite(logits_act.grad).all())
        self.assertGreater(float(logits_act.grad.abs().sum()), 0.0)
        self.assertIsNotNone(crit.tau_q.grad)
        self.assertNotEqual(float(crit.tau_q.grad.item()), 0.0)

        # Appending D_empty rows must leave loss identical
        logits_aug = torch.cat([logits_act.detach(), torch.randn(8, 4)], dim=0)
        labels_aug = torch.cat([labels_act, torch.full((8,), 2, dtype=torch.long)], dim=0)
        dom_aug = torch.cat([dom_act, torch.zeros(8, dtype=torch.long)], dim=0)
        loss_b, _ = crit(torch.softmax(logits_aug, dim=-1), labels_aug, dom_aug)
        self.assertAlmostEqual(float(loss_a.item()), float(loss_b.item()), places=5)

        # Boundary positives (logit_tau ~ 0.0) and high-FP negatives (logit_tau ~ +1.0)
        # must receive strictly stronger gradients than hopeless positives (~ -6.0) and easy negatives (~ -5.0).
        crit_f = DROPartialAUCLoss(beta=0.01, margin=0.40)
        logits_f = torch.zeros(40, 4)
        labels_f = torch.tensor([2, 2, 2, 2] + [0] * 36, dtype=torch.long)
        dom_f = torch.ones(40, dtype=torch.long)
        logits_f[0, 2] = 0.0
        logits_f[1, 2] = -6.0
        logits_f[2, 2] = 0.2
        logits_f[3, 2] = -0.5
        logits_f[4, 2] = 1.0
        logits_f[5:10, 2] = 0.5
        logits_f[10:, 2] = -5.0
        logits_f.requires_grad_(True)
        loss_f, _ = crit_f(torch.softmax(logits_f, dim=-1), labels_f, dom_f)
        loss_f.backward()
        gnorm = logits_f.grad.norm(dim=-1)
        self.assertGreater(float(gnorm[0]), float(gnorm[1]))
        self.assertGreater(float(gnorm[4]), float(gnorm[15]))

    def test_v4_models_forward_and_warmstart(self) -> None:
        obs = [_make_synthetic_obs(self.geometry, self.bounds, acoplanar=(i % 2 == 1), with_tracker=True) for i in range(4)]
        b1_items, pf_items = [], []
        for idx, ob in enumerate(obs):
            dom = domains.compute_observable_containment(ob, self.geometry, self.bounds)
            b1_items.append(extract_b1pp_item(idx, ob, idx, dom, self.geometry, self.bounds, {}))
            pf_items.append(extract_pf_item(idx, ob, idx, dom, self.geometry, self.bounds, {}))
        b1_batch = collate_b1pp(b1_items)
        pf_batch = collate_pf(pf_items)

        b1_base = ZAlignedStereoNet(self.layout, {'task': 'T4', 'dim': 64, 'n_long_layers': 1, 'heads': 4})
        b1_v4 = ZAlignedStereoNetV4(self.layout, {'task': 'T4', 'dim': 64, 'n_long_layers': 1, 'heads': 4, 'enable_nomad': True, 'enable_pauc': True})
        inc_b = b1_v4.load_state_dict(b1_base.state_dict(), strict=False)
        self.assertEqual(inc_b.unexpected_keys, [])
        self.assertTrue(all(k.startswith(('nomad_', 'pauc_loss.')) for k in inc_b.missing_keys))
        out_b = b1_v4(b1_batch)
        self.assertEqual(tuple(out_b.shape), (4, 4))
        self.assertTrue(torch.isfinite(out_b).all())

        pf_cfg = {'task': 'T4', 'dim': 64, 'heads': 4, 'n_local_layers': 1, 'n_global_layers': 1, 'enable_dense_gn2': False}
        pf_base = PinpointFormerMultitask(self.geometry, pf_cfg)
        pf_v4 = PinpointFormerV4(self.geometry, {**pf_cfg, 'enable_nomad': True, 'enable_pauc': True})
        inc_p = pf_v4.load_state_dict(pf_base.state_dict(), strict=False)
        self.assertEqual(inc_p.unexpected_keys, [])
        self.assertTrue(all(k.startswith(('nomad_', 'pauc_loss.')) for k in inc_p.missing_keys))
        out_p = pf_v4(pf_batch)
        self.assertEqual(tuple(out_p['probs4'].shape), (4, 4))
        self.assertTrue(torch.isfinite(out_p['probs4']).all())

    def test_oof_ensemble_no_fold_leakage_and_simplex(self) -> None:
        rng = np.random.default_rng(19)
        n = 500
        fams = np.arange(n, dtype=np.int64)
        labels = rng.integers(0, 4, size=n, dtype=np.int64)
        obs_dom = rng.choice([0, 1, 2, 3], size=n, p=[0.05, 0.15, 0.55, 0.25])
        ref_df = pd.DataFrame({
            'run': np.full(n, 10000, dtype=np.int64),
            'chunk': np.arange(n, dtype=np.int64),
            'row': np.zeros(n, dtype=np.int64),
            'family_hi': np.zeros(n, dtype=np.int64),
            'family_lo': fams,
            'label': labels,
            'obs_domain_id': obs_dom,
        })

        def _make_member(bias_tau: float, seed: int) -> pd.DataFrame:
            r = np.random.default_rng(seed)
            z = r.normal(0.0, 0.6, size=(n, 4))
            z[np.arange(n), labels] += 1.4
            z[:, 2] += bias_tau
            e = np.exp(z - np.max(z, axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            df = ref_df[['run', 'chunk', 'row', 'family_hi', 'family_lo', 'label']].copy()
            for c in range(4):
                df[f'p_{c}'] = p[:, c].astype(np.float32)
            df['charm_prob'] = r.uniform(0.0, 0.5, size=n).astype(np.float32)
            return df

        mframes = {'m1': _make_member(0.2, 1), 'm2': _make_member(-0.1, 2)}
        slices = {'All': np.ones(n, dtype=bool), 'D_scint_contained': obs_dom == 2}
        folds, _ = grouped_folds(ref_df[['run', 'chunk', 'row', 'family_hi', 'family_lo', 'label']], folds=5, seed=20261117)
        res_clean = fit_oof_domain_ensemble(mframes, ref_df, slices, n_folds=5, folds=folds)
        probs = res_clean['oof_probs']
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.allclose(probs[obs_dom == 0], 0.25))

        # Corrupt labels inside fold 0 only and confirm fold 0 predictions are bitwise invariant
        ref_corrupt = ref_df.copy()
        fold0 = folds == 0
        ref_corrupt.loc[fold0, 'label'] = (labels[fold0] + 1) % 4
        mframes_c = {k: v.copy() for k, v in mframes.items()}
        for df in mframes_c.values():
            df.loc[fold0, 'label'] = ref_corrupt.loc[fold0, 'label']
        res_corrupt = fit_oof_domain_ensemble(mframes_c, ref_corrupt, slices, n_folds=5, folds=folds)
        self.assertTrue(np.allclose(res_clean['oof_probs'][fold0], res_corrupt['oof_probs'][fold0], atol=1e-7))


if __name__ == '__main__':
    unittest.main()
