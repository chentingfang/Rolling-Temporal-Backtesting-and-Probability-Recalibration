"""Numerical and failure-path checks: python -m unittest -v test_reproduction.py"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import metrics as m
from reproduce import read_and_validate, MODELS


class NumericalTests(unittest.TestCase):
    def test_auc_against_pair_enumeration(self):
        y=np.array([1,0,1,0,0]); p=np.array([.8,.8,.3,.2,.3])
        pairs=[float(a>b)+.5*float(a==b) for a in p[y==1] for b in p[y==0]]
        self.assertAlmostEqual(m.discrimination(y,p)[0],np.mean(pairs),places=14)

    def test_grouped_ap_and_permutation_invariance(self):
        y=np.array([1,0,1]); p=np.array([.9,.9,.1])
        self.assertAlmostEqual(m.discrimination(y,p)[1],7/12)
        order=np.array([1,0,2])
        self.assertEqual(m.discrimination(y,p),m.discrimination(y[order],p[order]))
        self.assertAlmostEqual(m.discrimination(np.array([1,0,0]),np.ones(3))[1],1/3)

    def test_offset_solves_known_constant_probability_case(self):
        y=np.array([1]+[0]*9); p=np.full(10,.2)
        fit=m.fit_offset(y,p)
        expected=np.log(.1/.9)-np.log(.2/.8)
        self.assertTrue(fit['converged'])
        self.assertAlmostEqual(fit['alpha'],expected,places=9)
        self.assertLess(abs(m.sigmoid(m.logit(p)+fit['alpha']).mean()-.1),1e-12)

    def test_single_class_offset_fails_explicitly(self):
        fit=m.fit_offset(np.zeros(4),np.full(4,.1))
        self.assertFalse(fit['converged']); self.assertTrue(np.isnan(fit['alpha']))
        self.assertTrue(np.isnan(m.discrimination(np.zeros(4),np.arange(4))[0]))

    def test_log_boundaries_finite_and_brier_unclipped(self):
        y=np.array([1,0]); p=np.array([0.,1.])
        self.assertEqual(m.scores(y,p)['brier'],1.)
        self.assertTrue(np.isfinite(m.scores(y,p)['log_loss']))
        self.assertTrue(np.isfinite(m.logit(p)).all())

    def test_capacity_ceiling_and_identifier_tie_rule(self):
        y=np.array([0,1,0]); p=np.array([.9,.9,.1]); ids=np.array(['B','A','C'])
        result,idx=m.capacity(y,p,ids,.01)
        self.assertEqual(result['k'],1); self.assertEqual(ids[idx][0],'A')
        self.assertEqual(result['recall'],1.)

    def test_clipping_can_change_topk_and_must_not_be_assumed_invariant(self):
        ids=np.array(['A','B']); p=np.array([1e-10,2e-10])
        updated=m.sigmoid(m.logit(p)-1)
        self.assertFalse(np.array_equal(m.ranked_indices(p,ids),m.ranked_indices(updated,ids)))

    def test_no_predicted_events_precision_is_undefined(self):
        result=m.classification(np.array([0,1]),np.array([0,0]))
        self.assertTrue(np.isnan(result['precision']))
        self.assertEqual(result['recall'],0.)

    def test_reliability_bins_keep_all_equal_scores(self):
        bins=m.reliability_bins(np.array([0,0,1]),np.full(3,.1))
        self.assertEqual(len(bins),1); self.assertEqual(bins[0]['n'],3)
        self.assertEqual(bins[0]['events'],1)


class ValidationTests(unittest.TestCase):
    @staticmethod
    def frame():
        rows=[]
        for h in [1,2]:
            for t in range(2020,2025):
                for i in [0,1]:
                    r=dict(firm_id=f'F{i}',year=t,feature_year=t-h,horizon=h,y=i)
                    for slug in MODELS.values():
                        r.update({f'p_{slug}':.8 if i else .1,f'threshold_{slug}':.5,f'yhat_{slug}':i})
                    rows.append(r)
        return pd.DataFrame(rows)

    def test_duplicate_record_is_rejected_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frame=self.frame();frame=pd.concat([frame,frame.iloc[[0]]])
            frame.to_csv(root/'input.csv',index=False)
            with self.assertRaises(ValueError):read_and_validate(root/'input.csv',root)
            errors=pd.read_csv(root/'validation_errors.csv')
            self.assertIn('duplicate_key',errors.reason.tolist())

    def test_bad_probability_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame=self.frame();frame.loc[0,'p_rf']=1.5
            frame.to_csv(root/'input.csv',index=False)
            with self.assertRaises(ValueError):read_and_validate(root/'input.csv',root)

    def test_valid_input_and_identifier_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.frame().iloc[::-1].to_csv(root/'input.csv',index=False)
            d,report=read_and_validate(root/'input.csv',root)
            self.assertEqual(report['records'],20); self.assertEqual(d.iloc[0].firm_id,'F0')


if __name__=='__main__':
    unittest.main()
