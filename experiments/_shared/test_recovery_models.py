"""Checks for retained scalar recovery and counterfactual primitives."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sarima/.deps'))
import unittest
from unittest.mock import patch
import numpy as np
import recovery_models as models
from point_baselines import dates

class RecoveryChecks(unittest.TestCase):
    def test_counterfactual_ignores_post_2019_observations(self):
        history={d:{'A':1000+20*np.sin(i*2*np.pi/12)} for i,d in enumerate(dates('2010M01','2021M12'))}
        changed={d:{'A':999999. if d>'2019M12' else r['A']} for d,r in history.items()}
        for model in models.BASELINES:
            first,_=models.baseline_predict(history,'A','2019M12',dates('2020M01','2021M12'),model)
            second,_=models.baseline_predict(changed,'A','2019M12',dates('2020M01','2021M12'),model)
            np.testing.assert_array_equal(first,second)
    def test_counterfactual_rejects_future_fitting_cutoff(self):
        with self.assertRaises(ValueError):models.baseline_predict({},'A','2020M01',['2020M02'],models.BASELINES[0])
    def test_scalar_missing_and_constant_fallback(self):
        prediction,record=models.univariate(np.r_[np.nan,np.full(12,-1.),np.nan],5,'ets_level')
        np.testing.assert_array_equal(prediction,np.full(5,-1.))
        self.assertTrue(record['fallback'])
    def test_scalar_optimizer_exception_is_explicit(self):
        values=np.sin(np.arange(20))
        with patch.object(models,'fit_state',side_effect=ValueError('intentional')):
            prediction,record=models.univariate(values,3,'ar1')
        np.testing.assert_array_equal(prediction,np.full(3,values[-1]))
        self.assertFalse(record['converged'])
        self.assertTrue(record['fallback'])

if __name__=='__main__':unittest.main()