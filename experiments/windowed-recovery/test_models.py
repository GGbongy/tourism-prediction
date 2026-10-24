"""Strict calendar-window, numerical, and missing-data checks."""
import sys
from pathlib import Path
for deps in (Path(__file__).resolve().parent/'.deps',Path(__file__).resolve().parents[1]/'sarima/.deps'):
    if deps.is_dir():
        sys.path.insert(0,str(deps));break
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'_shared'))

import unittest
import numpy as np
from models import CONFIGS,predict


class WindowChecks(unittest.TestCase):
    def test_every_bounded_configuration_ignores_older_history(self):
        original=np.log(.2+np.arange(43)*.01+.02*np.sin(np.arange(43)))
        for config in CONFIGS:
            n=config['window']
            if n is None:continue
            changed=original.copy();changed[:-n]=99
            a,_=predict(original,12,config)
            b,_=predict(changed,12,config)
            np.testing.assert_array_equal(a,b,err_msg=config['candidate_id'])

    def test_missing_window_does_not_backfill_from_old_months(self):
        history=np.r_[np.log(np.full(30,5.)),np.nan,np.nan,np.nan]
        for config in CONFIGS:
            if config['window']!=3:continue
            result,info=predict(history,12,config)
            np.testing.assert_array_equal(result,np.ones(12))
            self.assertEqual(info['valid_observations'],0)
            self.assertTrue(info['fallback'])

    def test_linear_ratio_has_expected_increments_and_damping(self):
        ratios=.2+.01*np.arange(24)
        linear=next(c for c in CONFIGS if c['candidate_id']=='linear_r_n6_p1')
        damped=next(c for c in CONFIGS if c['candidate_id']=='linear_r_n6_p0.8')
        a,_=predict(np.log(ratios),12,linear)
        b,_=predict(np.log(ratios),12,damped)
        np.testing.assert_allclose(a,ratios[-1]+.01*np.arange(1,13),rtol=1e-12)
        self.assertTrue(np.all(b<a))

    def test_missing_month_positions_are_preserved(self):
        ratios=.2+.1*np.arange(6)
        history=np.log(ratios);history[2]=np.nan
        config=next(c for c in CONFIGS if c['candidate_id']=='linear_r_n6_p1')
        result,_=predict(history,3,config)
        np.testing.assert_allclose(result,[.8,.9,1.0],rtol=1e-12)

    def test_log_trend_is_log_linear_not_ratio_linear(self):
        z=-2+.02*np.arange(24)
        config=next(c for c in CONFIGS if c['candidate_id']=='linear_z_n6_p1')
        result,_=predict(z,12,config)
        np.testing.assert_allclose(result,np.exp(z[-1]+.02*np.arange(1,13)),rtol=1e-12)

    def test_legacy_model_does_not_overwrite_configuration_id(self):
        config=next(c for c in CONFIGS if c['candidate_id']=='ets_level_z_n12')
        result,info=predict(np.linspace(-2,-.5,20),12,config)
        self.assertNotIn('candidate_id',info)
        self.assertEqual(info['legacy_model_id'],'ets_level')

    def test_no_artificial_upper_limit_of_one(self):
        config=next(c for c in CONFIGS if c['candidate_id']=='linear_r_n6_p1')
        result,_=predict(np.log(np.linspace(.7,1.2,6)),12,config)
        self.assertGreater(result[-1],1.2)


if __name__=='__main__':unittest.main()
