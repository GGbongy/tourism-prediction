"""Run with: python -m unittest discover -s experiments/sarima -p test_*.py"""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run as sarima


class SarimaChecks(unittest.TestCase):
    def setUp(self):
        self.history = {d: {'A': 100 + 10*math.sin(i*2*math.pi/12) + i*.1}
                        for i, d in enumerate(sarima.dates('2010M01', '2021M12'))}

    def test_prepare_preserves_internal_missing_and_trims_leading(self):
        self.history['2010M01']['A'] = None
        self.history['2010M02']['A'] = None
        self.history['2011M02']['A'] = None
        values, factor, labels = sarima.prepare(self.history, 'A', '2020M12')
        self.assertEqual(labels[0], '2010M03')
        self.assertEqual(labels[-1], '2020M12')
        self.assertTrue(math.isnan(values[labels.index('2011M02')]))
        self.assertGreater(factor, 0)

    def test_forecasts_and_normalization_ignore_future_actuals(self):
        config = sarima.CANDIDATES[0]
        targets = sarima.dates('2021M01', '2021M07')
        expected, fit = sarima.fit_forecast(self.history, 'A', '2020M12', targets, config)
        changed = {d: {'A': 999999.0 if d > '2020M12' else row['A']}
                   for d, row in self.history.items()}
        observed, other_fit = sarima.fit_forecast(changed, 'A', '2020M12', targets, config)
        self.assertEqual(expected, observed)
        self.assertEqual(fit['normalization_factor'], other_fit['normalization_factor'])
        self.assertTrue(all(r['point'] >= 0 and math.isfinite(r['point']) for r in expected))

    def test_selection_rejects_nonconvergence_and_partial_coverage(self):
        rows = [dict(candidate_id='bad', converged=False, mae=0, n_scored=7, complexity=0),
                dict(candidate_id='partial', converged=True, mae=0, n_scored=6, complexity=0),
                dict(candidate_id='ok', converged=True, mae=2, n_scored=7, complexity=2)]
        self.assertEqual(sarima.select_candidate(rows)['candidate_id'], 'ok')
        with self.assertRaises(RuntimeError):
            sarima.select_candidate(rows[:2])

    def test_selection_tie_break_is_deterministic(self):
        rows = [dict(candidate_id='z', converged=True, mae=1, n_scored=7, complexity=3),
                dict(candidate_id='b', converged=True, mae=1, n_scored=7, complexity=2),
                dict(candidate_id='a', converged=True, mae=1, n_scored=7, complexity=2)]
        self.assertEqual(sarima.select_candidate(rows)['candidate_id'], 'a')

    def test_forecast_rejects_gap_after_origin(self):
        with self.assertRaises(ValueError):
            sarima.fit_forecast(self.history, 'A', '2020M12', ['2021M02'], sarima.CANDIDATES[0])

    def test_all_parameter_report_retains_failed_fits_and_missing_actuals(self):
        configs = sarima.CANDIDATES[:2]
        targets = sarima.dates('2023M08', '2024M02')
        predictions = [{'Date': d, 'destination': 'A', 'horizon': i+1,
                        'fit_cutoff': '2023M07', 'candidate_id': configs[0]['id'],
                        'raw_point': 5.0, 'point': 5.0, 'clipped_negative': False}
                       for i, d in enumerate(targets)]
        actuals = {d: {'A': 10.0} for d in targets}
        actuals[targets[0]]['A'] = 0.0
        actuals[targets[1]]['A'] = None
        fits = [{'candidate_id': c['id'], 'destination': 'A', 'converged': i==0,
                 'fit_seconds': 0.0, 'error': '' if i==0 else 'intentional failed fit'}
                for i, c in enumerate(configs)]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sarima, 'HERE', Path(directory)), patch.object(sarima, 'CANDIDATES', configs):
                result = sarima.save_candidate_reports('test', self.history, ['A'], targets,
                    predictions, actuals, fits, {})
                by_destination = result['by_destination']
                self.assertEqual(len(by_destination), 2)
                self.assertEqual(by_destination[0]['n_scored'], 6)
                self.assertEqual(by_destination[0]['mape_valid_n'], 5)
                self.assertEqual(by_destination[1]['n_actual'], 6)
                self.assertEqual(by_destination[1]['n_scored'], 0)
                self.assertIsNone(by_destination[1]['mae'])
                saved = sarima.load_csv(Path(directory)/'outputs/test/candidate_predictions_long.csv')
                self.assertEqual(len(saved), 14)
                self.assertTrue((Path(directory)/'metrics/test/candidate_summary.csv').is_file())
                self.assertFalse((Path(directory)/'metrics/validation').exists())


if __name__ == '__main__':
    unittest.main()
