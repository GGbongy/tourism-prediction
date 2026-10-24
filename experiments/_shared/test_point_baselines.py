"""Checks for temporal leakage, recursive horizons, missingness and metrics."""
import unittest
from point_baselines import dates, forecast, score, scales, aggregate


class BaselineChecks(unittest.TestCase):
    def setUp(self):
        self.history = {d: {'A': float(i + 1)}
                        for i, d in enumerate(dates('2019M01', '2022M12'))}

    def test_future_actuals_cannot_change_multistep_forecast(self):
        targets = dates('2021M01', '2022M12')
        for lag in (1, 12):
            expected = forecast(self.history, ['A'], '2020M12', targets, lag)
            altered = {d: {'A': -999999.0 if d > '2020M12' else row['A']}
                       for d, row in self.history.items()}
            self.assertEqual(expected, forecast(altered, ['A'], '2020M12', targets, lag))
            self.assertTrue(all(r['observed_source_month'] <= '2020M12' for r in expected))
            for r in expected:
                if lag == 1:
                    self.assertEqual(r['point'], self.history['2020M12']['A'])
                else:
                    self.assertEqual(r['point'], self.history[r['observed_source_month']]['A'])
            self.assertEqual(expected[0]['point'], expected[12]['point'])

    def test_scaling_excludes_future(self):
        expected = scales(self.history, ['A'], '2020M12')
        altered = {d: {'A': 999999 if d > '2020M12' else row['A']}
                   for d, row in self.history.items()}
        self.assertEqual(expected, scales(altered, ['A'], '2020M12'))
        self.assertEqual(expected['A']['mase_denominator'], 1)

    def test_zero_missing_and_exact_metrics(self):
        predictions = [{'Date': d, 'destination': 'A', 'point': p}
                       for d, p in [('2021M01', 2), ('2021M02', 3),
                                    ('2021M03', 5), ('2021M04', None)]]
        actuals = {'2021M01': {'A': 0}, '2021M02': {'A': 6},
                   '2021M03': {'A': None}, '2021M04': {'A': 10}}
        result = aggregate(score(predictions, actuals, {'A': {'mase_denominator': 2}}))
        self.assertEqual(result['n_actual'], 3)
        self.assertEqual(result['n_scored'], 2)
        self.assertEqual(result['mae'], 2.5)
        self.assertEqual(result['mase'], 1.25)
        self.assertEqual(result['mape_valid_n'], 1)
        self.assertEqual(result['mape_percent'], 50)

    def test_missing_seasonal_source_remains_missing(self):
        self.history['2020M01']['A'] = None
        result = forecast(self.history, ['A'], '2020M12', dates('2021M01', '2021M02'), 12)
        self.assertIsNone(result[0]['point'])
        self.assertIsNotNone(result[1]['point'])

    def test_constant_history_has_undefined_mase(self):
        history = {d: {'A': 5.0} for d in dates('2020M01', '2020M12')}
        scale = scales(history, ['A'], '2020M12')
        result = score([{'Date': '2021M01', 'destination': 'A', 'point': 5}],
                       {'2021M01': {'A': 6}}, scale)
        self.assertIsNone(result[0]['scaled_absolute_error'])


if __name__ == '__main__':
    unittest.main()
