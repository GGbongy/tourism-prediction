"""Simple rolling-window recovery models. No file I/O or target access."""
from __future__ import annotations

import math
import numpy as np
from recovery_models import univariate

WINDOWS = (3, 6, 9, 12, 18, None)


def candidates():
    result = [{'candidate_id': 'last_r_n1', 'family': 'last_r', 'window': 1, 'phi': 1.}]
    for n in WINDOWS:
        suffix = 'all' if n is None else str(n)
        for family, phis in [('mean_r', (1.,)), ('median_r', (1.,)), ('weighted_mean_r', (1.,)),
                             ('linear_r', (.8, .95, 1.)), ('median_slope_r', (.95, 1.)),
                             ('linear_z', (.95, 1.))]:
            for phi in phis:
                result.append({'candidate_id': f'{family}_n{suffix}_p{phi:g}',
                               'family': family, 'window': n, 'phi': phi})
    for n in (12, 18, None):
        for family in ('ets_level', 'ets_damped', 'ar1', 'arima011'):
            result.append({'candidate_id': f'{family}_z_n{n if n else "all"}',
                           'family': family+'_z', 'window': n, 'phi': None})
    return result


CONFIGS = candidates()


def predict(z_history, steps, config):
    """Only the last n calendar positions may affect the forecast or fallback."""
    values = np.asarray(z_history, dtype=float)
    if config['window'] is not None:
        values = values[-config['window']:]
    positions = np.arange(1-len(values), 1, dtype=float)
    valid = np.isfinite(values)
    record = {'window_months_used': len(values), 'valid_observations': int(valid.sum()),
              'fallback': '', 'params': {}, 'converged': True}
    if not valid.any():
        record['fallback'] = 'No observations inside window: recovery index 1'
        return np.ones(steps), record
    z, x = values[valid], positions[valid]
    r = np.exp(z)
    family, phi = config['family'], config['phi']
    if family.endswith('_z') and family != 'linear_z':
        prediction, info = univariate(values, steps, family[:-2])
        info = dict(info)
        info['legacy_model_id'] = info.pop('candidate_id', family[:-2])
        record.update(info)
        output = np.exp(prediction)
    elif family in ('last_r', 'mean_r', 'median_r', 'weighted_mean_r'):
        if family == 'last_r':
            level = r[-1]
        elif family == 'mean_r':
            level = r.mean()
        elif family == 'median_r':
            level = np.median(r)
        else:
            weights = 2.**(x/3.)  # Fixed calendar half-life = 3 months.
            level = np.average(r, weights=weights)
        record['params'] = {'terminal_level': float(level)}
        output = np.full(steps, level)
    elif len(z) < 3:
        record['fallback'] = 'Fewer than three observations inside window: last recovery level'
        output = np.full(steps, r[-1])
    else:
        if family == 'median_slope_r':
            # Differences use actual month distances, including gaps.
            slope = float(np.median(np.diff(r)/np.diff(x)))
            level = float(r[-1]-slope*x[-1])
        else:
            target = z if family == 'linear_z' else r
            level, slope = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), target, rcond=None)[0]
        increments = np.cumsum(float(phi)**np.arange(1, steps+1))
        forecast = level+slope*increments
        with np.errstate(over='ignore', invalid='ignore'):
            output = np.exp(forecast) if family == 'linear_z' else np.maximum(0, forecast)
        record['params'] = {'terminal_level': float(level), 'slope': float(slope),
                            'phi': float(phi), 'space': 'z' if family=='linear_z' else 'r'}
    if not np.isfinite(output).all():
        record['fallback'] = 'Nonfinite extrapolation: last observed recovery level inside window'
        record['converged'] = False
        output = np.full(steps, r[-1])
    return np.maximum(0, output), record
