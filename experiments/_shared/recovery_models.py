"""Counterfactual and univariate recovery primitives. No test-file access."""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.statespace.sarimax import SARIMAX

from point_baselines import date, dates, month

EPSILON = 1.0
BASELINES = ('seasonal_mean_60', 'damped_log_trend_120', 'sarima_011_011')
PRE_CUTOFFS = ('2016M12', '2017M12', '2018M12')


def values_at(history, country, labels):
    return np.array([history.get(d, {}).get(country)
                     if history.get(d, {}).get(country) is not None else np.nan
                     for d in labels], dtype=float)


def fit_state(model, steps):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted = model.fit(method='lbfgs', maxiter=150, disp=False, cov_type='none')
        attempts = [{'optimizer': 'lbfgs', 'converged': bool(fitted.mle_retvals['converged'])}]
        if not attempts[-1]['converged']:
            fitted = model.fit(start_params=fitted.params, method='powell', maxiter=100,
                               disp=False, cov_type='none')
            attempts.append({'optimizer': 'powell', 'converged': bool(fitted.mle_retvals['converged'])})
        prediction = np.asarray(fitted.forecast(steps=steps), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite state-space prediction')
    return prediction, {'converged': bool(fitted.mle_retvals['converged']), 'attempts': attempts,
                         'params': dict(zip(fitted.param_names, map(float, fitted.params))),
                         'warnings': sorted({str(w.message) for w in caught})}


def baseline_predict(history, country, cutoff, targets, candidate):
    if cutoff > '2019M12':
        raise ValueError('Baseline fitting may not use post-2019 data')
    labels = sorted(d for d in history if d <= cutoff and history[d][country] is not None)
    if len(labels) < 24:
        raise ValueError('Insufficient pre-pandemic history')
    first = labels[0]
    complete = dates(first, cutoff)
    values = values_at(history, country, complete)
    horizons = np.array([month(d)-month(cutoff) for d in targets])
    if np.any(horizons <= 0):
        raise ValueError('Baseline targets must follow its fitting origin')
    record = {'candidate_id': candidate, 'destination': country, 'fit_cutoff': cutoff,
              'converged': True, 'warnings': [], 'fallback': ''}
    if candidate == 'seasonal_mean_60':
        start = max(month(first), month(cutoff)-59)
        used = [(d, history[d][country]) for d in labels if month(d) >= start]
        seasonal = [np.mean([v for d, v in used if month(d) % 12 == m])
                    if any(month(d) % 12 == m for d, _ in used) else np.nan for m in range(12)]
        result = np.array([seasonal[month(d) % 12] for d in targets])
        record.update({'fit_start': date(start), 'seasonal_means': list(map(float, seasonal))})
    elif candidate == 'damped_log_trend_120':
        start = max(month(first), month(cutoff)-119)
        used = [(d, history[d][country]) for d in labels if month(d) >= start]
        def design(d, trend):
            return [1.0, trend] + [float(month(d) % 12 == m) for m in range(1, 12)]
        matrix = np.array([design(d, month(d)-month(cutoff)) for d, _ in used])
        coefs = np.linalg.lstsq(matrix, np.log1p([v for _, v in used]), rcond=None)[0]
        damp = .98*(1-.98**horizons)/(1-.98)
        future = np.array([design(d, t) for d, t in zip(targets, damp)])
        result = np.expm1(future @ coefs)
        record.update({'fit_start': date(start), 'damping': .98, 'coefficients': coefs.tolist()})
    elif candidate == 'sarima_011_011':
        factor = max(float(np.nanmean(values)), 1.0)
        model = SARIMAX(values/factor, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                        trend='n', enforce_stationarity=True, enforce_invertibility=True)
        full, diagnostics = fit_state(model, int(horizons.max()))
        result = full[horizons-1]*factor
        record.update({'fit_start': first, 'normalization_factor': factor, **diagnostics})
    else:
        raise ValueError(candidate)
    if not np.isfinite(result).all() or np.any(result <= 1):
        raise ValueError('Invalid counterfactual baseline (nonfinite or <=1 tourist)')
    return result, record


def ets_filter(values, alpha, beta=0., phi=1.):
    observed = np.flatnonzero(np.isfinite(values))
    if not len(observed):
        raise ValueError('No finite recovery values')
    level, trend = float(values[observed[0]]), 0.0
    errors = []
    for index in range(observed[0]+1, len(values)):
        prior = level + phi*trend
        old = level
        if np.isfinite(values[index]):
            errors.append(values[index]-prior)
            level = alpha*values[index]+(1-alpha)*prior
            trend = beta*(level-old)+(1-beta)*phi*trend
        else:
            level, trend = prior, phi*trend
    return level, trend, np.asarray(errors)


def univariate(values, steps, candidate):
    """Conditional additive ETS or Gaussian ARIMA; missing observations stay missing."""
    values = np.asarray(values, dtype=float)
    good = np.flatnonzero(np.isfinite(values))
    if not len(good):
        raise ValueError('No observed recovery history')
    values = values[good[0]:]
    record = {'candidate_id': candidate, 'effective_observations': int(len(good)),
              'converged': True, 'fallback': '', 'warnings': []}
    if len(good) < 12 or np.nanstd(values) < 1e-8:
        record['fallback'] = 'last_observed_log_recovery: fewer than 12 observations or constant series'
        return np.full(steps, values[np.flatnonzero(np.isfinite(values))[-1]]), record
    if candidate in ('ets_level', 'ets_damped'):
        initial = [.3] if candidate == 'ets_level' else [.3, .1, .9]
        bounds = [(.001, .999)] if candidate == 'ets_level' else [(.001, .999), (.001, .999), (.8, .98)]
        def objective(parameters):
            _, _, errors = ets_filter(values, *parameters)
            return float(np.mean(errors**2))
        optimum = minimize(objective, initial, method='L-BFGS-B', bounds=bounds)
        if not optimum.success:
            optimum = minimize(objective, optimum.x, method='Powell', bounds=bounds,
                               options={'maxiter': 200})
        level, trend, _ = ets_filter(values, *optimum.x)
        phi = optimum.x[2] if len(optimum.x) == 3 else 1.0
        damp = np.cumsum(phi**np.arange(1, steps+1))
        prediction = level + trend*damp
        record.update({'converged': bool(optimum.success), 'params': optimum.x.tolist(),
                       'terminal_level': float(level), 'terminal_trend': float(trend),
                       'optimizer_message': str(optimum.message)})
    elif candidate in ('ar1', 'arima011'):
        model = SARIMAX(values, order=(1, 0, 0) if candidate == 'ar1' else (0, 1, 1),
                        trend='c' if candidate == 'ar1' else 'n',
                        enforce_stationarity=True, enforce_invertibility=True)
        try:
            prediction, diagnostics = fit_state(model, steps)
            record.update(diagnostics)
        except Exception as error:
            prediction = np.full(steps, values[np.flatnonzero(np.isfinite(values))[-1]])
            record.update({'converged': False,
                           'warnings': [f'{type(error).__name__}: {error}']})
    else:
        raise ValueError(candidate)
    if not record['converged'] or not np.isfinite(prediction).all():
        # Fixed before looking at outcomes; never silently masquerade as a fitted model.
        record['fallback'] = 'last_observed_log_recovery: optimizer failed or nonfinite forecast'
        prediction = np.full(steps, values[np.flatnonzero(np.isfinite(values))[-1]])
    return np.asarray(prediction), record

