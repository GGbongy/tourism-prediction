"""Rebuild the historical frozen counterfactual stage from pre-2020 data only."""
from point_baselines import dates,date,month,score,scales,aggregate,write_csv
from recovery_models import BASELINES,PRE_CUTOFFS,baseline_predict
from experiment_io import json_write

def baseline_bundle(history, countries, directory):
    scored_rows, fits, metrics, selection = [], [], [], []
    for country in countries:
        choices = []
        for candidate in BASELINES:
            local = []
            valid = True
            for cutoff in PRE_CUTOFFS:
                targets = dates(date(month(cutoff)+1), date(month(cutoff)+12))
                try:
                    predictions, fit = baseline_predict(history, country, cutoff, targets, candidate)
                    if not fit['converged']:
                        valid = False
                    rows = [{'Date': d, 'destination': country, 'horizon': h+1,
                             'fit_cutoff': cutoff, 'candidate_id': candidate, 'point': float(v)}
                            for h, (d, v) in enumerate(zip(targets, predictions))]
                except Exception as error:
                    valid = False
                    fit = {'destination': country, 'candidate_id': candidate, 'fit_cutoff': cutoff,
                           'converged': False, 'error': f'{type(error).__name__}: {error}'}
                    rows = [{'Date': d, 'destination': country, 'horizon': h+1,
                             'fit_cutoff': cutoff, 'candidate_id': candidate, 'point': None}
                            for h, d in enumerate(targets)]
                fits.append(fit)
                scored = score(rows, history, scales(history, countries, cutoff))
                local.extend(scored)
                metrics.append({'destination': country, 'candidate_id': candidate,
                                'fold_cutoff': cutoff, 'converged': fit['converged'], **aggregate(scored)})
            metric = {'destination': country, 'candidate_id': candidate,
                      'converged_all_folds': valid, **aggregate(local)}
            choices.append(metric)
            scored_rows.extend(local)
        eligible = [r for r in choices if r['converged_all_folds'] and r['n_scored'] >= 24]
        if not eligible:
            raise RuntimeError(f'No valid pre-pandemic baseline for {country}')
        chosen = min(eligible, key=lambda r: (r['mae'], BASELINES.index(r['candidate_id'])))
        selection.append(chosen)
        print(f'Baseline {len(selection)}/20: {country}: {chosen["candidate_id"]}', flush=True)
    baseline, final_fits, baseline_rows = {}, [], []
    targets = dates('2020M01', '2024M07')
    for chosen in selection:
        country, requested = chosen['destination'], chosen['candidate_id']
        try:
            values, fit = baseline_predict(history, country, '2019M12', targets, requested)
            if not fit['converged']:
                raise ValueError('Selected baseline failed to converge')
        except Exception as error:
            values, fit = baseline_predict(history, country, '2019M12', targets, 'seasonal_mean_60')
            fit['fallback'] = f'Fixed seasonal-mean fallback after invalid extrapolation: {error}'
        fit['requested_candidate_id'] = requested
        final_fits.append(fit)
        for d, value in zip(targets, values):
            baseline[(country, d)] = float(value)
            baseline_rows.append({'Date': d, 'destination': country, 'baseline': float(value),
                                  'baseline_candidate_id': fit['candidate_id'],
                                  'fit_cutoff': '2019M12'})
    write_csv(directory/'outputs/baseline/baseline_long.csv', baseline_rows)
    write_csv(directory/'outputs/baseline/cv_scored_predictions.csv', scored_rows)
    write_csv(directory/'metrics/baseline_cv/fold_by_destination.csv', metrics)
    write_csv(directory/'metrics/baseline_cv/selected_baselines.csv', selection)
    json_write(directory/'outputs/baseline/cv_fits.json', fits)
    json_write(directory/'outputs/baseline/final_fits.json', final_fits)
    return baseline

