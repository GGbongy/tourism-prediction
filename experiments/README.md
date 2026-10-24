# Algorithm experiments

Use this directory for development scripts, candidate configurations, and local
validation results. The submission notebook is
`../Tourism-Forecasting-Starter.ipynb`; it must run independently of this directory.

## Method layout and current runs

Each method has its own code, generated README metric report, `outputs/` for
raw calculations, and `metrics/` for structured metric tables. Validation and
test results must always be written to separate files in separate subdirectories
(`outputs/validation/`, `outputs/test/`, `metrics/validation/`, `metrics/test/`).
Update the method README when regenerating results. Visualizations should read
these saved results rather than recomputing the method.

| Directory | Contents | Run from repository root |
| --- | --- | --- |
| [lag-1](lag-1/README.md) | Recursive last-observation point baseline; MAE, MASE, MAPE. | `python experiments/lag-1/run.py` |
| [lag-12](lag-12/README.md) | Seasonal naive point baseline; MAE, MASE, MAPE. | `python experiments/lag-12/run.py` |
| [visual](visual/README.md) | HTML figures, visualization code, and figure index. | See its README. |
| `_shared/` | Shared baseline implementation and temporal-leakage/metric checks. | `python -m unittest discover -s experiments/_shared -p "test_*.py"` |
| `data/` | Public-history CSV cache shared by method experiments. | Use `--history PATH` to select a different source. |

Every method reports both its own valid samples and the common valid samples of
lag-1 and lag-12. Common-support metric files stay inside each evaluation
partition under `metrics/<partition>/common_support/`.

## Evaluation protocol

- Fit candidate models and preprocessing using observations through `2022M07`.
- Compare candidates using validation targets `2022M08`–`2023M02`.
- Freeze model selection and interval calibration rules before inspecting the
  locked public audit (`2023M03`–`2023M07`). Record the freeze using
  `../templates/Forecast-Protocol.md`.
- Run the locked audit once. Label subsequent investigations as exploratory.
- Refit the frozen workflow using observations through `2023M07` to forecast
  `2023M08`–`2024M07`.

## User-defined local exploratory experiments

The repository owner permits independent scripts in this directory to load
`../testset.csv` for local algorithm testing, comparison, and exploratory tuning.
This permission is a local workflow rule, not an official course amendment.
Record every use and whether it influenced model, feature, hyperparameter, or
interval selection. Results influenced by these targets are exploratory and
must not be described as an untouched final evaluation.

The submission notebook must not read, reference, or depend on `testset.csv`,
embed its actual target values, or obtain those values indirectly through caches
or derived files. It must generate forecasts independently. These runtime
restrictions do not undo leakage from prior tuning on final-period targets.

Compare lag-1 and lag-12 baselines with improved candidates. Record the data
cutoff, configuration, seed, runtime, point metrics (MAE, MASE, MAPE), and interval
metrics (PICP, width, interval score or WIS, absolute coverage error) for each run.
Report results by destination and forecast horizon.

Include all code and permitted inputs needed to reproduce the selected method
in the submission notebook. Disclose any final-set influence on its selection;
use the original course protocol for claims of compliant validation.
