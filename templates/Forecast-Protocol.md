# Assignment 2 forecast protocol template

Submit the completed file as
`A2-Group-NAME-Protocol.md`.

Complete Sections A–C before opening the locked public audit. Complete Section
D after the single audit run. Do not rewrite the frozen selection rule after
seeing audit outcomes; record any deviation separately.

## A. Data and temporal freeze

- Group ID:
- Date frozen:
- Public-data version/hash:
- Training cutoff:
- Validation target months:
- Locked public-audit target months:
- Final hidden forecast months:
- External data, availability dates, and licences, or `none`:
- Leakage controls:
- Location of frozen code/configuration:
- Confirmation that audit outcomes have not yet been examined: `yes / no`

## B. Point forecast selection

- Lag-1 implementation/check:
- Lag-12 implementation/check:
- Candidate improved methods:
- Features and preprocessing:
- Hyperparameter/search budget:
- Primary validation metric:
- Guardrail metrics:
- Model-selection rule:
- Selected point method and validation evidence:
- Seeds, compute, and software environment:

## C. Interval forecast selection

- Residual or seasonal-bootstrap baseline:
- Candidate improved probabilistic methods:
- Uncertainty represented by each candidate:
- 80% and 95% interval construction:
- Calibration data and cutoff controls:
- Primary validation metric:
- Coverage and sharpness guardrails:
- Method-selection rule:
- Selected interval method and validation evidence:

### Optional advanced stochastic/GP investigation

- Used: `yes / no`
- Advanced idea and course connection:
- Simpler probabilistic comparator:
- Expected improvement:
- What is held constant:
- Compute/resource budget:
- Validation outcome and decision to adopt or reject:
- Reused Assignment 1 artefacts and acknowledgement, or `none`:

## D. Locked public-audit result

- Date/run identifier:
- Point metrics and baseline differences:
- Interval metrics and baseline differences:
- Destination/horizon consistency:
- Uncertainty or stability evidence:
- Runtime/resource result:
- Important failures:
- Protocol deviation, if any:
- Final workflow decision and justification:

## E. Exploratory work after audit

List any work performed after viewing the audit. It must not replace the
original confirmatory result.

- Change made:
- Reason:
- Evidence produced:
- How it is labelled in the report:
