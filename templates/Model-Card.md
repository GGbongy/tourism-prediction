# Assignment 2 model card template

Submit as `A2-Group-NAME-Model-Card.md`.

## Purpose of this file

The model card is a concise handoff record for the final forecasting system that produced `Forecast.csv` and `Intervals.csv`. It lets the teaching team identify the exact selected methods, data cutoff, dependencies, reproduction requirements, intended use, and known limitations without reconstructing those facts from the notebook.

The three A2 documents have different roles:

- the **Protocol** records the evaluation and model-selection rules fixed before the locked audit;
- the **Report** presents the comparative evidence and argues for the final choice; and
- the **Model Card** describes the frozen final system, how to reproduce it, and where it should or should not be trusted.

Keep this file brief and factual. Refer to the Report or Notebook for detailed analysis instead of repeating it.

## Identification

- Group ID:
- System name/version:
- Date:
- Notebook version or repository commit:
- `model_label` used in `Intervals.csv`:

## Final system

- Selected point model:
- Selected interval method:
- Core dataset and public-history cutoff:
- External data, or `none`:
- Essential preprocessing and leakage controls:
- Random seeds or nondeterministic components:

## Intended use

- Intended tourism stakeholder and decision:
- Forecast horizon and destinations:
- Appropriate use:
- Non-intended or unsafe use:

## Reproduction requirements

- Software and major dependency versions:
- Hardware or hosted service:
- Ordered command or notebook steps:
- Expected generated files:
- Approximate runtime, API calls, and cost:
- Anything that cannot be reproduced and why:

## Known limitations

- Important failed destinations, horizons, or conditions:
- Calibration or coverage limitations:
- Distribution-shift and pandemic/recovery limitations:
- Causal claims that must not be made:
- Privacy, licensing, security, or external-service risks:

## AI and external tools

| Tool/model and access month | Purpose | Material affected | Verification performed |
| --- | --- | --- | --- |
|  |  |  |  |
