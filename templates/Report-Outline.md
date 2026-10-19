# Assignment 2 report outline

Submit the completed report as `A2-Group-NAME-Report.pdf`. The main report is
limited to 15 pages, excluding references and a clearly labelled
reproducibility appendix. Use this outline as guidance; concise variations are
acceptable when they preserve the required evidence.

## 1. Executive recommendation

- State the selected point and interval methods.
- Identify the intended tourism stakeholder or decision.
- Summarise the strongest validation and locked-audit evidence.
- State the most important limitation or failure.

## 2. Data and temporal protocol

- Identify the public dataset, version, licence, and cutoff.
- Show the training, validation, locked-audit, and final forecast periods.
- Explain missing-data, preprocessing, external-data, and leakage controls.
- Refer to the submitted Protocol for the frozen selection rules.

## 3. Methods and reproducibility

- Describe lag-1, lag-12, and the improved point method.
- Describe the interval baseline and improved probabilistic method.
- Record features, tuning budget, seeds, software, hardware, runtime, and cost.
- Explain how the notebook regenerates both submitted CSV files.

## 4. Validation results and selection

- Compare point methods with MAE, MASE, and MAPE.
- Compare interval methods with PICP, width, interval score or WIS, and ACE.
- Include destination and horizon evidence plus important failures.
- Explain the frozen method-selection decision before the locked audit.

## 5. Locked public-audit results

- Report the single locked-audit result separately from validation.
- Compare the frozen workflow with both point baselines and the interval
  baseline.
- Discuss stability, distribution shift, failed destinations or horizons, and
  any protocol deviation.
- Label all work performed after viewing the audit as exploratory.

## 6. Final workflow and limitations

- Justify the final point and interval methods from the evidence.
- Explain calibration and sharpness together.
- State intended use, non-intended use, uncertainty, and resource trade-offs.
- Avoid causal claims unless a defensible causal design supports them.

## References and reproducibility appendix

- Cite datasets, external code, models, services, literature, and AI tools.
- Put detailed environment, commands, supplementary tables, and rerun notes in
  the appendix rather than the main narrative.
