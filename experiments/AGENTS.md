# Experiment rules

- Never modify `Tourism-Forecasting-Starter.ipynb`.
- Use one model family and one hyperparameter configuration across destinations;
  only fitted coefficients may differ. Do not select models per destination.
- Each method has its own directory. Save raw calculations in `outputs/` and
  structured metrics in `metrics/`, with validation and test in separate files
  and subdirectories. Report metrics in the method README.
- Keep the frozen counterfactual baseline unchanged when comparing recovery
  extrapolation rules. Document its historical per-destination fits and fallbacks.
- `experiments/README.md` contains only validation/test metric comparisons for
  the retained three recovery configurations, fixed SARIMA, lag-1 and lag-12.
- Visualization code belongs in `visual/`. Each HTML contains one figure, with
  the established serif fonts, colors, dataset boundaries, legend visibility,
  zoom and pan. Do not add buttons, toolbars or explanatory page text.
- Maintain the figure index in `visual/README.md`. Read saved predictions for
  visualization rather than refitting models.
- Original MA and MB have been removed at the user's request; do not recreate
  their method folders or figures unless requested.
- Generated CSV results, PNG previews, JSON logs/fitted-state records/manifests,
  and runtime logs are local ignored outputs. Clean them after preparing HTML
  and README reports. Preserve raw input datasets and do not ignore them.
