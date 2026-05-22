metrics_generator
   |
   +--> 1. process_experiment(experiment_config, predictions)
   |         - receives one experiment configuration and model predictions
   |         - validates prediction structure and label consistency
   |         - initializes metric computation pipeline
   |
   +--> 2. build_confusion_statistics()
   |         - computes per-class TP, FP, FN, and support counts
   |         - derives global prediction statistics
   |         - constructs class-level statistical summaries
   |
   +--> 3. compute_metrics()
   |         - computes global accuracy
   |         - computes per-class sensitivity (recall)
   |         - computes false negative rates
   |         - associates all metrics with class support sizes
   |
   +--> 4. estimate_confidence_intervals()
   |         - computes 95% confidence intervals for all proportion metrics
   |         - adjusts interval width according to sample size
   |         - supports statistically robust interval estimation methods
   |
   +--> 5. build_result_tables()
             - formats experiment metrics into publication-ready rows
             - generates classification performance summaries
             - generates clinical false negative summaries


| Model | Global Acc. | 95% CI | MEL Sens. | MEL CI | MEL n | NV Sens. | NV CI | NV n | ... |
|-------|--------------|---------|------------|---------|-------|-----------|--------|------|-----|
| MLP   | 81.0%        | 79–83   | 74.0%      | 69–78   | 420   | 89.0%     | 88–90  | 3800 | ... |


| Model | MEL FN | MEL FN Rate | MEL CI | MEL n | BCC FN | BCC FN Rate | BCC CI | ... |
|-------|---------|--------------|---------|-------|---------|---------------|---------|-----|
| MLP   | 122     | 29.0%        | 25–34   | 420   | ...     | ...           | ...     | ... |