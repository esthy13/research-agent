# Frozen MMLU transfer evaluation

The PubMedQA-selected configuration and both predefined baselines were frozen before evaluation. MMLU results were not used for search, adaptation, or configuration selection.

College Chemistry and College Physics are two subjects from the same MMLU benchmark family, not two independent datasets.

## Accuracy and cost

| Benchmark | Configuration | n | Accuracy (95% CI) | Total tokens | Latency (s) | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mmlu_college_chemistry | naive | 100 | 0.620 [0.522, 0.709] | 49776 | 482.4 | 13 |
| mmlu_college_chemistry | careful | 100 | 0.600 [0.502, 0.691] | 53767 | 505.4 | 14 |
| mmlu_college_chemistry | selected | 100 | 0.640 [0.542, 0.727] | 187855 | 1381.6 | 7 |
| mmlu_college_physics | naive | 91 | 0.857 [0.771, 0.915] | 35049 | 279.4 | 2 |
| mmlu_college_physics | careful | 91 | 0.868 [0.784, 0.923] | 39008 | 305.1 | 4 |
| mmlu_college_physics | selected | 91 | 0.912 [0.836, 0.955] | 113146 | 681.5 | 1 |
| mmlu_science_combined | naive | 191 | 0.733 [0.666, 0.791] | 84825 | 761.8 | 15 |
| mmlu_science_combined | careful | 191 | 0.728 [0.661, 0.786] | 92775 | 810.5 | 18 |
| mmlu_science_combined | selected | 191 | 0.770 [0.705, 0.824] | 301001 | 2063.1 | 8 |

## Paired comparisons

| Benchmark | Baseline | Accuracy difference (pp, 95% CI) | Selected-only correct | Baseline-only correct | Exact McNemar p | Token ratio | Latency ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mmlu_college_chemistry | naive | 2.0 [-6.0, 10.0] | 9 | 7 | 0.8036 | 3.77x | 2.86x |
| mmlu_college_chemistry | careful | 4.0 [-2.0, 10.0] | 7 | 3 | 0.3438 | 3.49x | 2.73x |
| mmlu_college_physics | naive | 5.5 [0.0, 11.0] | 6 | 1 | 0.1250 | 3.23x | 2.44x |
| mmlu_college_physics | careful | 4.4 [1.1, 8.8] | 4 | 0 | 0.1250 | 2.90x | 2.23x |
| mmlu_science_combined | naive | 3.7 [-1.0, 8.4] | 15 | 8 | 0.2100 | 3.55x | 2.71x |
| mmlu_science_combined | careful | 4.2 [0.5, 7.9] | 11 | 3 | 0.0574 | 3.24x | 2.55x |

## Interpretation notes

- The combined row pools item-level outcomes across both MMLU subjects.
- Accuracy intervals are Wilson intervals.
- Difference intervals use a deterministic paired bootstrap with 10,000 resamples.
- McNemar p-values are exact, two-sided, and exploratory; they are not corrected for multiple comparisons.
- A transfer failure does not invalidate the search system; it indicates that the PubMedQA-selected configuration is domain-specific.
