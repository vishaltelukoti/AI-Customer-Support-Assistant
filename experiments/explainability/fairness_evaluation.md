# Day 10 Subgroup Diagnostic

This evaluates the saved Day 3 category model on the held-out test split using only available non-sensitive metadata.

Minimum group size: 50

## version

| Group | Count | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 400 | 1576 | 0.668147 | 0.790176 | 0.613518 | 0.671903 |
| 51 | 75 | 0.600000 | 0.561508 | 0.518085 | 0.532016 |
| 52 | 800 | 0.627500 | 0.738261 | 0.536478 | 0.583456 |

Largest observed metric differences:
- accuracy: 0.068147 (51 to 400)
- precision_macro: 0.228668 (51 to 400)
- recall_macro: 0.095433 (51 to 400)
- f1_macro: 0.139887 (51 to 400)

## type

| Group | Count | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Change | 250 | 0.648000 | 0.750249 | 0.619427 | 0.653210 |
| Incident | 956 | 0.686192 | 0.799067 | 0.566267 | 0.628003 |
| Problem | 537 | 0.685289 | 0.818117 | 0.614570 | 0.664408 |
| Request | 708 | 0.584746 | 0.746909 | 0.546530 | 0.610727 |

Largest observed metric differences:
- accuracy: 0.101447 (Request to Incident)
- precision_macro: 0.071207 (Request to Problem)
- recall_macro: 0.072897 (Request to Change)
- f1_macro: 0.053681 (Request to Problem)

## Limitations

- This is not proof that the model is fair or biased.
- The classification data is English-only, so no valid English-vs-German fairness comparison is available.
- No sensitive demographic attributes are inferred.
- Version and ticket type may reflect synthetic data-generation patterns rather than real user populations.
