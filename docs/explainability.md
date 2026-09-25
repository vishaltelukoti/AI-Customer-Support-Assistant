# Explainability and Subgroup Diagnostics

Day 10 explains the saved Day 3 optimized category classifier. The model is the existing `experiments/optimization/models/category_optimized.joblib` TF-IDF + Logistic Regression pipeline. It was not retrained and the Day 3 model-selection process was not changed.

## Method

SHAP was attempted first, but installation failed in the current Python 3.14 Windows environment because SHAP needed Microsoft C++ Build Tools to compile a wheel. The implemented fallback is an equivalent lightweight method for this model family: each active TF-IDF feature contribution is computed as:

```text
tf-idf value * saved Logistic Regression coefficient for the predicted class
```

This uses the actual saved TF-IDF vocabulary and classifier weights. Positive features support the predicted category. Negative features oppose it.

## Representative Examples

Exactly five held-out test examples are selected deterministically by confidence quantiles `[0, 0.25, 0.5, 0.75, 1.0]`, preferring distinct predicted categories where possible. Test examples are used only after model selection for explanation, not for training or tuning.

| Ticket | Actual | Predicted | Confidence |
| --- | --- | --- | ---: |
| KAGGLE-f187c090e595-008850 | Returns and Exchanges | Billing and Payments | 0.174072 |
| KAGGLE-f187c090e595-026513 | IT Support | Technical Support | 0.447547 |
| KAGGLE-f187c090e595-020230 | IT Support | IT Support | 0.593814 |
| KAGGLE-f187c090e595-019967 | Product Support | Product Support | 0.773817 |
| KAGGLE-f187c090e595-022784 | Service Outages and Maintenance | Service Outages and Maintenance | 0.996855 |

All five explanations succeeded. The confidence range is 0.174072 to 0.996855.

Confidence is the saved Logistic Regression `predict_proba` value for the predicted class. These probabilities are model probabilities and are not calibrated correctness probabilities because no calibration step was performed.

## Subgroup Diagnostic

The classification dataset is English-only, so it does not support a valid English-vs-German fairness comparison for this model. No race, gender, age, religion, disability or other sensitive attributes are inferred.

The diagnostic uses available non-sensitive metadata from `historical_tickets.csv`: `version` and `type`. Groups below 50 test examples would be excluded; none were excluded in the current test split.

### Version

| Group | Count | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 400 | 1576 | 0.668147 | 0.790176 | 0.613518 | 0.671903 |
| 51 | 75 | 0.600000 | 0.561508 | 0.518085 | 0.532016 |
| 52 | 800 | 0.627500 | 0.738261 | 0.536478 | 0.583456 |

Largest observed version differences: accuracy 0.068147, macro precision 0.228668, macro recall 0.095433, macro F1 0.139887.

### Type

| Group | Count | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Change | 250 | 0.648000 | 0.750249 | 0.619427 | 0.653210 |
| Incident | 956 | 0.686192 | 0.799067 | 0.566267 | 0.628003 |
| Problem | 537 | 0.685289 | 0.818117 | 0.614570 | 0.664408 |
| Request | 708 | 0.584746 | 0.746909 | 0.546530 | 0.610727 |

Largest observed type differences: accuracy 0.101447, macro precision 0.071207, macro recall 0.072897, macro F1 0.053681.

## Limitations

This is a subgroup performance/distribution diagnostic, not proof that the model is fair or biased. The groups are synthetic dataset metadata, not protected demographic groups. The data is English-only and synthetic, label quality is not independently adjudicated, and model probabilities are uncalibrated.

Possible next mitigations include collecting more representative labeled data, stratified evaluation over meaningful operational groups, monitoring subgroup performance after integration, reviewing class imbalance, and calibration or threshold analysis where appropriate.
