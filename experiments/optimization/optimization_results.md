# Day 3 optimization results

Scores are fractions (0-1). Test is evaluated once after selection.

## Category

| Method | CV/Objective Macro-F1 | Validation Macro-F1 |
| --- | ---: | ---: |
| Grid | 0.467677 | 0.479943 |
| Random | 0.525552 | 0.574854 |
| Bayesian | 0.465354 | 0.502149 |

Selected method: random
Selected parameters: `{"classifier__C": 10.0, "classifier__class_weight": null, "tfidf__max_df": 0.95, "tfidf__max_features": 20000, "tfidf__min_df": 1, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": false}`

| Test metric | Value |
| --- | ---: |
| accuracy | 0.652795 |
| precision_macro | 0.771353 |
| recall_macro | 0.584357 |
| f1_macro | 0.641538 |
| f1_weighted | 0.651417 |

## Priority

| Method | CV/Objective Macro-F1 | Validation Macro-F1 |
| --- | ---: | ---: |
| Grid | 0.563099 | 0.590207 |
| Random | 0.603171 | 0.630936 |
| Bayesian | 0.528075 | 0.526622 |

Selected method: random
Selected parameters: `{"classifier__C": 10.0, "classifier__class_weight": null, "tfidf__max_df": 0.95, "tfidf__max_features": 20000, "tfidf__min_df": 1, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": false}`

| Test metric | Value |
| --- | ---: |
| accuracy | 0.676867 |
| precision_macro | 0.682126 |
| recall_macro | 0.652074 |
| f1_macro | 0.662208 |
| f1_weighted | 0.674342 |

