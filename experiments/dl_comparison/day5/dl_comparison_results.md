# Day 5 DL comparison

Category-only comparison on the unchanged train/validation/test split.

| Model | Validation Macro-F1 | Test Accuracy | Test Precision Macro | Test Recall Macro | Test Macro-F1 | Test Weighted-F1 | Training time seconds | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CNN | 0.045488 | 0.289270 | 0.028939 | 0.099859 | 0.044873 | 0.129988 | 1.710 | completed |
| RNN | 0.044922 | 0.289678 | 0.028968 | 0.100000 | 0.044922 | 0.130130 | 0.836 | completed |
| LSTM | 0.044922 | 0.289678 | 0.028968 | 0.100000 | 0.044922 | 0.130130 | 4.436 | completed |
| ATTENTION | 0.044922 | 0.289678 | 0.028968 | 0.100000 | 0.044922 | 0.130130 | 1.193 | completed |
| DISTILBERT | 0.121775 | 0.194206 | 0.182285 | 0.162085 | 0.120542 | 0.187129 | 57.209 | completed |

Selected model: distilbert based on validation Macro-F1.
Final selected-model test Macro-F1: 0.120542.
