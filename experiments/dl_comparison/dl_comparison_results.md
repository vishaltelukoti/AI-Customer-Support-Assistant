# Day 4 DL comparison

Category-only CNN/RNN/LSTM comparison on the unchanged train/validation/test split.

| Model | Validation Macro-F1 | Test Accuracy | Test Precision Macro | Test Recall Macro | Test Macro-F1 | Training time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.045488 | 0.289270 | 0.028939 | 0.099859 | 0.044873 | 1.787 |
| RNN | 0.044922 | 0.289678 | 0.028968 | 0.100000 | 0.044922 | 0.855 |
| LSTM | 0.044922 | 0.289678 | 0.028968 | 0.100000 | 0.044922 | 2.122 |

Selected model: cnn based on validation Macro-F1.
Final selected-model test Macro-F1: 0.044873.
