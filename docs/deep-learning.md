# Deep-learning comparisons

## Scope

Day 4 and Day 5 are category-classification experiments only. They do not train priority models and they are not integrated into the FastAPI endpoint.

All models use the existing English `ticket_text` split with seed 42: 11,436 train, 2,451 validation, and 2,451 test rows. Training fits on train data only. Validation Macro-F1 is the selection metric. Test metrics are final reporting only and are not used to choose a model.

## Model configurations

The CNN, RNN, LSTM, and Attention models use the existing lightweight NumPy text-encoder pattern because TensorFlow is not compatible with this Python 3.14 environment. They share:

- vocabulary size: 5,000
- sequence length: 60
- embedding dimension: 32
- hidden dimension: 32
- classifier head: Logistic Regression with max_iter 200

Attention adds actual token-level attention over embedded token sequences, then projects the attended context before classification.

DistilBERT uses Hugging Face Transformers with `distilbert-base-uncased`. To keep the POC runnable on CPU, the base encoder is frozen and only the classification head is fine-tuned for one epoch. Training uses a deterministic train-only cap of 60 examples per category, batch size 64, and sequence length 48. Validation and test metrics are computed on the full validation/test splits.

## Results

Generated artifact: `experiments/dl_comparison/day5/dl_comparison_results.json`.

| Model | Validation Macro-F1 | Test accuracy | Test Macro-F1 | Test weighted-F1 | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.045488 | 0.289270 | 0.044873 | 0.129988 | 1.710 |
| RNN | 0.044922 | 0.289678 | 0.044922 | 0.130130 | 0.836 |
| LSTM | 0.044922 | 0.289678 | 0.044922 | 0.130130 | 4.436 |
| Attention | 0.044922 | 0.289678 | 0.044922 | 0.130130 | 1.193 |
| DistilBERT | 0.121775 | 0.194206 | 0.120542 | 0.187129 | 57.209 |

Selected Day 5 DL model: DistilBERT, selected based on validation Macro-F1. Final selected-model test Macro-F1 is 0.120542.

The pretrained model completed successfully after downloading the public checkpoint into a local Hugging Face cache. No Hugging Face token or external API was used.

## Artifacts

Day 5 artifacts are under `experiments/dl_comparison/day5/`:

- `dl_comparison_results.json`
- `dl_comparison_results.md`
- `selected_model_metadata.json`

The selected-model confusion matrix is stored in the JSON artifact. The unused DistilBERT checkpoint directory was removed during final repository cleanup because the integrated application does not load the Day 5 DL model; the assessment evidence is preserved in the result JSON, Markdown report, and selected-model metadata.

## Limitations

The neural models perform poorly compared with the Day 3 optimized classical category model, whose final test Macro-F1 is 0.641538. DistilBERT was intentionally resource-capped for the POC and should not be read as a production-quality transformer training recipe. The synthetic dataset, class imbalance, frozen encoder, small fine-tuning subset, and lack of hyperparameter search all limit conclusions.

The existing Day 4 artifacts in `experiments/dl_comparison/` are locked to SYSTEM/Administrators on this machine, so Day 5 artifacts were written to the `day5` subdirectory instead of overwriting those files.
