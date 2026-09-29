# Deep-learning comparisons

## Scope

Day 4 and Day 5 are category-classification experiments only; they do not train a DL priority model. Neither the Day 5 DistilBERT selection nor the other DL experiment models are loaded by the FastAPI endpoint. The live application currently uses the Day 3 optimized classical models for both category and priority.

All models use the same English `ticket_text` split with seed 42: 11,436 train, 2,451 validation, and 2,451 test rows. CNN/RNN/LSTM/Attention fit their Logistic Regression heads using training data; their seeded embedding and encoder weights are fixed. DistilBERT trains its task head on a deterministic 600-example subset drawn only from training data. The selection expression uses validation Macro-F1 only. The current runner also calculates and records test metrics for each model before computing the validation-based selection; test values do not enter that selection.

## Model configurations

The CNN, RNN, LSTM, and Attention implementations use a lightweight NumPy sequence-feature pattern because TensorFlow was not compatible with the experiment's Python 3.14 environment. Token embeddings and each encoder's seeded weights are fixed; the only fitted parameters are in a separate Logistic Regression classification head. These are fixed neural-style feature extractors, not end-to-end trained neural networks. They share:

- vocabulary size: 5,000
- sequence length: 60
- embedding dimension: 32
- hidden dimension: 32
- classifier head: Logistic Regression with max_iter 200

Attention computes token scores from a fixed seeded query vector, applies a padding mask and softmax, pools the token embeddings, and applies a fixed seeded projection before classification. This is the implemented attention mechanism, not a Transformer encoder.

DistilBERT uses the pretrained Hugging Face checkpoint `distilbert-base-uncased`. To keep the POC runnable on CPU, its base encoder is frozen and only the 10-class sequence-classification head is fine-tuned for one epoch. Training uses a deterministic train-only cap of 60 examples per category (600 examples total), batch size 64, and sequence length 48. Validation and test metrics use the full held-out splits.

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

## Selection And Application Boundary

The Day 5 artifact selects DistilBERT by validation Macro-F1 (`0.121775`); its recorded test Macro-F1 is `0.120542`. The test metric is not the selection criterion. This selection applies only to the DL comparison. The live application classifier remains the Day 3 optimized TF-IDF + Logistic Regression model for category and priority. The Day 5 DistilBERT artifact is not loaded in the application.

The pretrained model completed successfully after downloading the public checkpoint into a local Hugging Face cache. No Hugging Face token or external API was used.

## Artifacts

Day 5 artifacts are under `experiments/dl_comparison/day5/`:

- `dl_comparison_results.json`
- `dl_comparison_results.md`
- `selected_model_metadata.json`

The selected-model confusion matrix is stored in the JSON artifact. The result JSON, Markdown report, and selected-model metadata are under `experiments/dl_comparison/day5/`. The selected-model metadata points to `models/distilbert`; this experiment artifact is not used by the live classifier.

## Limitations

The DL experiment models perform below the Day 3 optimized classical category model, whose recorded test Macro-F1 is 0.641538. DistilBERT was intentionally resource-capped for the POC: the base encoder is frozen and its head is trained for one epoch on 600 training examples. The synthetic dataset, class imbalance, fixed random feature extractors for CNN/RNN/LSTM/Attention, frozen DistilBERT encoder, small fine-tuning subset, and lack of hyperparameter search limit conclusions. DistilBERT integration into the live API remains an outstanding assessment gap.
