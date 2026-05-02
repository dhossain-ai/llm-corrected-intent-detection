# robust-intent-detection

Clean baseline project for robust intent detection experiments using CLINC150.

Phase 1 sets up the repository structure and adds a Hugging Face CLINC150 dataset
loader. Training, noise generation, model integrations, evaluation, and UI work
are intentionally deferred.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Phase 1 Loader

```bash
python -m src.data_loader
```

The loader downloads `clinc_oos` from Hugging Face, prefers the `plus`
configuration, falls back to loading without a config, standardizes the
`train`, `validation`, and `test` splits, and prints dataset counts with sample
rows.

## Phase 1 Status

- Project folders are initialized with placeholders where needed.
- CLINC150 loading is implemented in `src/data_loader.py`.
- Shared project paths are defined in `src/config.py`.
- Phase 1 dependencies are listed in `requirements.txt`.
- A small import sanity test is included in `tests/test_imports.py`.

## Phase 2 Noise Generation

Phase 2 adds reproducible character-level synthetic noise for chatbot messages:

- Character deletion
- Character insertion
- Adjacent character swaps
- QWERTY neighbor typos
- Mixed noise controlled by `noise_level`

Generate clean and noisy CLINC150 test files:

```bash
python -m src.noise_generator
```

Run a quick sample export:

```bash
python -m src.noise_generator --sample-size 20
```

Generated CSV files are written to `data/processed/` and ignored by git.

## Phase 3 TF-IDF Baseline

Train the baseline intent classifier:

```bash
python -m src.train_tfidf
```

Generate clean and noisy test data first if the processed CSVs are missing:

```bash
python -m src.noise_generator
```

Evaluate the TF-IDF model:

```bash
python -m src.evaluate --model tfidf
```

Metrics are saved to `outputs/metrics/tfidf_results.csv`, and sample
predictions are saved to `outputs/predictions/tfidf_sample_predictions.csv`.
Generated model and output files are ignored by git.

Accuracy is the share of examples with the correct intent. Macro F1 averages F1
equally across labels, which makes weaker classes visible even when labels are
imbalanced. Robustness score compares noisy accuracy to clean accuracy; clean is
`1.0`, and noisy scores are `noisy_accuracy / clean_accuracy`.

## Planned Next Phases

- LLM correction workflow
- DistilBERT baseline
- Evaluation reports and figures
- Streamlit application
