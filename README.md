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

## Planned Next Phases

- Noise generation
- Baseline model training
- LLM correction workflow
- Evaluation reports and figures
- Streamlit application
