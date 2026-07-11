# Churn Prioritization Engine

This project implements a small, auditable churn modeling pipeline with explicit data validation, a single train/test split, encoded features fit on train data only, and versioned model artifacts.

## Structure

- `config/config.yaml` stores all tunable parameters.
- `data/raw/` is immutable input data.
- `src/data/` handles loading, cleaning, and splitting.
- `src/features/` handles feature encoding.
- `src/models/` trains and evaluates the model.
- `models/v1/` stores versioned artifacts and metadata.

## Run the pipeline

```bash
python run_pipeline.py
```

## Tests

```bash
pytest -q
```
