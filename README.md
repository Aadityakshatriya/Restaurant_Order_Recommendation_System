# Restaurant Order Recommendation System (CSAO)

Hackathon submission repository for a contextual cart add-on recommendation system with:
- Main ranking model (LightGBM LambdaRank)
- FastAPI backend
- Zomato-style interactive UI
- Included train/val/test splits and trained main model artifact

## What Is Included

- `csao/`: training, inference, serving, and evaluation code
- `csao_data/`: split parquet files (`train`, `val`, `test`)
- `csao_models/`: trained model artifacts and metrics summary
- `csao/serving/static/`: production UI
- `zomato_csao_datagen_2.ipynb`: original data-generation notebook (kept as-is)

## Quick Start (Local)

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start API + UI with the pretrained main model (no retraining):

```bash
python -m csao.serve_pretrained
```

3. Open UI:

- `http://localhost:7860/ui`

If you prefer training + eval + serving in one command:

```bash
python -m csao.main
```

## Main API Endpoints

- `GET /health`
- `GET /ui`
- `GET /ui/options`
- `GET /ui/restaurants/{restaurant_id}/menu`
- `POST /recommend-main` (single-call main-model path for low latency)
- `POST /recommend` (rank prebuilt candidate rows)
- `POST /recommend-lite` (backward-compatible alias)

## Notes For Evaluation

- Existing users only in the current UI flow.
- Business-model training is disabled; main model is the active production model.
- Top-2 recommendations are shown in UI after cart updates.
- First cold request can be slower due to in-memory warmup; steady-state requests are faster.

## Deploy to Hugging Face Space (Optional)

Use the helper script:

```bash
./deploy_hf_space.sh <owner/space-name> [public|private]
```

Prereqs:
- `hf auth login` with write token
- `git-lfs` installed

## Reproducibility

See:
- `REPRODUCE.md`
