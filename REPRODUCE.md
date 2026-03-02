# Reproduce This Project

## 1) Environment

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run Training + Evaluation + API

```powershell
python -m csao.main
```

The API starts on port `8000` unless occupied, then it falls back to `8001`.

## 3) Health Check

```powershell
curl.exe http://localhost:8000/health
```

If `8000` is busy, use `8001`.

## 4) UI

Open:

- `http://localhost:8000/ui`

The UI flow is:

- Choose existing `user_id`
- Choose `restaurant_id`
- Add menu item(s) or combos to cart
- Recommendations update live via `/recommend` and show top-2 only

## 5) Minimal-Input Recommendation API (for UI integration)

Use `/recommend-lite` so UI sends only minimal context and backend assembles full model features.

```powershell
curl.exe -X POST http://localhost:8000/recommend-lite `
  -H "Content-Type: application/json" `
  -d "{\"user_id\":\"u_123\",\"restaurant_id\":\"r_42\",\"cart_item_ids\":[\"item_1\",\"item_7\"],\"top_k\":5}"
```

Optional fields supported:

- `city`
- `hour`
- `meal_slot`
- `weather_temp_c`
- `step`
- `candidate_item_ids` (explicit candidate pool override)
- `max_candidates` (default `80`)

Response:

- `recommended_item_ids`
- `candidate_count`
- `cold_start_user` (true if user not found in reference history)

## 6) Data Included

This package includes split parquet files under `csao_data/`:

- `cart_sessions_train.parquet`
- `cart_sessions_val.parquet`
- `cart_sessions_test.parquet`

So runs are reproducible without the original raw parquet.
