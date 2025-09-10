# Multi‑Robot Path Planning — Run Guide

This repo contains three related pieces:

1) A PyQt demo app for sequential multi‑robot planning and path quality scoring (`mrpp_selector.py`).
2) A Learn‑to‑Rank (L2R) pipeline to score K‑shortest candidates (`make_dataset.py`, `train_ranker.py`, `ranker_infer.py`).
3) A weight‑regression pipeline that predicts cost weights from map/context features (`make_weight_dataset.py`, `train_weight_regressor.py`, `weight_infer.py`).

Use this README as a step‑by‑step playbook to run everything locally.

## Install

Python 3.9+ recommended.

```
pip install -U numpy pandas scikit-learn joblib xgboost PyQt5
```

If you only need the UI (no ML), `numpy` and `PyQt5` are enough. If you only need training/inference, you can skip `PyQt5`.

## Maps

Grid maps are plain text files with rows of `0` (free) and `1` (obstacle), no spaces. Example (`map.txt`):

```
111111
100001
101101
100001
111111
```

The UI loads a built‑in demo by default. Other scripts construct the same demo grid in‑code unless noted.

---

## A) Learn‑to‑Rank Candidate Scorer

Two options are available:

- Cost regressor (simple): learns to predict total cost; the UI inverts sign so higher is better.
- True ranking model (recommended): learns to score candidates directly; higher is better.

1) Generate dataset

```
python make_dataset.py
```

This writes `l2r_dataset.csv`. You can tune the number of cases inside the script (`cases` variable).

2) Train ranker (cost regressor)

```
python train_ranker.py
```

Outputs `ranker_model.joblib` (XGBoost regressor + feature list). The UI treats predictions as cost and displays `L2R_pred = -pred_cost` so higher is better.

2b) Train ranker (true ranking, LightGBM LambdaRank)

```
python make_ranker_dataset.py --maps_dir ./maps --out_csv ranker_dataset.csv --cases_per_map 600 --K 8
python train_ranker_lgbm.py --dataset ranker_dataset.csv
```

This produces a `ranker_model.joblib` that includes `{"model": ..., "features": [...], "model_type": "ranker"}`.
The UI auto‑detects `model_type == "ranker"` and uses raw model scores (higher is better) in the `L2R_pred` column.

3) Inference example

```python
from mrpp_selector import MapGrid, yen_k_shortest
from ranker_infer import L2RRanker

# small demo grid (same as scripts)
grid = MapGrid([[1 if ch=='1' else 0 for ch in line] for line in [
    "11111111111111111111",
    "10000000000000000001",
    "10111011011011011101",
    "10000000000000000001",
    "11111111111111111111",
]])

start, goal = (1,1), (3,18)
committed_paths = []  # previously planned robots
cands = yen_k_shortest(grid, start, goal, K=8)

ranker = L2RRanker("ranker_model.joblib")
scores = ranker.score_candidates(cands, committed_paths)  # higher is better
best_idx = max(range(len(cands)), key=lambda i: scores[i])
best_path = cands[best_idx]
```

---

## B) Weight Regressor (map/context → cost weights)

This pipeline learns to predict cost weights used by the hand‑crafted scoring function.

1) Generate dataset

```
python make_weight_dataset.py
```

Writes `weight_dataset.csv` with map/context features and the “best” weights (from random search on each case). You can tune `cases`, `K`, and `M` inside the script.

2) Train regressor

```
python train_weight_regressor.py
```

Outputs `weight_model.joblib` and prints per‑weight MAE.

3) Inference example

```python
from mrpp_selector import astar  # A* (now compatible with both grid APIs)
from context_features import MapGrid as CFMap, compute_context_features
from weight_infer import WeightPredictor

grid = CFMap([[1 if ch=='1' else 0 for ch in line] for line in [
    "11111111111111111111",
    "10000000000000000001",
    "10111011011011011101",
    "10000000000000000001",
    "11111111111111111111",
]])
committed, start, goal, candidates = [], (1,1), (3,18), []

features = compute_context_features(
    grid, committed, start, goal, candidates,
    astar_fn=astar  # passes the A* function to estimate shortest path length
)
pred = WeightPredictor("weight_model.joblib").predict(features)
print(pred)  # dict of weights: {"w_len": ..., "w_turn": ..., ...}
```

---

## C) Interactive UI (sequential planning + scoring)

```
python mrpp_selector.py
```

What it does:
- Click to set Start for Robot 1, then End; it plans immediately.
- Repeats for each robot; later plans consider earlier robots’ paths.
- Optionally shows K‑shortest alternatives for the most recent robot.
- Displays a candidate cost table and prints detailed breakdown to the console.
- One‑click ON/OFF per component (length, turns, cell/edge overlaps, h2h, deadlock, self‑cycle):
  - Heuristic scoring: disabled components have weight 0.
  - Learned weights: predicted weights are masked to 0 for disabled components.
  - Learned ranker: disabled features are neutralized within a case so they cannot affect ranking.
  - New: `path_overlap` (same cell at the same timestamp) is detected and can be weighted separately from `cell_overlap` (static cell reuse across time).

Notes:
- Requires a display environment (PyQt5). On headless servers, use a virtual display (e.g., Xvfb) or run locally.
- Load a custom TXT map via the UI button. `1` means obstacle, `0` free.

---

## Tips & Troubleshooting

- Missing packages: Install `xgboost` for the ranker and `scikit-learn`, `pandas`, `joblib` for both pipelines.
- PyQt5 not installed: `pip install PyQt5`.
- Headless environment: the UI won’t open without a display; the training/inference scripts work fine in headless mode.
- Map format: this code expects no spaces in map lines. If you have a space‑separated map, remove spaces or adapt the loader.
- Reproducibility: dataset scripts set a fixed random seed. Increase `cases` for better models.

---

## File Overview

- `mrpp_selector.py`: UI + A*/Yen + scoring.
- `ranker_features.py`: feature computation for L2R.
- `make_dataset.py`: generate `l2r_dataset.csv`.
- `train_ranker.py`: train XGBoost model → `ranker_model.joblib`.
- `ranker_infer.py`: load and score candidates.
- `context_features.py`: map/context feature extraction used by weight pipeline.
- `make_weight_dataset.py`: generate `weight_dataset.csv`.
- `train_weight_regressor.py`: train MLP regressor → `weight_model.joblib`.
- `weight_infer.py`: load and predict weights from features.
