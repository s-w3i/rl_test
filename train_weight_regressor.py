# train_weight_regressor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump

# Keep in sync with dataset script
FEATS = [
    "free_ratio","largest_component_ratio","avg_degree","narrow_count",
    "n_committed","occupied_cell_ratio","max_cell_occupancy","total_cell_occupancy",
    "manhattan_start_goal","shortest_len_est","k_len_mean","k_len_std","overlap_with_shortest"
]
TARGETS = ["w_len","w_turn","w_overlap_cell","w_path_overlap","w_overlap_edge","w_h2h","w_deadlock","w_self_cycle"]

df = pd.read_csv("weight_dataset.csv")
X = df[FEATS].values.astype(np.float32)
Y = df[TARGETS].values.astype(np.float32)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)

# MLP with standardization; multi-output wrapper
base = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64,64,64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
        verbose=False
    ))
])
model = MultiOutputRegressor(base)
model.fit(Xtr, Ytr)

# quick report
pred = model.predict(Xte)
mae = np.mean(np.abs(pred - Yte), axis=0)
print("Per-weight MAE:", dict(zip(TARGETS, mae)))
dump({"model": model, "features": FEATS, "targets": TARGETS}, "weight_model.joblib")
print("Saved weight_model.joblib")
