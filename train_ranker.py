# train_ranker.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from joblib import dump

DF = pd.read_csv("l2r_dataset.csv")
FEATS = ["length","turns","cell_overlap","path_overlap","edge_overlap","h2h","deadlock","self_cycle","n_others","K"]
X = DF[FEATS].values
y = DF["teacher_cost"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
)
model.fit(Xtr, ytr)

# quick eval: MSE and "regret" proxy
from sklearn.metrics import mean_squared_error
print("Test MSE:", mean_squared_error(yte, model.predict(Xte)))

dump({"model": model, "features": FEATS}, "ranker_model.joblib")
print("Saved ranker_model.joblib")
