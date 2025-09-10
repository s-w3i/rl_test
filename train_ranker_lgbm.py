# train_ranker_lgbm.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from joblib import dump
import lightgbm as lgb

FEATS = [
    "length","turns","cell_overlap","path_overlap","edge_overlap",
    "h2h","deadlock","self_cycle","n_others","K"
]

def load_dataset(path):
    df = pd.read_csv(path)
    # ensure groups have at least 2 items
    grp_sizes = df.groupby("case_id").size()
    keep_ids = grp_sizes[grp_sizes >= 2].index
    df = df[df.case_id.isin(keep_ids)].copy()
    return df

def split_groups(df, test_size=0.2, random_state=42):
    groups = df["case_id"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (tr_idx, te_idx) = next(splitter.split(df, groups=groups))
    return df.iloc[tr_idx], df.iloc[te_idx]

def group_sizes_for_lgb(df):
    sizes = df.groupby("case_id").size()
    df_sorted = df.sort_values("case_id").copy()
    groups = [int(sizes.loc[cid]) for cid in df_sorted["case_id"].drop_duplicates().tolist()]
    return df_sorted, groups

def to_int_labels(y: np.ndarray, groups: list[int]) -> np.ndarray:
    """Convert per-group float relevances into integer relevance levels.
    For NDCG@1 training, label the best item(s) in each group as 1 and others 0.
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=int)
    start = 0
    for g in groups:
        end = start + g
        grp = y[start:end]
        m = np.max(grp)
        out[start:end] = (grp >= m - 1e-12).astype(int)
        start = end
    return out

def ndcg_at_1(y_true, y_pred, groups):
    scores = []
    start = 0
    for g in groups:
        end = start + g
        yt = y_true[start:end]
        yp = y_pred[start:end]
        scores.append(1.0 if int(np.argmax(yp)) == int(np.argmax(yt)) else 0.0)
        start = end
    return float(np.mean(scores))

def main():
    ap = argparse.ArgumentParser(description="Train LightGBM LambdaRank on grouped dataset.")
    ap.add_argument("--dataset", type=str, default="ranker_dataset.csv")
    args = ap.parse_args()

    df = load_dataset(args.dataset)
    df_tr, df_te = split_groups(df)
    df_tr, g_tr = group_sizes_for_lgb(df_tr)
    df_te, g_te = group_sizes_for_lgb(df_te)

    X_tr = df_tr[FEATS].values
    y_tr = df_tr["relevance"].values
    X_te = df_te[FEATS].values
    y_te = df_te["relevance"].values
    # LightGBM LambdaRank requires integer labels; binarize per group for NDCG@1
    y_tr = to_int_labels(y_tr, g_tr)
    y_te = to_int_labels(y_te, g_te)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1],
        label_gain=[0, 1],
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=2000,
        min_data_in_leaf=50,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr, y_tr,
        group=g_tr,
        eval_set=[(X_te, y_te)],
        eval_group=[g_te],
        eval_at=[1],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100),
        ],
    )

    yhat = model.predict(X_te, num_iteration=model.best_iteration_)
    print("NDCG@1:", ndcg_at_1(y_te, yhat, g_te))

    dump({"model": model, "features": FEATS, "model_type": "ranker"}, "ranker_model.joblib")
    print("Saved ranker_model.joblib")

if __name__ == "__main__":
    main()
