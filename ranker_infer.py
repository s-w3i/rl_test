# ranker_infer.py
from typing import List, Tuple, Dict
import numpy as np
from joblib import load
from ranker_features import compute_components

class L2RRanker:
    """Lightweight inference wrapper for offline scoring.

    Supports both:
    - True ranking models (e.g., LightGBM/XGBoost rankers): model_type == "ranker".
      Returns raw model scores where higher is better.
    - Cost regressors (e.g., XGBRegressor): no model_type or anything else.
      Returns negative predicted cost so higher is better.
    """
    def __init__(self, model_path="ranker_model.joblib"):
        pack = load(model_path)
        if isinstance(pack, dict):
            self.model = pack.get("model")
            self.feats = pack.get("features", [])
            self.model_type = (pack.get("model_type") or "").lower()
        else:
            self.model = pack
            self.feats = []
            self.model_type = ""

    def score_candidates(self, candidates, committed_paths) -> List[float]:
        # Return scores where higher is better.
        rows = []
        for cand in candidates:
            comp = compute_components(cand, committed_paths)
            row = {
                "length": comp["length"], "turns": comp["turns"],
                "cell_overlap": comp["cell_overlap"], "edge_overlap": comp["edge_overlap"],
                "h2h": comp["h2h"], "deadlock": comp["deadlock"], "self_cycle": comp["self_cycle"],
                "n_others": len(committed_paths), "K": len(candidates),
            }
            # Feature order comes from the artifact
            rows.append([row.get(k, 0.0) for k in self.feats])
        X = np.array(rows, dtype=float)
        pred = self.model.predict(X)
        # If it's a true ranker, predictions are already scores
        if self.model_type == "ranker":
            return list(map(float, pred))
        # Otherwise treat predictions as cost and return negative cost as a score
        return (-(pred)).tolist()
