# weight_infer.py
from typing import Dict, List
import numpy as np
from joblib import load

# Must match the ranges in make_weight_dataset.py
RANGES = {
    "w_len": (0.5, 3.0),
    "w_turn": (0.0, 1.0),
    "w_overlap_cell": (0.0, 5.0),
    "w_path_overlap": (0.0, 5.0),
    "w_overlap_edge": (0.0, 3.0),
    "w_h2h": (10.0, 120.0),
    "w_deadlock": (30.0, 200.0),
    "w_self_cycle": (0.0, 20.0),
}
ORDER = ["w_len","w_turn","w_overlap_cell","w_path_overlap","w_overlap_edge","w_h2h","w_deadlock","w_self_cycle"]

class WeightPredictor:
    def __init__(self, joblib_path="weight_model.joblib"):
        pack = load(joblib_path)
        self.model = pack["model"]
        self.features = pack["features"]

    def predict(self, feature_dict: Dict[str,float]) -> Dict[str,float]:
        x = np.array([[feature_dict[k] for k in self.features]], dtype=float)
        y = self.model.predict(x)[0]
        # clip to safe ranges
        out = {}
        for k, v in zip(ORDER, y):
            lo, hi = RANGES[k]
            out[k] = float(min(hi, max(lo, v)))
        return out
