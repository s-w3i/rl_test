# make_weight_dataset.py
import random, csv
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

from mrpp_selector import MapGrid as MRPPMap, astar, yen_k_shortest, cost_components, ScoreWeights
from context_features import MapGrid as CFMap, compute_context_features

Coord = Tuple[int,int]
PathT = List[Coord]

# ---- weight ranges (keep in sync with training/inference) ----
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
ORDER = list(RANGES.keys())

def sample_weights() -> ScoreWeights:
    vals = {k: random.uniform(*RANGES[k]) for k in ORDER}
    return ScoreWeights(**vals)

def random_free_cell(grid: MRPPMap) -> Coord:
    while True:
        r = random.randrange(grid.rows); c = random.randrange(grid.cols)
        if grid.grid[r][c] == 0: return (r,c)

def simulate_committed_paths(grid: MRPPMap, n_others: int) -> List[PathT]:
    res = []
    tries = 0
    while len(res) < n_others and tries < 200:
        tries += 1
        s = random_free_cell(grid); g = random_free_cell(grid)
        p = astar(grid, s, g)
        if p: res.append(p)
    return res

def objective_J(path: PathT, others: List[PathT], weights: ScoreWeights) -> float:
    # Use the SAME components you compute for planning, but a slightly
    # different mixing for label stability (optional).
    comp, base_cost = cost_components(path, others, weights)
    # Normalize a couple of terms to keep scales reasonable:
    Lnorm = comp["length"]
    cellOv = comp["cell_overlap"] / max(1, comp["length"])
    pathOv = comp["path_overlap"] / max(1, comp["length"])
    # Penalize conflicts heavier than length:
    return (0.5 * Lnorm) + (0.5 * comp["turns"]) + 3.0 * cellOv + 4.0 * pathOv + 6.0 * comp["h2h"] + 10.0 * comp["deadlock"]

def best_weights_for_case(grid: MRPPMap, committed: List[PathT], start: Coord, goal: Coord, K: int, M: int):
    cands = yen_k_shortest(grid, start, goal, K)
    if not cands: return None, None
    best_val = float("inf")
    best_w = None
    for _ in range(M):
        w = sample_weights()
        # pick candidate with min cost under w
        costs = [cost_components(p, committed, w)[1] for p in cands]
        p_best = cands[int(np.argmin(costs))]
        val = objective_J(p_best, committed, w)
        if val < best_val:
            best_val = val
            best_w = w
    return cands, best_w

def demo_map() -> MRPPMap:
    txt = [
         "11111111111111111111",
        "10000000000000000001",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10000000000000000001",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10111011011011011101",
        "10000000000000000001",
        "11111111111111111111"
    ]
    grid = [[1 if ch=='1' else 0 for ch in line] for line in txt]
    return MRPPMap(grid)

def main():
    random.seed(0)
    grid = demo_map()
    out = Path("weight_dataset.csv")
    feats_order = [
        "free_ratio","largest_component_ratio","avg_degree","narrow_count",
        "n_committed","occupied_cell_ratio","max_cell_occupancy","total_cell_occupancy",
        "manhattan_start_goal","shortest_len_est","k_len_mean","k_len_std","overlap_with_shortest"
    ]
    header = feats_order + ORDER
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        cases = 1000   # start small; scale up later
        K = 8
        M = 64         # weight samples per case
        for _ in range(cases):
            n_others = random.randint(0, 3)
            committed = simulate_committed_paths(grid, n_others)
            s = random_free_cell(grid); g = random_free_cell(grid)
            cands = yen_k_shortest(grid, s, g, K)
            if not cands: continue
            # compute context features (pass A* to get shortest_len_est & overlap)
            cf = compute_context_features(
                CFMap(grid.grid), committed, s, g, cands,
                astar_fn=astar
            )
            _, wbest = best_weights_for_case(grid, committed, s, g, K, M)
            if wbest is None: continue
            row = {k: cf[k] for k in feats_order}
            for k in ORDER:
                row[k] = getattr(wbest, k)
            wr.writerow(row)
    print("Saved", out)

if __name__ == "__main__":
    main()
