# make_ranker_dataset.py
import csv, random, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# Use your existing implementations
from mrpp_selector import (
    MapGrid, yen_k_shortest, cost_components, ScoreWeights, astar
)

Coord = Tuple[int, int]
PathT = List[Coord]

# Must match L2RRanker.features in mrpp_selector.py
FEATS = [
    "length","turns","cell_overlap","path_overlap","edge_overlap",
    "h2h","deadlock","self_cycle","n_others","K"
]

def candidate_features(path: PathT, committed: List[PathT]) -> Dict[str, float]:
    comp, _ = cost_components(path, committed, ScoreWeights())  # weights unused for comps
    return {
        "length": float(comp["length"]),
        "turns": float(comp["turns"]),
        "cell_overlap": float(comp["cell_overlap"]),
        "path_overlap": float(comp["path_overlap"]),
        "edge_overlap": float(comp["edge_overlap"]),
        "h2h": float(comp["h2h"]),
        "deadlock": float(comp["deadlock"]),
        "self_cycle": float(comp["self_cycle"]),
    }

def random_free_cell(grid: MapGrid) -> Coord:
    # assume there is at least one free cell
    while True:
        r = random.randrange(grid.rows); c = random.randrange(grid.cols)
        if grid.grid[r][c] == 0:
            return (r, c)

def simulate_committed_paths(grid: MapGrid, n_others: int) -> List[PathT]:
    res = []
    tries = 0
    while len(res) < n_others and tries < 400:
        tries += 1
        s = random_free_cell(grid); g = random_free_cell(grid)
        if s == g: 
            continue
        p = astar(grid, s, g)
        if p:
            res.append(p)
    return res

def relevance_from_costs(costs: List[float]) -> List[float]:
    # Higher relevance for lower cost, normalized within the case
    mx = max(costs); mn = min(costs)
    if mx - mn < 1e-9:
        return [1.0 for _ in costs]
    return [(mx - c) / (mx - mn) for c in costs]

def build_from_maps(
    maps_dir: Path,
    out_csv: Path,
    cases_per_map: int = 600,
    K: int = 8,
    seed: int = 0,
    min_viable: int = 2,
):
    random.seed(seed)
    map_files = sorted(p for p in maps_dir.glob("*.txt") if p.is_file())
    if not map_files:
        raise SystemExit(f"No .txt map files found in: {maps_dir}")

    label_w = ScoreWeights(  # used to compute labels for training
        w_len=1.0, w_turn=0.25, w_overlap_cell=2.0, w_overlap_edge=1.0,
        w_path_overlap=2.0,
        w_h2h=20.0, w_deadlock=50.0, w_self_cycle=5.0
    )

    rows = []
    case_id = 0

    for mpath in map_files:
        try:
            grid = MapGrid.from_txt(str(mpath))
        except Exception as e:
            print(f"[skip] Failed to load {mpath.name}: {e}")
            continue

        free_cells = sum(1 for r in range(grid.rows) for c in range(grid.cols) if grid.grid[r][c] == 0)
        if free_cells < 4:
            print(f"[skip] Map {mpath.name} has too few free cells.")
            continue

        print(f"[map] {mpath.name}: size=({grid.rows}x{grid.cols}), free={free_cells}")

        produced = 0
        attempts = 0
        max_attempts = cases_per_map * 10  # to avoid infinite loops on hard maps
        while produced < cases_per_map and attempts < max_attempts:
            attempts += 1
            # Random committed paths (other robots)
            n_committed = random.randint(0, 4)
            committed = simulate_committed_paths(grid, n_committed)

            # Random start/end for the current robot
            s = random_free_cell(grid); g = random_free_cell(grid)
            if s == g:
                continue

            cands = yen_k_shortest(grid, s, g, K)
            if len(cands) < min_viable:
                continue

            # Label each candidate by heuristic total cost under label_w
            costs = [cost_components(p, committed, label_w)[1] for p in cands]
            rels = relevance_from_costs(costs)

            # Emit K rows for this case
            for p, rel in zip(cands, rels):
                f = candidate_features(p, committed)
                f["n_others"] = float(len(committed))
                f["K"] = float(len(cands))
                row = {
                    "case_id": case_id,
                    "map_file": mpath.name,
                    "start_r": s[0], "start_c": s[1],
                    "goal_r": g[0],  "goal_c": g[1],
                    "relevance": rel,
                }
                row.update({k: f[k] for k in FEATS})
                rows.append(row)

            case_id += 1
            produced += 1

        print(f"  -> cases generated: {produced} (attempts {attempts})")

    if not rows:
        raise SystemExit("No dataset rows generated. Check your maps and parameters.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["case_id","map_file","start_r","start_c","goal_r","goal_c","relevance"] + FEATS)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    print(f"[done] Saved {out_csv} with {case_id} cases and {len(rows)} rows across {len(map_files)} map(s).")

def main():
    ap = argparse.ArgumentParser(description="Build grouped ranking dataset from TXT maps.")
    ap.add_argument("--maps_dir", type=Path, required=True, help="Folder containing .txt map files")
    ap.add_argument("--out_csv", type=Path, default=Path("ranker_dataset.csv"))
    ap.add_argument("--cases_per_map", type=int, default=600)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build_from_maps(args.maps_dir, args.out_csv, args.cases_per_map, args.K, args.seed)

if __name__ == "__main__":
    main()
