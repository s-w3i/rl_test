# make_dataset.py
import random, json, csv
from pathlib import Path
from typing import List, Tuple
import numpy as np

from mrpp_selector import MapGrid, yen_k_shortest, astar
from ranker_features import compute_components, compute_cost, DEFAULT_W

Coord = Tuple[int,int]; PathT = List[Coord]

def random_free_cell(grid: MapGrid) -> Coord:
    while True:
        r = random.randrange(grid.rows); c = random.randrange(grid.cols)
        if grid.grid[r][c] == 0:
            return (r,c)

def simulate_committed_paths(grid: MapGrid, n_others: int) -> List[PathT]:
    committed = []
    for _ in range(n_others):
        s = random_free_cell(grid); g = random_free_cell(grid)
        p = astar(grid, s, g)
        if p: committed.append(p)
    return committed

def row_from_case(grid: MapGrid, committed: List[PathT], start: Coord, goal: Coord, K: int):
    cands = yen_k_shortest(grid, start, goal, K)
    feats = [compute_components(p, committed) for p in cands]
    costs = [compute_cost(f, DEFAULT_W) for f in feats]
    if not cands: return []
    # label = index of min cost (teacher)
    y = int(np.argmin(costs))
    rows = []
    for i,(f,c) in enumerate(zip(feats, costs)):
        rows.append({
            "k_index": i, "is_best": int(i==y), "teacher_cost": c,
            "length": f["length"], "turns": f["turns"],
            "cell_overlap": f["cell_overlap"], "path_overlap": f["path_overlap"], "edge_overlap": f["edge_overlap"],
            "h2h": f["h2h"], "deadlock": f["deadlock"], "self_cycle": f["self_cycle"],
            # (optional) tiny context: how many others and K
            "n_others": len(committed), "K": K
        })
    return rows

def main():
    random.seed(0)
    # small demo map like your default
    grid = MapGrid([[1 if ch=='1' else 0 for ch in line] for line in [
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
    ]])

    out = Path("l2r_dataset.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "k_index","is_best","teacher_cost",
            "length","turns","cell_overlap","path_overlap","edge_overlap","h2h","deadlock","self_cycle",
            "n_others","K"
        ])
        writer.writeheader()
        cases = 1500  # increase later
        for _ in range(cases):
            n_others = random.randint(0, 3)   # 0..3 other robots
            committed = simulate_committed_paths(grid, n_others)
            start = random_free_cell(grid); goal = random_free_cell(grid)
            rows = row_from_case(grid, committed, start, goal, K=8)
            for r in rows: writer.writerow(r)
    print("Saved", out)

if __name__ == "__main__":
    main()
