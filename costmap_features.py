from __future__ import annotations
from typing import List, Tuple

import numpy as np


def build_costmap_inputs(grid: 'MapGrid', committed_paths: List['Path'], goal: 'Coord') -> np.ndarray:
    rows, cols = grid.rows, grid.cols
    obstacles = np.zeros((rows, cols), dtype=np.float32)
    occupancy = np.zeros((rows, cols), dtype=np.float32)
    goal_dist = np.zeros((rows, cols), dtype=np.float32)
    narrow = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            obstacles[r, c] = 1.0 if grid.grid[r][c] == 1 else 0.0
            # narrowness
            if grid.grid[r][c] == 0:
                deg = 0
                for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if 0 <= rr < rows and 0 <= cc < cols and grid.grid[rr][cc] == 0:
                        deg += 1
                narrow[r, c] = 1.0 if deg <= 2 else 0.0

    # occupancy from committed paths
    for p in committed_paths:
        for (r, c) in p:
            if 0 <= r < rows and 0 <= c < cols:
                occupancy[r, c] += 1.0
    if occupancy.max() > occupancy.min():
        occ_min, occ_max = float(occupancy.min()), float(occupancy.max())
        occupancy = (occupancy - occ_min) / (occ_max - occ_min)
    else:
        occupancy.fill(0.0)

    # goal proximity (closer = 1)
    gr, gc = goal
    norm = float(rows + cols) if (rows + cols) > 0 else 1.0
    for r in range(rows):
        for c in range(cols):
            d = abs(r - gr) + abs(c - gc)
            goal_dist[r, c] = 1.0 - min(1.0, d / norm)

    inp = np.stack([obstacles, occupancy, goal_dist, narrow], axis=0).astype(np.float32)
    # add batch dim
    inp = inp[None, ...]  # (1,4,H,W)
    return inp
