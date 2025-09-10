# context_features.py
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque

Coord = Tuple[int,int]
Path = List[Coord]

class MapGrid:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    def in_bounds(self, r,c): return 0 <= r < self.rows and 0 <= c < self.cols
    def is_free(self, r,c): return self.grid[r][c] == 0
    def nbrs4(self, r,c):
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r+dr, c+dc
            if self.in_bounds(rr,cc) and self.is_free(rr,cc):
                yield rr,cc

def _free_cells(grid: MapGrid):
    return [(r,c) for r in range(grid.rows) for c in range(grid.cols) if grid.is_free(r,c)]

def _largest_component_ratio(grid: MapGrid) -> float:
    seen = [[False]*grid.cols for _ in range(grid.rows)]
    best = 0
    free = 0
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.is_free(r,c):
                free += 1
                if not seen[r][c]:
                    q = deque([(r,c)])
                    seen[r][c] = True
                    sz = 0
                    while q:
                        rr,cc = q.popleft()
                        sz += 1
                        for nr,nc in grid.nbrs4(rr,cc):
                            if not seen[nr][nc]:
                                seen[nr][nc] = True
                                q.append((nr,nc))
                    best = max(best, sz)
    return best / max(1, free)

def _avg_degree_and_narrow(grid: MapGrid):
    free = _free_cells(grid)
    degs = []
    narrow = 0
    for r,c in free:
        d = sum(1 for _ in grid.nbrs4(r,c))
        degs.append(d)
        if d <= 2:
            narrow += 1
    avg_deg = sum(degs)/max(1,len(degs)) if degs else 0.0
    return avg_deg, narrow

def _committed_occupancy(committed: List[Path]):
    occ = defaultdict(int)
    for p in committed:
        for rc in p:
            occ[rc] += 1
    if not occ: return 0,0,0
    unique = len(occ)
    mx = max(occ.values()) if occ else 0
    total = sum(occ.values())
    return unique, mx, total

def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def _path_len(p: Path) -> int:
    return max(0, len(p)-1)

def _shortest_est_len(grid: MapGrid, start: Coord, goal: Coord, astar_fn) -> int:
    p = astar_fn(grid, start, goal)
    return _path_len(p) if p else 0

def _overlap_with_path(target: Path, committed: List[Path]) -> int:
    S = set(target)
    cnt = 0
    for p in committed:
        for rc in p:
            if rc in S: cnt += 1
    return cnt

def compute_context_features(
    grid: MapGrid,
    committed_paths: List[Path],
    start: Coord,
    goal: Coord,
    candidates: Optional[List[Path]] = None,
    astar_fn=None
) -> Dict[str,float]:
    total = grid.rows * grid.cols
    free_cells = sum(1 for r in range(grid.rows) for c in range(grid.cols) if grid.is_free(r,c))
    free_ratio = free_cells / max(1,total)

    largest_comp = _largest_component_ratio(grid)
    avg_deg, narrow_cnt = _avg_degree_and_narrow(grid)

    n_committed = len(committed_paths)
    uniq_occ, max_occ, total_occ = _committed_occupancy(committed_paths)
    occupied_ratio = uniq_occ / max(1, free_cells)

    manh = _manhattan(start, goal)
    shortest_est = 0
    if astar_fn is not None:
        shortest_est = _shortest_est_len(grid, start, goal, astar_fn)

    k_mean = k_std = overlap_shortest = 0.0
    if candidates:
        lens = [_path_len(p) for p in candidates]
        if lens:
            k_mean = sum(lens)/len(lens)
            mu = k_mean
            k_std = (sum((x-mu)*(x-mu) for x in lens)/len(lens))**0.5
        if astar_fn is not None and shortest_est > 0:
            # get a shortest path once
            sp = astar_fn(grid, start, goal)
            if sp:
                overlap_shortest = _overlap_with_path(sp, committed_paths)

    return {
        "free_ratio": free_ratio,
        "largest_component_ratio": largest_comp,
        "avg_degree": avg_deg,
        "narrow_count": float(narrow_cnt),
        "n_committed": float(n_committed),
        "occupied_cell_ratio": occupied_ratio,
        "max_cell_occupancy": float(max_occ),
        "total_cell_occupancy": float(total_occ),
        "manhattan_start_goal": float(manh),
        "shortest_len_est": float(shortest_est),
        "k_len_mean": float(k_mean),
        "k_len_std": float(k_std),
        "overlap_with_shortest": float(overlap_shortest),
    }
