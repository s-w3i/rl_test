#!/usr/bin/env python3
"""
Multi‑Robot Path Quality Selector (PyQt Demo) — Sequential per‑robot planning
-----------------------------------------------------------------------------
Environment: grid-based map loaded from TXT ("1"=obstacle, "0"=free)
Flow: For each robot — select Start → select End → plan immediately.
Each plan considers paths already committed by earlier robots.

What’s included
---------------
• Whole‑path conflict scoring (ROS2‑style): head‑to‑head (swap & pass), wait‑for
  deadlocks (≥3 robots), path overlap, and self‑cycles. Detection runs on the
  **entire path** (no window).
• Planned path drawn as a **polyline only**; optional dashed K‑alternatives are
  shown only for the **most recently planned robot**.
• **Candidate cost table** (right panel) + **console breakdown** for every
  candidate: see each term and the final ∑cost.
• **Learning‑to‑Rank (L2R)** scorer (optional): toggle in UI between
  Heuristic / Learned / Blend 50‑50. A small model predicts cost; if the model
  file isn’t present, it gracefully falls back to the heuristic.

Run:  python3 mrpp_selector.py
Deps: PyQt5 (pip install PyQt5)
Optional for L2R: numpy, joblib, xgboost (only needed if you use Learned/Blend)
"""
from __future__ import annotations
import sys
import heapq
from dataclasses import dataclass
import math
from typing import List, Tuple, Optional, Dict, Set, Union

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover
    raise SystemExit("PyQt5 is required. Install with: pip install PyQt5") from e

# Optional weight predictor and context features (used for Learned weights)
try:  # pragma: no cover - optional
    from weight_infer import WeightPredictor  # type: ignore
except Exception:  # pragma: no cover
    WeightPredictor = None  # type: ignore
try:  # pragma: no cover - optional
    from context_features import MapGrid as CFMap, compute_context_features  # type: ignore
except Exception:  # pragma: no cover
    CFMap = None  # type: ignore
    compute_context_features = None  # type: ignore
try:  # pragma: no cover - optional
    from costmap_utils import build_analytic_costmap, CostMapEngine  # type: ignore
except Exception:  # pragma: no cover
    build_analytic_costmap = None  # type: ignore
    CostMapEngine = None  # type: ignore

# ===============
# Core Data Types
# ===============
Coord = Tuple[int, int]  # (row, col)
Path  = List[Coord]

@dataclass
class MapGrid:
    grid: List[List[int]]  # 1=obstacle, 0=free

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def in_bounds(self, rc: Coord) -> bool:
        r, c = rc
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, rc: Coord) -> bool:
        r, c = rc
        return self.grid[r][c] == 0

    def neighbors4(self, rc: Coord) -> List[Coord]:
        # If a custom adjacency map is present (e.g., from YAML lanes), use it.
        adj = getattr(self, '_adjacency', None)
        if isinstance(adj, dict):
            return list(adj.get(rc, []))
        # Default to 4-connected in-bounds neighbors (occupancy filtered in A*)
        r, c = rc
        cand = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [p for p in cand if self.in_bounds(p)]

    @staticmethod
    def from_txt(path: str) -> "MapGrid":
        grid: List[List[int]] = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [1 if ch == '1' else 0 for ch in line]
                grid.append(row)
        if not grid:
            raise ValueError("Empty map file")
        w = max(len(r) for r in grid)
        grid = [r + [1]*(w-len(r)) for r in grid]  # pad ragged lines with obstacles
        return MapGrid(grid)

    @staticmethod
    def from_yaml(path: str,
                  obstacle_labels: Optional[Set[str]] = None,
                  radius: int = 0,
                  max_dim: int = 80) -> "MapGrid":
        """Load a YAML graph-like file and mark vertices with certain labels
        as obstacles on a generated grid. Coordinates are scaled/quantized to
        fit within max_dim for display.

        Interpreting vertices
        - Each entry under any `vertices:` list corresponds to a single grid cell.
        - Recognized formats:
          • RMF-style: [x, y, {meta...}]  (x first, y second)
          • Dict with coords: {x, y} or pose/point with x/y

        Label detection looks for any string field equal to a target label, or
        common fields like 'label', 'name', 'type', 'category'.
        """
        obstacle_labels = obstacle_labels or {"pickup_dispenser", "dropoff_ingestor"}
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required to load YAML maps. pip install pyyaml") from e

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Collect candidate (x,y,label_match)
        # Collect full vertex list with obstacle flag
        points: List[Tuple[float, float, bool]] = []  # (y, x, is_obstacle)

        def extract_xy(obj) -> Optional[Tuple[float, float]]:
            if not isinstance(obj, dict):
                return None
            if 'x' in obj and 'y' in obj and isinstance(obj['x'], (int, float)) and isinstance(obj['y'], (int, float)):
                return float(obj['y']), float(obj['x'])  # return as (row(y), col(x))
            for key in ('pose', 'point', 'pos', 'position'):
                p = obj.get(key)
                if isinstance(p, dict) and 'x' in p and 'y' in p:
                    return float(p['y']), float(p['x'])
            return None

        def is_obstacle_vertex(obj) -> bool:
            if not isinstance(obj, dict):
                return False
            # If any key name is one of the obstacle labels (e.g., 'pickup_dispenser')
            if any(k in obj for k in obstacle_labels):
                return True
            # direct label value fallback
            for k in ('label', 'type', 'category', 'kind', 'tag', 'name'):
                v = obj.get(k)
                if isinstance(v, str) and v in obstacle_labels:
                    return True
            # any string field matching labels
            for v in obj.values():
                if isinstance(v, str) and v in obstacle_labels:
                    return True
            return False

        def process_vertices_list(lst):
            for entry in lst:
                # RMF-style: [x, y, meta]
                if (
                    isinstance(entry, list)
                    and len(entry) >= 2
                    and isinstance(entry[0], (int, float))
                    and isinstance(entry[1], (int, float))
                ):
                    meta = entry[2] if len(entry) >= 3 and isinstance(entry[2], dict) else {}
                    # Store as (row=y, col=x)
                    yv = float(entry[1]); xv = float(entry[0])
                    points.append((yv, xv, bool(is_obstacle_vertex(meta))))
                    continue
                # Dict-style
                if isinstance(entry, dict):
                    xy = extract_xy(entry)
                    if xy is not None:
                        yv, xv = xy
                        points.append((float(yv), float(xv), bool(is_obstacle_vertex(entry))))

        # Traverse data and collect labeled vertices having coordinates
        edges_raw: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []  # directed edges between (y,x)
        oneway_raw: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []  # store non-bidirectional lanes

        def search(obj):
            if isinstance(obj, dict):
                # Prefer structured parse of RMF levels
                if 'levels' in obj and isinstance(obj['levels'], dict):
                    for lvl in obj['levels'].values():
                        if isinstance(lvl, dict):
                            vlist = lvl.get('vertices')
                            if isinstance(vlist, list):
                                # Build local index->(y,x)
                                local_coords: List[Tuple[float, float]] = []
                                for entry in vlist:
                                    if (
                                        isinstance(entry, list)
                                        and len(entry) >= 2
                                        and isinstance(entry[0], (int, float))
                                        and isinstance(entry[1], (int, float))
                                    ):
                                        meta = entry[2] if len(entry) >= 3 and isinstance(entry[2], dict) else {}
                                        yv = float(entry[1]); xv = float(entry[0])
                                        local_coords.append((yv, xv))
                                        points.append((yv, xv, bool(is_obstacle_vertex(meta))))
                                    elif isinstance(entry, dict):
                                        xy = extract_xy(entry)
                                        if xy is not None:
                                            yv, xv = xy
                                            local_coords.append((float(yv), float(xv)))
                                            points.append((float(yv), float(xv), bool(is_obstacle_vertex(entry))))
                                # lanes referencing local indices
                                lanes = lvl.get('lanes')
                                if isinstance(lanes, list):
                                    for lane in lanes:
                                        if isinstance(lane, list) and len(lane) >= 2:
                                            u = lane[0]; v = lane[1]
                                            props = lane[2] if len(lane) >= 3 and isinstance(lane[2], dict) else {}
                                            if isinstance(u, int) and isinstance(v, int):
                                                if 0 <= u < len(local_coords) and 0 <= v < len(local_coords):
                                                    is_bi = bool(props.get('is_bidirectional', True))
                                                    yu, xu = local_coords[u]
                                                    yv2, xv2 = local_coords[v]
                                                    edges_raw.append(((yu, xu), (yv2, xv2)))
                                                    if is_bi:
                                                        edges_raw.append(((yv2, xv2), (yu, xu)))
                                                    else:
                                                        oneway_raw.append(((yu, xu), (yv2, xv2)))
                        # Done with this level
                    return  # already traversed structured levels
                # Generic scan fallback
                for k, v in obj.items():
                    if k == 'vertices' and isinstance(v, list):
                        process_vertices_list(v)
                    else:
                        search(v)
            elif isinstance(obj, list):
                for it in obj:
                    search(it)

        search(data)

        if not points:
            raise ValueError("YAML did not contain any vertices with coordinates.")

        # Map unique coordinates directly to grid indices (one vertex = one cell)
        # Round to reduce floating noise when building unique axes.
        def q(v: float) -> float:
            return round(v, 6)

        xs_sorted = sorted(set(q(x) for _, x, _ in points))
        ys_sorted = sorted(set(q(y) for y, _, _ in points))
        x_to_col = {x: i for i, x in enumerate(xs_sorted)}
        y_to_row = {y: i for i, y in enumerate(ys_sorted)}
        rows, cols = len(ys_sorted), len(xs_sorted)
        # Initialize as obstacles (1) for every grid position that has no vertex
        grid = [[1 for _ in range(cols)] for _ in range(rows)]

        vertex_cells: Set[Coord] = set()
        for y, x, is_obs in points:
            r = y_to_row[q(y)]
            c = x_to_col[q(x)]
            # Present vertex: free if not labeled obstacle; obstacle if labeled
            grid[r][c] = 1 if is_obs else 0
            vertex_cells.add((r, c))
        mg = MapGrid(grid)
        # Build adjacency map from edges (directed), using grid coordinates
        if edges_raw:
            adj: Dict[Coord, List[Coord]] = {}
            for (yu, xu), (yv2, xv2) in edges_raw:
                ru = y_to_row.get(q(yu)); cu = x_to_col.get(q(xu))
                rv = y_to_row.get(q(yv2)); cv = x_to_col.get(q(xv2))
                if ru is None or cu is None or rv is None or cv is None:
                    continue
                u_rc = (ru, cu); v_rc = (rv, cv)
                adj.setdefault(u_rc, []).append(v_rc)
            setattr(mg, '_adjacency', adj)
        # Store one-way arrows for drawing, if any
        if 'oneway_raw' in locals() and oneway_raw:
            arrows: List[Tuple[Coord, Coord]] = []
            for (yu, xu), (yv2, xv2) in oneway_raw:
                ru = y_to_row.get(q(yu)); cu = x_to_col.get(q(xu))
                rv = y_to_row.get(q(yv2)); cv = x_to_col.get(q(xv2))
                if ru is None or cu is None or rv is None or cv is None:
                    continue
                arrows.append(((ru, cu), (rv, cv)))
            setattr(mg, '_oneway_arrows', arrows)
        # Map one-way lanes to grid coords so we can draw direction arrows
        if oneway_raw:
            arrows: List[Tuple[Coord, Coord]] = []
            for (yu, xu), (yv2, xv2) in oneway_raw:
                ru = y_to_row.get(q(yu)); cu = x_to_col.get(q(xu))
                rv = y_to_row.get(q(yv2)); cv = x_to_col.get(q(xv2))
                if ru is None or cu is None or rv is None or cv is None:
                    continue
                arrows.append(((ru, cu), (rv, cv)))
            setattr(mg, '_oneway_arrows', arrows)
        # Record which grid cells correspond to YAML vertices
        setattr(mg, '_vertex_cells', vertex_cells)
        return mg

    def to_txt_string(self) -> str:
        return "\n".join("".join('1' if v == 1 else '0' for v in row) for row in self.grid)

# ======================
# A* and K-Shortest (Yen)
# ======================

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar(
    grid: MapGrid,
    start: Coord,
    goal: Coord,
    blocked_nodes: Optional[Set[Coord]] = None,
    blocked_edges: Optional[Set[Tuple[Coord, Coord]]] = None,
    allow_occupied: Optional[Set[Coord]] = None,
    cost_map: Optional[Union['np.ndarray', List[List[float]]]] = None,
    turn_penalty: Optional[Union['np.ndarray', List[List[float]]]] = None,
) -> Optional[Path]:
    """A* on 4-connected grid with optional blocked nodes/edges and costs.
       blocked_edges treats edges as directed pairs (u,v).

       Optional cost inputs (both default to None → all zeros):
       - cost_map: per-cell base cost (np.ndarray or list[list[float]])
       - turn_penalty: per-cell turn penalty applied when the direction
         changes at cell `u` while moving `u → v` (np.ndarray or list[list[float]]).

       Step cost u→v:
           1 + avg(cost_map[u], cost_map[v]) + (turned ? turn_penalty[u] : 0)

       Compatibility: accepts either this module's MapGrid API (tuple-based)
       or a grid with methods like `in_bounds(r,c)`, `is_free(r,c)`, and
       `nbrs4(r,c)` (as used in context_features.py).
    """

    def _in_bounds(g, rc: Coord) -> bool:
        try:
            return g.in_bounds(rc)  # tuple-style
        except TypeError:
            r, c = rc
            return g.in_bounds(r, c)
        except AttributeError:
            r, c = rc
            rows = getattr(g, 'rows', len(getattr(g, 'grid', [])))
            cols = getattr(g, 'cols', len(getattr(g, 'grid', [[]])[0]) if getattr(g, 'grid', []) else 0)
            return 0 <= r < rows and 0 <= c < cols

    allow_occupied = set(allow_occupied or [])

    def _is_free(g, rc: Coord) -> bool:
        if rc in allow_occupied:
            return True
        try:
            return g.is_free(rc)  # tuple-style
        except TypeError:
            r, c = rc
            return g.is_free(r, c)
        except AttributeError:
            r, c = rc
            return getattr(g, 'grid')[r][c] == 0

    def _neighbors4(g, rc: Coord) -> List[Coord]:
        # Prefer object's neighbors if provided (e.g., lane-based adjacency),
        # then filter by occupancy to honor allow_occupied.
        if hasattr(g, 'neighbors4'):
            try:
                nbrs = list(g.neighbors4(rc))
            except TypeError:
                r, c = rc
                nbrs = list(g.neighbors4(r, c))
            return [p for p in nbrs if _in_bounds(g, p) and _is_free(g, p)]
        if hasattr(g, 'nbrs4'):
            r, c = rc
            nbrs = list(g.nbrs4(r, c))
            return [p for p in nbrs if _in_bounds(g, p) and _is_free(g, p)]
        r, c = rc
        cand = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [p for p in cand if _in_bounds(g, p) and _is_free(g, p)]

    if not _in_bounds(grid, start) or not _in_bounds(grid, goal):
        return None
    if not _is_free(grid, start) or not _is_free(grid, goal):
        return None

    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()

    def _cell_val(arr: Optional[Union['np.ndarray', List[List[float]]]], rc: Coord) -> float:
        if arr is None:
            return 0.0
        r, c = rc
        try:
            return float(arr[r][c])  # supports list[list] and numpy arrays
        except Exception:
            # Last-resort: try attribute access like arr[r, c]
            try:
                return float(arr[r, c])  # type: ignore[index]
            except Exception:
                return 0.0

    def _dir(a: Coord, b: Coord) -> Tuple[int, int]:
        return (b[0] - a[0], b[1] - a[1])

    openq: List[Tuple[float, int, Coord]] = []
    g: Dict[Coord, float] = {start: 0.0}
    came: Dict[Coord, Coord] = {}
    cnt = 0
    heapq.heappush(openq, (float(manhattan(start, goal)), cnt, start))
    closed = set()

    while openq:
        _, _, u = heapq.heappop(openq)
        if u in closed:
            continue
        if u == goal:
            path: Path = [u]
            while u in came:
                u = came[u]
                path.append(u)
            path.reverse()
            return path
        closed.add(u)
        for v in _neighbors4(grid, u):
            if v in blocked_nodes:
                continue
            if (u, v) in blocked_edges:
                continue
            # Compute per-step cost with optional maps
            # turn penalty applies when direction changes at u (entering v)
            turned = False
            if u in came:
                prev = came[u]
                turned = _dir(prev, u) != _dir(u, v)
            base_avg = 0.5 * (_cell_val(cost_map, u) + _cell_val(cost_map, v))
            turn_cost = _cell_val(turn_penalty, u) if turned else 0.0
            step = 1.0 + base_avg + turn_cost
            tentative = g[u] + step
            if tentative < g.get(v, 10**9):
                g[v] = tentative
                came[v] = u
                cnt += 1
                f = tentative + float(manhattan(v, goal))
                heapq.heappush(openq, (f, cnt, v))
    return None


def yen_k_shortest(
    grid: MapGrid,
    start: Coord,
    goal: Coord,
    K: int = 5,
    allow_occupied: Optional[Set[Coord]] = None,
    cost_map: Optional[Union['np.ndarray', List[List[float]]]] = None,
    turn_penalty: Optional[Union['np.ndarray', List[List[float]]]] = None,
) -> List[Path]:
    """Yen's algorithm on top of A* (4-connected)."""
    A: List[Path] = []  # accepted
    B: List[Tuple[float, Path]] = []  # candidates (cost, path)

    p0 = astar(
        grid, start, goal,
        allow_occupied=allow_occupied,
        cost_map=cost_map,
        turn_penalty=turn_penalty,
    )
    if p0 is None:
        return []
    A.append(p0)

    def _cell_val(arr: Optional[Union['np.ndarray', List[List[float]]]], rc: Coord) -> float:
        if arr is None:
            return 0.0
        r, c = rc
        try:
            return float(arr[r][c])
        except Exception:
            try:
                return float(arr[r, c])  # type: ignore[index]
            except Exception:
                return 0.0

    def path_cost(p: Path) -> float:
        # Preserve legacy behavior exactly when no maps are provided
        if cost_map is None and turn_penalty is None:
            return float(len(p) - 1)
        if not p:
            return 0.0
        total = 0.0
        for i in range(len(p) - 1):
            u, v = p[i], p[i+1]
            turned = False
            if i > 0:
                prev = p[i-1]
                if (u[0]-prev[0], u[1]-prev[1]) != (v[0]-u[0], v[1]-u[1]):
                    turned = True
            base_avg = 0.5 * (_cell_val(cost_map, u) + _cell_val(cost_map, v))
            turn_cost = _cell_val(turn_penalty, u) if turned else 0.0
            total += 1.0 + base_avg + turn_cost
        return total

    for _ in range(1, K):
        prev = A[-1]
        for i in range(len(prev)-1):
            spur_node = prev[i]
            root_path = prev[:i+1]

            blocked_nodes: Set[Coord] = set(root_path[:-1])
            blocked_edges: Set[Tuple[Coord, Coord]] = set()
            for p in A:
                if len(p) > i and p[:i+1] == root_path:
                    blocked_edges.add((p[i], p[i+1]))

            spur_path = astar(
                grid, spur_node, goal,
                blocked_nodes, blocked_edges,
                allow_occupied=allow_occupied,
                cost_map=cost_map,
                turn_penalty=turn_penalty,
            )
            if spur_path is None:
                continue
            cand = root_path[:-1] + spur_path
            if cand not in [bp for _, bp in B] and cand not in A:
                heapq.heappush(B, (path_cost(cand), cand))
        if not B:
            break
        _, best = heapq.heappop(B)
        A.append(best)
    return A

# ======================
# Path Quality Scoring (ROS2‑style, whole path)
# ======================
@dataclass
class ScoreWeights:
    w_len: float = 1.0
    w_turn: float = 0.25
    # simple overlaps (small default)
    w_overlap_cell: float = 2.0
    w_overlap_edge: float = 1.0
    # time-synchronized node conflicts (same cell, same timestamp)
    w_path_overlap: float = 0.0
    # ROS2-style conflict penalties
    w_h2h: float = 20.0         # head-to-head swaps/head-on passes
    w_deadlock: float = 50.0    # directed wait-for cycles (>=3 robots)
    w_self_cycle: float = 5.0   # partial trivial cyclic


# ----- utilities for overlaps/edges -----

def count_turns(path: Path) -> int:
    if len(path) < 3:
        return 0
    def dir_of(a: Coord, b: Coord) -> Tuple[int,int]:
        return (b[0]-a[0], b[1]-a[1])
    turns = 0
    prev = dir_of(path[0], path[1])
    for i in range(1, len(path)-1):
        cur = dir_of(path[i], path[i+1])
        if cur != prev:
            turns += 1
        prev = cur
    return turns


def undirected_edges(path: Path) -> Set[Tuple[Coord, Coord]]:
    s = set()
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        if u <= v:
            s.add((u, v))
        else:
            s.add((v, u))
    return s


# ----- align paths across full timeline (no window) -----

def _extend_to_length(path: Path, L: int) -> Path:
    if not path:
        return []
    if len(path) >= L:
        return path[:L]
    return path + [path[-1]]*(L - len(path))


def _align_paths(candidate: Path, others: List[Path]) -> Tuple[List[Coord], List[List[Coord]]]:
    T = max([len(candidate)] + [len(o) for o in others] + [1])
    cand = _extend_to_length(candidate, T)
    oth = [_extend_to_length(o, T) for o in others]
    return cand, oth


# ----- ROS2-like conflict metrics over full horizon -----

def _count_path_overlap_full(cand: List[Coord], others: List[List[Coord]]) -> int:
    cnt = 0
    for o in others:
        for t in range(len(cand)):
            if cand[t] == o[t]:
                cnt += 1
    return cnt


def _count_head_to_head_full(cand: List[Coord], others: List[List[Coord]]) -> int:
    cnt = 0
    T = len(cand)
    for o in others:
        for t in range(T-1):
            c_curr, c_next = cand[t], cand[t+1]
            o_curr, o_next = o[t],    o[t+1]
            # ① classic swap
            if c_curr == o_next and o_curr == c_next:
                cnt += 1
                continue
            # ② head-on convergence turning into a pass (both continue into each other's start)
            if c_next == o_next and c_curr != o_curr and t+2 < T:
                cont1 = cand[t+2] == o_curr
                cont2 = o[t+2]    == c_curr
                if cont1 and cont2:
                    cnt += 1
    return cnt


def _count_self_trivial_cycles(path: Path) -> int:
    seen: Dict[Coord, int] = {}
    repeats = 0
    for i, node in enumerate(path):
        if node in seen:
            repeats += 1
        else:
            seen[node] = i
    return repeats


def _count_deadlock_cycles_involving_candidate(cand: List[Coord], others: List[List[Coord]]) -> int:
    # Build wait-for graph at every t: r→s if r's next node equals s's current node
    # Count cycles (>=3 robots) that include the candidate robot.
    robots_idx = list(range(len(others) + 1))  # 0=candidate, then others 1..N
    all_paths = [cand] + others
    T = len(cand)

    def has_cycle_including_zero(adj: Dict[int, Set[int]]) -> bool:
        visited, stack = set(), []
        onstack: Set[int] = set()

        def dfs(v: int) -> bool:
            visited.add(v)
            stack.append(v)
            onstack.add(v)
            for nbr in adj.get(v, set()):
                if nbr not in visited:
                    if dfs(nbr):
                        return True
                elif nbr in onstack:
                    idx = stack.index(nbr)
                    cyc = set(stack[idx:])
                    # ignore 2-cycles (treated as head-to-head)
                    if len(cyc) >= 3 and 0 in cyc:
                        return True
            stack.pop()
            onstack.remove(v)
            return False

        for v in adj.keys():
            if v not in visited and dfs(v):
                return True
        return False

    deadlocks = 0
    for t in range(T-1):
        adj: Dict[int, Set[int]] = {i: set() for i in robots_idx}
        for i in robots_idx:
            curr_i = all_paths[i][t]
            next_i = all_paths[i][t+1]
            if next_i == curr_i:
                continue
            for j in robots_idx:
                if i == j:
                    continue
                curr_j = all_paths[j][t]
                if next_i == curr_j:
                    adj[i].add(j)
        if has_cycle_including_zero(adj):
            deadlocks += 1
    return deadlocks


def _conflict_cells_for_candidate(cand: List[Coord], others: List[List[Coord]]) -> Tuple[Set[Coord], Set[Coord], Set[Coord]]:
    """Return sets of cells along the candidate path involved in conflicts.
    - path_overlap: same cell at same timestamp with any other robot → mark cand[t]
    - h2h: head-to-head swap/pass situations → mark cand[t+1]
    - deadlock: wait-for cycle (>=3) including candidate at time t → mark cand[t+1]
    """
    T = len(cand)
    path_ov: Set[Coord] = set()
    h2h_cells: Set[Coord] = set()
    dead_cells: Set[Coord] = set()
    # path overlap
    for o in others:
        for t in range(T):
            if cand[t] == o[t]:
                path_ov.add(cand[t])
    # h2h (swap or pass)
    for o in others:
        for t in range(T-1):
            c_curr, c_next = cand[t], cand[t+1]
            o_curr, o_next = o[t],    o[t+1]
            swap = (c_curr == o_next and o_curr == c_next)
            pass_through = False
            if c_next == o_next and c_curr != o_curr and t+2 < T:
                pass_through = (cand[t+2] == o_curr and o[t+2] == c_curr)
            if swap or pass_through:
                h2h_cells.add(c_next)
    # deadlock detection per t
    robots_idx = list(range(len(others) + 1))  # 0=cand
    all_paths = [cand] + others
    def has_cycle_including_zero(adj: Dict[int, Set[int]]) -> bool:
        visited: Set[int] = set()
        onstack: Set[int] = set()
        stack: List[int] = []
        def dfs(v: int) -> bool:
            visited.add(v); onstack.add(v); stack.append(v)
            for nbr in adj.get(v, set()):
                if nbr not in visited:
                    if dfs(nbr):
                        return True
                elif nbr in onstack:
                    try:
                        idx = stack.index(nbr)
                        cyc = set(stack[idx:])
                    except ValueError:
                        cyc = {nbr}
                    if len(cyc) >= 3 and 0 in cyc:
                        return True
            stack.pop(); onstack.remove(v)
            return False
        for v in list(adj.keys()):
            if v not in visited and dfs(v):
                return True
        return False
    for t in range(T-1):
        adj: Dict[int, Set[int]] = {i: set() for i in robots_idx}
        for i in robots_idx:
            curr_i = all_paths[i][t]
            next_i = all_paths[i][t+1]
            if next_i == curr_i:
                continue
            for j in robots_idx:
                if i == j:
                    continue
                curr_j = all_paths[j][t]
                if next_i == curr_j:
                    adj[i].add(j)
        if has_cycle_including_zero(adj):
            dead_cells.add(cand[t+1])
    return path_ov, h2h_cells, dead_cells


# ----- Cost computation (components for display) -----

def cost_components(candidate: Path, others: List[Path], w: ScoreWeights) -> Tuple[Dict[str, float], float]:
    L = max(1, len(candidate)-1)
    Tturns = count_turns(candidate)
    cand_aligned, others_aligned = _align_paths(candidate, others)
    # time-synchronized node conflicts
    path_overlap = _count_path_overlap_full(cand_aligned, others_aligned)
    # static cell reuse across any time (unique cells)
    cells_cand = set(candidate)
    cells_others = set().union(*map(set, others)) if others else set()
    cell_overlap = len(cells_cand & cells_others)
    edge_overlap = len(undirected_edges(candidate) & set().union(*map(undirected_edges, others))) if others else 0
    h2h = _count_head_to_head_full(cand_aligned, others_aligned)
    dead = _count_deadlock_cycles_involving_candidate(cand_aligned, others_aligned)
    self_cyc = _count_self_trivial_cycles(candidate)
    components = {
        'length': L,
        'turns': Tturns,
        'cell_overlap': cell_overlap,
        'path_overlap': path_overlap,
        'edge_overlap': edge_overlap,
        'h2h': h2h,
        'deadlock': dead,
        'self_cycle': self_cyc,
    }
    cost = (
        w.w_len * L + w.w_turn * Tturns +
        w.w_overlap_cell * cell_overlap + w.w_overlap_edge * edge_overlap + w.w_path_overlap * path_overlap +
        w.w_h2h * h2h + w.w_deadlock * dead + w.w_self_cycle * self_cyc
    )
    return components, cost


def score_path(candidate: Path, others: List[Path], w: ScoreWeights) -> float:
    _, cost = cost_components(candidate, others, w)
    return -cost  # higher is better

# ==============
# Optional L2R Ranker (self-contained loader)
# ==============
class L2RRanker:
    """Loads a joblib-packed model for Learning-to-Rank.
    - If model_type == "ranker" (e.g., LightGBM/XGBoost rankers), predict() is
      interpreted as a score where higher is better.
    - Otherwise, the model is assumed to predict cost; we convert to a score by
      returning -pred_cost so that higher is still better.

    Expected joblib content: {"model": model, "features": [...], "model_type": str}
    If not available, .available() is False and calls will be ignored.
    """
    def __init__(self, model_path: str = "ranker_model.joblib"):
        self.ok = False
        self.model = None
        self.features = [
            "length","turns","cell_overlap","path_overlap","edge_overlap",
            "h2h","deadlock","self_cycle","n_others","K"
        ]
        self.model_type = None  # 'ranker' or None
        try:
            from joblib import load  # type: ignore
            pack = load(model_path)
            # allow either dict pack or bare model
            if isinstance(pack, dict):
                self.model = pack.get("model", None)
                self.features = pack.get("features", self.features)
                self.model_type = pack.get("model_type", None)
            else:
                self.model = pack
            # quick numpy import test
            import numpy as np  # noqa: F401
            if self.model is not None:
                self.ok = True
        except Exception:
            self.ok = False

    def available(self) -> bool:
        return self.ok

    def predict_neg_cost_scores(self, candidates: List[Path], committed_paths: List[Path], disabled: Optional[Set[str]] = None) -> List[float]:
        """Return scores for candidates where higher is better.
        For ranker models, uses model.predict directly.
        For regressors predicting cost, returns -pred_cost.
        """
        if not self.ok or not candidates:
            return [float('-inf')]*len(candidates)
        # Build feature rows matching training
        rows = []
        # Use default weights only for feature computation; the model learned its own mapping
        default_w = ScoreWeights()
        disabled = set(disabled or [])
        for cand in candidates:
            comp, _ = cost_components(cand, committed_paths, default_w)
            row = {
                "length": comp['length'],
                "turns": comp['turns'],
                "cell_overlap": comp['cell_overlap'],
                "path_overlap": comp['path_overlap'],
                "edge_overlap": comp['edge_overlap'],
                "h2h": comp['h2h'],
                "deadlock": comp['deadlock'],
                "self_cycle": comp['self_cycle'],
                "n_others": len(committed_paths),
                "K": len(candidates),
            }
            # Neutralize disabled features so they can't sway ordering within this case
            for k in ("length","turns","cell_overlap","path_overlap","edge_overlap","h2h","deadlock","self_cycle"):
                if k in disabled:
                    row[k] = 0.0
            rows.append([row.get(k, 0.0) for k in self.features])
        import numpy as np
        X = np.array(rows, dtype=float)
        pred = self.model.predict(X)
        # If it's a true ranker, treat pred as score; else assume it's cost
        if (self.model_type or "").lower() == "ranker":
            return list(map(float, pred))
        # Fallback: negative cost as a score
        return (-(pred)).tolist()

# ============
# PyQt5  UI
# ============
class GridView(QtWidgets.QGraphicsView):
    cellClicked = QtCore.pyqtSignal(int, int)  # row, col

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.cell_size = 22
        self.map: Optional[MapGrid] = None
        self.rect_items: List[List[QtWidgets.QGraphicsRectItem]] = []
        # Zoom setup
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self._zoom_steps = 0
        self._zoom_step_factor = 1.2
        self._zoom_min_steps = -15
        self._zoom_max_steps = 25

    def load_map(self, grid: MapGrid):
        self.map = grid
        self.scene.clear()
        self.rect_items = []
        if not grid:
            return
        s = self.cell_size
        for r in range(grid.rows):
            row_items = []
            for c in range(grid.cols):
                rect = QtCore.QRectF(c*s, r*s, s, s)
                item = self.scene.addRect(
                    rect,
                    pen=QtGui.QPen(QtCore.Qt.gray),
                    brush=QtGui.QBrush(QtCore.Qt.black if grid.grid[r][c] == 1 else QtCore.Qt.white),
                )
                item.setData(0, "base")  # tag as base grid cell
                row_items.append(item)
            self.rect_items.append(row_items)
        self.setSceneRect(0, 0, grid.cols*s, grid.rows*s)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if not self.map:
            return
        pos = self.mapToScene(event.pos())
        s = self.cell_size
        c = int(pos.x() // s)
        r = int(pos.y() // s)
        if 0 <= r < self.map.rows and 0 <= c < self.map.cols:
            self.cellClicked.emit(r, c)
        super().mousePressEvent(event)

    # --- Zooming/panning helpers ---
    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Ctrl + Wheel to zoom (preserve normal scroll without Ctrl)
        if event.modifiers() & QtCore.Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            elif delta < 0:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def zoom_in(self):
        if self._zoom_steps >= self._zoom_max_steps:
            return
        self._zoom_steps += 1
        self.scale(self._zoom_step_factor, self._zoom_step_factor)

    def zoom_out(self):
        if self._zoom_steps <= self._zoom_min_steps:
            return
        self._zoom_steps -= 1
        factor = 1.0 / self._zoom_step_factor
        self.scale(factor, factor)

    def reset_zoom(self):
        self._zoom_steps = 0
        self.resetTransform()

    def color_cell(self, r: int, c: int, color: QtGui.QColor):
        if not self.map:
            return
        item = self.rect_items[r][c]
        item.setBrush(QtGui.QBrush(color))

    def _add_overlay_item(self, item: QtWidgets.QGraphicsItem):
        item.setData(0, "overlay")
        return item

    def draw_polyline(self, path: Path, pen: QtGui.QPen):
        if not path:
            return
        s = self.cell_size
        qpath = QtGui.QPainterPath()
        qpath.moveTo(path[0][1]*s + s/2, path[0][0]*s + s/2)
        for (r, c) in path[1:]:
            qpath.lineTo(c*s + s/2, r*s + s/2)
        item = self.scene.addPath(qpath, pen)
        self._add_overlay_item(item)

    def draw_planned(self, path: Path, color: QtGui.QColor):
        pen = QtGui.QPen(color)
        pen.setWidth(3)
        self.draw_polyline(path, pen)

    def draw_candidate(self, path: Path):
        pen = QtGui.QPen(QtGui.QColor(120,120,120))  # gray
        pen.setWidth(2)
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        self.draw_polyline(path, pen)

    def draw_highlight(self, path: Path, color: QtGui.QColor):
        # Emphasized overlay for selected candidate (dashed colored)
        pen = QtGui.QPen(color.lighter(110))
        pen.setWidth(5)
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        self.draw_polyline(path, pen)

    def draw_costmap_overlay(self, cost_map) -> None:
        # Draw a translucent heat overlay per cell based on cost value [0..3]
        if not self.map or cost_map is None:
            return
        try:
            import numpy as np  # type: ignore
        except Exception:
            return
        try:
            rows, cols = int(cost_map.shape[0]), int(cost_map.shape[1])  # type: ignore[attr-defined]
        except Exception:
            return
        rows = min(rows, self.map.rows)
        cols = min(cols, self.map.cols)
        s = self.cell_size
        nop = QtGui.QPen(QtCore.Qt.NoPen)
        for r in range(rows):
            for c in range(cols):
                try:
                    v = float(cost_map[r][c])
                except Exception:
                    continue
                if v <= 1e-6:
                    continue
                t = max(0.0, min(1.0, v / 3.0))
                # Alpha proportional to magnitude; color = red
                alpha = int(200 * t)
                if alpha <= 0:
                    continue
                color = QtGui.QColor(255, 0, 0, alpha)
                rect = QtCore.QRectF(c*s, r*s, s, s)
                item = self.scene.addRect(rect, nop, QtGui.QBrush(color))
                self._add_overlay_item(item)

    def draw_dashed_colored(self, path: Path, color: QtGui.QColor, width: int = 3):
        pen = QtGui.QPen(color)
        pen.setWidth(width)
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        self.draw_polyline(path, pen)

    def draw_arrow(self, u: Coord, v: Coord, color: QtGui.QColor):
        # Arrow from center of u to center of v
        s = self.cell_size
        x1, y1 = u[1]*s + s/2.0, u[0]*s + s/2.0
        x2, y2 = v[1]*s + s/2.0, v[0]*s + s/2.0
        pen = QtGui.QPen(color)
        pen.setWidth(1)
        pen.setCosmetic(True)
        path = QtGui.QPainterPath(QtCore.QPointF(x1, y1))
        path.lineTo(QtCore.QPointF(x2, y2))
        item = self.scene.addPath(path, pen)
        self._add_overlay_item(item)
        # Head
        import math as _m
        angle = _m.atan2(y2 - y1, x2 - x1)
        ah = s * 0.25
        tip = QtCore.QPointF(x2, y2)
        left = QtCore.QPointF(x2 - _m.cos(angle - _m.radians(25)) * ah,
                               y2 - _m.sin(angle - _m.radians(25)) * ah)
        right = QtCore.QPointF(x2 - _m.cos(angle + _m.radians(25)) * ah,
                                y2 - _m.sin(angle + _m.radians(25)) * ah)
        poly = QtGui.QPolygonF([tip, left, right])
        head = self.scene.addPolygon(poly, pen, QtGui.QBrush(color))
        self._add_overlay_item(head)

    def draw_arrow(self, u: Coord, v: Coord, color: QtGui.QColor):
        # Draw a small arrow from cell u to cell v (center to center)
        s = self.cell_size
        x1 = u[1]*s + s/2.0; y1 = u[0]*s + s/2.0
        x2 = v[1]*s + s/2.0; y2 = v[0]*s + s/2.0
        pen = QtGui.QPen(color)
        pen.setWidth(1)
        pen.setCosmetic(True)
        path = QtGui.QPainterPath(QtCore.QPointF(x1, y1))
        path.lineTo(QtCore.QPointF(x2, y2))
        item = self.scene.addPath(path, pen)
        self._add_overlay_item(item)
        # Arrow head
        import math as _m
        angle = _m.atan2(y2 - y1, x2 - x1)
        ah = s * 0.25
        p_tip = QtCore.QPointF(x2, y2)
        p_l = QtCore.QPointF(x2 - _m.cos(angle - _m.radians(25)) * ah,
                              y2 - _m.sin(angle - _m.radians(25)) * ah)
        p_r = QtCore.QPointF(x2 - _m.cos(angle + _m.radians(25)) * ah,
                              y2 - _m.sin(angle + _m.radians(25)) * ah)
        head = QtGui.QPolygonF([p_tip, p_l, p_r])
        head_item = self.scene.addPolygon(head, pen, QtGui.QBrush(color))
        self._add_overlay_item(head_item)

    def draw_start_marker(self, r: int, c: int, color: QtGui.QColor):
        # Draw filled triangle marker for Start
        s = self.cell_size
        margin = s * 0.2
        left   = c*s + margin
        right  = c*s + s - margin
        top    = r*s + margin
        bottom = r*s + s - margin
        path = QtGui.QPainterPath()
        path.moveTo((left+right)/2.0, top)        # apex top center
        path.lineTo(left, bottom)                 # bottom left
        path.lineTo(right, bottom)                # bottom right
        path.closeSubpath()
        pen = QtGui.QPen(QtGui.QColor(255,255,255))
        pen.setWidthF(1.5)
        brush = QtGui.QBrush(color)
        item = self.scene.addPath(path, pen, brush)
        self._add_overlay_item(item)
        item.setZValue(10)

    def draw_end_marker(self, r: int, c: int, color: QtGui.QColor):
        # Draw solid square marker for End
        s = self.cell_size
        margin = s * 0.22
        rect = QtCore.QRectF(c*s + margin, r*s + margin, s - 2*margin, s - 2*margin)
        pen = QtGui.QPen(QtGui.QColor(255,255,255))
        pen.setWidthF(1.5)
        brush = QtGui.QBrush(color)
        item = self.scene.addRect(rect, pen, brush)
        self._add_overlay_item(item)
        item.setZValue(10)

    # --- Conflict markers ---
    def draw_conflict_circle(self, rc: Coord, color: QtGui.QColor):
        # Small filled circle at cell center
        r, c = rc
        s = self.cell_size
        radius = s * 0.18
        cx = c*s + s/2.0
        cy = r*s + s/2.0
        rect = QtCore.QRectF(cx - radius, cy - radius, 2*radius, 2*radius)
        pen = QtGui.QPen(QtGui.QColor(255,255,255))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        brush = QtGui.QBrush(color)
        item = self.scene.addEllipse(rect, pen, brush)
        self._add_overlay_item(item)

    def draw_conflict_cross(self, rc: Coord, color: QtGui.QColor):
        # A small X centered in the cell
        r, c = rc
        s = self.cell_size
        margin = s * 0.22
        x1 = c*s + margin; y1 = r*s + margin
        x2 = c*s + s - margin; y2 = r*s + s - margin
        pen = QtGui.QPen(color)
        pen.setWidth(2)
        pen.setCosmetic(True)
        l1 = self.scene.addLine(x1, y1, x2, y2, pen)
        l2 = self.scene.addLine(x1, y2, x2, y1, pen)
        self._add_overlay_item(l1)
        self._add_overlay_item(l2)

    def draw_conflict_diamond(self, rc: Coord, color: QtGui.QColor):
        # Rotated square (diamond) inside the cell
        r, c = rc
        s = self.cell_size
        cx = c*s + s/2.0
        cy = r*s + s/2.0
        d = s * 0.28
        poly = QtGui.QPolygonF([
            QtCore.QPointF(cx, cy - d),  # top
            QtCore.QPointF(cx + d, cy),  # right
            QtCore.QPointF(cx, cy + d),  # bottom
            QtCore.QPointF(cx - d, cy),  # left
        ])
        pen = QtGui.QPen(color)
        pen.setWidth(2)
        pen.setCosmetic(True)
        brush = QtGui.QBrush(QtCore.Qt.transparent)
        item = self.scene.addPolygon(poly, pen, brush)
        self._add_overlay_item(item)

    def reset_visuals(self):
        if not self.map:
            return
        # Reset base cell colors
        for r in range(self.map.rows):
            for c in range(self.map.cols):
                base = QtCore.Qt.black if self.map.grid[r][c] == 1 else QtCore.Qt.white
                self.rect_items[r][c].setBrush(QtGui.QBrush(base))
        # Remove overlays
        for item in list(self.scene.items()):
            if item.data(0) != "base":
                self.scene.removeItem(item)
        # Keep zoom as-is intentionally on visual reset


class MainWindow(QtWidgets.QWidget):
    COLORS = [
        QtGui.QColor(220,20,60),   # crimson
        QtGui.QColor(30,144,255),  # dodger blue
        QtGui.QColor(60,179,113),  # medium sea green
        QtGui.QColor(238,130,238), # violet
        QtGui.QColor(255,140,0),   # dark orange
        QtGui.QColor(0,191,255),   # deep sky blue
        QtGui.QColor(154,205,50),  # yellow green
        QtGui.QColor(255,99,71),   # tomato
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi‑Robot Path Quality Selector — Sequential Demo")
        self.resize(1320, 900)

        self.grid_view = GridView()
        # Secondary view for cost-map (in a tab)
        self.costmap_view = GridView()
        self.load_btn = QtWidgets.QPushButton("Load Map…")
        self.nrobots_spin = QtWidgets.QSpinBox(); self.nrobots_spin.setRange(1, 8); self.nrobots_spin.setValue(3)

        self.mode_label = QtWidgets.QLabel("Click grid to set Start for Robot 1")
        self.set_start_btn = QtWidgets.QPushButton("Select Starts")
        self.set_end_btn   = QtWidgets.QPushButton("Select Ends")
        self.undo_btn      = QtWidgets.QPushButton("Undo Last Robot")
        self.clear_btn     = QtWidgets.QPushButton("Clear All")
        self.show_alts_chk = QtWidgets.QCheckBox("Show other K‑shortest alternatives (current robot only)")
        self.show_alts_chk.setChecked(False)
        self.show_arrows_chk = QtWidgets.QCheckBox("Show one‑way lane arrows")
        self.show_arrows_chk.setChecked(True)
        self.use_costmap_chk = QtWidgets.QCheckBox("Use analytic cost-map")
        self.use_costmap_chk.setChecked(False)
        self.use_learned_costmap_chk = QtWidgets.QCheckBox("Use learned cost-map")
        self.use_learned_costmap_chk.setChecked(False)
        self.show_costmap_chk = QtWidgets.QCheckBox("Show cost-map overlay")
        self.show_costmap_chk.setChecked(False)

        # Scoring / search controls
        self.k_spin = QtWidgets.QSpinBox(); self.k_spin.setRange(1,100); self.k_spin.setValue(5)
        self.wlen = QtWidgets.QDoubleSpinBox(); self.wlen.setRange(0.0, 10.0); self.wlen.setSingleStep(0.1); self.wlen.setValue(1.0)
        self.wturn = QtWidgets.QDoubleSpinBox(); self.wturn.setRange(0.0, 10.0); self.wturn.setSingleStep(0.05); self.wturn.setValue(0.25)
        self.wcell = QtWidgets.QDoubleSpinBox(); self.wcell.setRange(0.0, 50.0); self.wcell.setSingleStep(0.5); self.wcell.setValue(2.0)
        self.wpath = QtWidgets.QDoubleSpinBox(); self.wpath.setRange(0.0, 50.0); self.wpath.setSingleStep(0.5); self.wpath.setValue(0.0)
        self.wedge = QtWidgets.QDoubleSpinBox(); self.wedge.setRange(0.0, 50.0); self.wedge.setSingleStep(0.5); self.wedge.setValue(1.0)
        self.wh2h = QtWidgets.QDoubleSpinBox(); self.wh2h.setRange(0.0, 200.0); self.wh2h.setSingleStep(1.0); self.wh2h.setValue(20.0)
        self.wdead = QtWidgets.QDoubleSpinBox(); self.wdead.setRange(0.0, 500.0); self.wdead.setSingleStep(5.0); self.wdead.setValue(50.0)
        self.wself = QtWidgets.QDoubleSpinBox(); self.wself.setRange(0.0, 100.0); self.wself.setSingleStep(1.0); self.wself.setValue(5.0)

        weights_form = QtWidgets.QFormLayout()
        weights_form.addRow("K-shortest:", self.k_spin)
        weights_form.addRow("w_len:", self.wlen)
        weights_form.addRow("w_turn:", self.wturn)
        weights_form.addRow("w_overlap_cell:", self.wcell)
        weights_form.addRow("w_path_overlap:", self.wpath)
        weights_form.addRow("w_overlap_edge:", self.wedge)
        weights_form.addRow("w_h2h:", self.wh2h)
        weights_form.addRow("w_deadlock:", self.wdead)
        weights_form.addRow("w_self_cycle:", self.wself)

        # Scorer toggle (L2R ranker)
        self.scorer_box = QtWidgets.QComboBox()
        self.scorer_box.addItems(["Heuristic", "Learned", "Blend 50/50"])

        # Weights toggle (Fixed / Learned / Blend)
        self.weight_mode_box = QtWidgets.QComboBox()
        self.weight_mode_box.addItems(["Fixed", "Learned", "Blend 50/50"])  # default Fixed

        # Candidate cost table
        self.cost_table = QtWidgets.QTableWidget(0, 12)
        self.cost_table.setHorizontalHeaderLabels([
            "#", "len", "turns", "cellOv", "pathOv", "edgeOv", "h2h", "deadlock", "selfCyc", "∑cost", "L2R_pred", "selected"
        ])
        self.cost_table.horizontalHeader().setStretchLastSection(True)
        self.cost_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cost_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cost_table.setMinimumHeight(240)

        # Tabbed container for main map and cost-map
        self.views = QtWidgets.QTabWidget()
        self.views_tab_idx_map = self.views.addTab(self.grid_view, "Map")
        self.views_tab_idx_cost = self.views.addTab(self.costmap_view, "Cost-map")

        left = QtWidgets.QVBoxLayout(); left.addWidget(self.views, 1)
        # Zoom controls (above the grid)
        zoom_row = QtWidgets.QHBoxLayout()
        self.zoom_out_btn = QtWidgets.QToolButton(); self.zoom_out_btn.setText("-")
        self.zoom_in_btn = QtWidgets.QToolButton();  self.zoom_in_btn.setText("+")
        self.zoom_reset_btn = QtWidgets.QToolButton(); self.zoom_reset_btn.setText("Reset")
        zoom_row.addWidget(QtWidgets.QLabel("Zoom:"))
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addWidget(self.zoom_reset_btn)
        zoom_row.addStretch(1)
        left.insertLayout(0, zoom_row)

        right = QtWidgets.QVBoxLayout()
        topbar = QtWidgets.QHBoxLayout()
        topbar.addWidget(self.load_btn)
        topbar.addWidget(QtWidgets.QLabel("Robots:"))
        topbar.addWidget(self.nrobots_spin)
        right.addLayout(topbar)
        right.addWidget(self.mode_label)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.set_start_btn)
        btnrow.addWidget(self.set_end_btn)
        right.addLayout(btnrow)

        btnrow2 = QtWidgets.QHBoxLayout()
        btnrow2.addWidget(self.undo_btn)
        btnrow2.addWidget(self.clear_btn)
        right.addLayout(btnrow2)

        right.addWidget(self.show_alts_chk)
        right.addWidget(self.show_arrows_chk)
        right.addWidget(self.use_costmap_chk)
        right.addWidget(self.use_learned_costmap_chk)
        right.addWidget(self.show_costmap_chk)
        self.show_costmap_tab_chk = QtWidgets.QCheckBox("Show cost-map tab")
        self.show_costmap_tab_chk.setChecked(False)
        right.addWidget(self.show_costmap_tab_chk)
        self.costmap_mode_label = QtWidgets.QLabel("Cost shaping: None")
        right.addWidget(self.costmap_mode_label)
        self.costmap_status = QtWidgets.QLabel("cost-map: off")
        right.addWidget(self.costmap_status)
        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("Scoring / Search Settings"))
        right.addLayout(weights_form)
        # Component toggles (one-click ON/OFF)
        comps_row = QtWidgets.QHBoxLayout()
        comps_row.addWidget(QtWidgets.QLabel("Enable components:"))
        self.comp_len = QtWidgets.QCheckBox("length"); self.comp_len.setChecked(True)
        self.comp_turn = QtWidgets.QCheckBox("turns"); self.comp_turn.setChecked(True)
        self.comp_cell = QtWidgets.QCheckBox("cellOv"); self.comp_cell.setChecked(True)
        self.comp_path = QtWidgets.QCheckBox("pathOv"); self.comp_path.setChecked(True)
        self.comp_edge = QtWidgets.QCheckBox("edgeOv"); self.comp_edge.setChecked(True)
        self.comp_h2h = QtWidgets.QCheckBox("h2h"); self.comp_h2h.setChecked(True)
        self.comp_dead = QtWidgets.QCheckBox("deadlock"); self.comp_dead.setChecked(True)
        self.comp_self = QtWidgets.QCheckBox("selfCyc"); self.comp_self.setChecked(True)
        for w in [self.comp_len, self.comp_turn, self.comp_cell, self.comp_path, self.comp_edge, self.comp_h2h, self.comp_dead, self.comp_self]:
            comps_row.addWidget(w)
        comps_row.addStretch(1)
        right.addLayout(comps_row)
        right.addSpacing(8)
        right.addWidget(QtWidgets.QLabel("Scorer"))
        right.addWidget(self.scorer_box)
        right.addSpacing(8)
        right.addWidget(QtWidgets.QLabel("Weights"))
        right.addWidget(self.weight_mode_box)
        right.addSpacing(8)
        right.addWidget(QtWidgets.QLabel("Candidate costs (last planned robot)"))
        right.addWidget(self.cost_table)
        self.reset_highlight_btn = QtWidgets.QPushButton("Reset Highlight")
        right.addWidget(self.reset_highlight_btn)
        right.addStretch(1)

        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(left, 4)
        root.addLayout(right, 3)

        # state
        self.grid: Optional[MapGrid] = None
        self.nrobots = self.nrobots_spin.value()
        self.starts: List[Optional[Coord]] = [None]*self.nrobots
        self.goals:  List[Optional[Coord]] = [None]*self.nrobots
        self.paths:  List[Path] = [[] for _ in range(self.nrobots)]  # committed paths
        self.candidates: List[List[Path]] = [[] for _ in range(self.nrobots)]  # K-shortest per robot
        self.candidate_costs: List[List[Tuple[Dict[str, float], float]]] = [[] for _ in range(self.nrobots)]
        self.candidate_pred_costs: List[List[Optional[float]]] = [[] for _ in range(self.nrobots)]
        self.applied_weights: List[Optional[ScoreWeights]] = [None for _ in range(self.nrobots)]
        self.current_mode = 'start'  # 'start' or 'end'
        self.current_robot_idx = 0
        self.last_planned_idx: Optional[int] = None  # only show alts for this robot
        self.highlighted_candidate_idx: Optional[int] = None  # row index in candidates[last_planned_idx]
        self.last_costmap = None  # latest generated cost-map (np.ndarray)

        # load L2R if available
        self.l2r = L2RRanker("ranker_model.joblib")
        # load weight predictor if available
        self.wp = None
        if WeightPredictor is not None:
            try:
                self.wp = WeightPredictor("weight_model.joblib")
            except Exception:
                self.wp = None

        # load learned/analytic cost-map engine if available
        self.cmap_engine = None
        if CostMapEngine is not None:
            try:
                self.cmap_engine = CostMapEngine()
            except Exception:
                self.cmap_engine = None

        # connections
        self.load_btn.clicked.connect(self.on_load)
        self.nrobots_spin.valueChanged.connect(self.on_nrobots_changed)
        self.set_start_btn.clicked.connect(self.on_mode_start)
        self.set_end_btn.clicked.connect(self.on_mode_end)
        self.undo_btn.clicked.connect(self.on_undo_last)
        self.clear_btn.clicked.connect(self.on_clear)
        self.show_alts_chk.toggled.connect(self.on_toggle_alts)
        self.grid_view.cellClicked.connect(self.on_cell_clicked)
        self.weight_mode_box.currentIndexChanged.connect(lambda _=None: None)
        self.cost_table.cellClicked.connect(self.on_cost_row_clicked)
        self.reset_highlight_btn.clicked.connect(self.on_reset_highlight)
        self.show_arrows_chk.toggled.connect(lambda _=None: self.redraw_all())
        self.use_costmap_chk.toggled.connect(lambda _=None: None)
        self.use_learned_costmap_chk.toggled.connect(lambda _=None: None)
        self.show_costmap_chk.toggled.connect(lambda _=None: self.redraw_all())
        self.show_costmap_tab_chk.toggled.connect(self.on_toggle_costmap_tab)
        # zoom
        self.zoom_in_btn.clicked.connect(self.grid_view.zoom_in)
        self.zoom_out_btn.clicked.connect(self.grid_view.zoom_out)
        self.zoom_reset_btn.clicked.connect(self.grid_view.reset_zoom)
        # Component toggles
        self.comp_len.toggled.connect(self.on_components_changed)
        self.comp_turn.toggled.connect(self.on_components_changed)
        self.comp_cell.toggled.connect(self.on_components_changed)
        self.comp_path.toggled.connect(self.on_components_changed)
        self.comp_edge.toggled.connect(self.on_components_changed)
        self.comp_h2h.toggled.connect(self.on_components_changed)
        self.comp_dead.toggled.connect(self.on_components_changed)
        self.comp_self.toggled.connect(self.on_components_changed)

        self.update_mode_label()
        # preload demo map
        self.grid = demo_map()
        self.grid_view.load_map(self.grid)
        self.costmap_view.load_map(self.grid)
        # Initially disable cost-map tab until user opts in
        self.views.setTabEnabled(self.views_tab_idx_cost, False)

    # ---- UI helpers ----
    def _weights(self) -> ScoreWeights:
        # Read from spinboxes, then apply component mask (disabled -> weight=0)
        w = ScoreWeights(
            w_len=self.wlen.value(), w_turn=self.wturn.value(),
            w_overlap_cell=self.wcell.value(), w_overlap_edge=self.wedge.value(),
            w_path_overlap=self.wpath.value(),
            w_h2h=self.wh2h.value(), w_deadlock=self.wdead.value(), w_self_cycle=self.wself.value(),
        )
        return self._mask_weights(w)

    def _disabled_components(self) -> Set[str]:
        disabled: Set[str] = set()
        if not self.comp_len.isChecked():
            disabled.add("length")
        if not self.comp_turn.isChecked():
            disabled.add("turns")
        if not self.comp_cell.isChecked():
            disabled.add("cell_overlap")
        if not self.comp_path.isChecked():
            disabled.add("path_overlap")
        if not self.comp_edge.isChecked():
            disabled.add("edge_overlap")
        if not self.comp_h2h.isChecked():
            disabled.add("h2h")
        if not self.comp_dead.isChecked():
            disabled.add("deadlock")
        if not self.comp_self.isChecked():
            disabled.add("self_cycle")
        return disabled

    def _mask_weights(self, w: ScoreWeights) -> ScoreWeights:
        d = self._disabled_components()
        return ScoreWeights(
            w_len=(0.0 if "length" in d else w.w_len),
            w_turn=(0.0 if "turns" in d else w.w_turn),
            w_overlap_cell=(0.0 if "cell_overlap" in d else w.w_overlap_cell),
            w_path_overlap=(0.0 if "path_overlap" in d else w.w_path_overlap),
            w_overlap_edge=(0.0 if "edge_overlap" in d else w.w_overlap_edge),
            w_h2h=(0.0 if "h2h" in d else w.w_h2h),
            w_deadlock=(0.0 if "deadlock" in d else w.w_deadlock),
            w_self_cycle=(0.0 if "self_cycle" in d else w.w_self_cycle),
        )

    def msg(self, text: str):
        QtWidgets.QMessageBox.information(self, "Info", text)

    def update_mode_label(self):
        i = self.current_robot_idx + 1
        if self.current_robot_idx >= self.nrobots:
            self.mode_label.setText("All robots planned. Use Undo or Clear to modify.")
            return
        mode = "Start" if self.current_mode == 'start' else "End"
        self.mode_label.setText(f"Click grid to set {mode} for Robot {i}")

    def on_toggle_alts(self, _checked: bool):
        self.redraw_all()

    def on_components_changed(self, _checked: bool = False):
        # Reflect toggles on spinboxes' enabled state
        self.wlen.setEnabled(self.comp_len.isChecked())
        self.wturn.setEnabled(self.comp_turn.isChecked())
        self.wcell.setEnabled(self.comp_cell.isChecked())
        self.wpath.setEnabled(self.comp_path.isChecked())
        self.wedge.setEnabled(self.comp_edge.isChecked())
        self.wh2h.setEnabled(self.comp_h2h.isChecked())
        self.wdead.setEnabled(self.comp_dead.isChecked())
        self.wself.setEnabled(self.comp_self.isChecked())
        # If a robot has been planned, re-evaluate with current settings
        if self.last_planned_idx is not None and self.candidates[self.last_planned_idx]:
            self.plan_robot(self.last_planned_idx)
        else:
            self.update_cost_table()
            self.redraw_all()

    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Map", filter="Text files (*.txt);;YAML files (*.yaml *.yml)")
        if not path:
            return
        try:
            if path.lower().endswith(('.yaml', '.yml')):
                # Build grid from vertices (one vertex = one cell). Obstacles from labeled vertices.
                self.grid = MapGrid.from_yaml(path)
                # Dump a TXT-like representation to console for verification
                try:
                    print("\nYAML grid (0=free, 1=obstacle):\n" + self.grid.to_txt_string() + "\n")
                except Exception:
                    pass
            else:
                self.grid = MapGrid.from_txt(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load map: {e}")
            return
        self.grid_view.load_map(self.grid)
        self.costmap_view.load_map(self.grid)
        self.on_clear()

    def on_nrobots_changed(self, val: int):
        self.nrobots = val
        self.starts = [None]*val
        self.goals  = [None]*val
        self.paths  = [[] for _ in range(val)]
        self.candidates = [[] for _ in range(val)]
        self.candidate_costs = [[] for _ in range(val)]
        self.candidate_pred_costs = [[] for _ in range(val)]
        self.applied_weights = [None for _ in range(val)]
        self.current_robot_idx = 0
        self.current_mode = 'start'
        self.last_planned_idx = None
        self.highlighted_candidate_idx = None
        self.grid_view.reset_visuals()
        self.update_mode_label()
        self.clear_cost_table()

    def on_mode_start(self):
        if self.current_robot_idx < self.nrobots:
            self.current_mode = 'start'
            self.update_mode_label()

    def on_mode_end(self):
        if self.current_robot_idx < self.nrobots:
            self.current_mode = 'end'
            self.update_mode_label()

    def on_undo_last(self):
        if self.current_robot_idx == 0 and not any(self.paths):
            return
        if self.current_robot_idx >= self.nrobots:
            # all done; step back to last robot
            self.current_robot_idx = self.nrobots - 1
        idx = self.current_robot_idx
        # clear current robot's selections
        self.paths[idx] = []
        self.candidates[idx] = []
        self.candidate_costs[idx] = []
        self.candidate_pred_costs[idx] = []
        self.highlighted_candidate_idx = None
        # clear applied weights for this robot
        if hasattr(self, 'applied_weights'):
            if 0 <= idx < len(self.applied_weights):
                self.applied_weights[idx] = None
        self.goals[idx] = None
        self.starts[idx] = None
        self.current_mode = 'start'
        # recompute last_planned_idx = last robot < idx with a path
        self.last_planned_idx = None
        for j in range(idx-1, -1, -1):
            if self.paths[j]:
                self.last_planned_idx = j
                break
        # backtrack further if necessary
        while idx > 0 and not self.paths[idx-1] and self.starts[idx-1] is None and self.goals[idx-1] is None:
            idx -= 1
            self.current_robot_idx = idx
        self.redraw_all()
        self.update_mode_label()
        self.update_cost_table()

    def on_clear(self):
        if self.grid:
            self.grid_view.reset_visuals()
            self.costmap_view.reset_visuals()
        self.starts = [None]*self.nrobots
        self.goals  = [None]*self.nrobots
        self.paths  = [[] for _ in range(self.nrobots)]
        self.candidates = [[] for _ in range(self.nrobots)]
        self.candidate_costs = [[] for _ in range(self.nrobots)]
        self.candidate_pred_costs = [[] for _ in range(self.nrobots)]
        self.applied_weights = [None for _ in range(self.nrobots)]
        self.current_robot_idx = 0
        self.current_mode = 'start'
        self.last_planned_idx = None
        self.highlighted_candidate_idx = None
        self.last_costmap = None
        self.redraw_all()
        self.update_mode_label()
        self.clear_cost_table()

    def on_toggle_costmap_tab(self, checked: bool):
        # Enable/disable the cost-map tab; refresh content on enable
        try:
            self.views.setTabEnabled(self.views_tab_idx_cost, bool(checked))
            if checked and self.grid is not None:
                self.refresh_costmap_tab()
                # switch to the cost-map tab if user just enabled it
                self.views.setCurrentIndex(self.views_tab_idx_cost)
        except Exception:
            pass

    def refresh_costmap_tab(self):
        if not self.grid:
            return
        # Repaint the cost-map view based on stored last_costmap
        self.costmap_view.load_map(self.grid)
        if getattr(self, 'last_costmap', None) is not None and self.show_costmap_tab_chk.isChecked():
            self.costmap_view.draw_costmap_overlay(self.last_costmap)

    def on_cell_clicked(self, r: int, c: int):
        if not self.grid or self.current_robot_idx >= self.nrobots:
            return
        # If this grid came from a sparse YAML vertex set, restrict selection
        # to only cells that correspond to actual vertices. Filler cells (no
        # vertex) are obstacles and cannot be Start/End.
        if hasattr(self.grid, '_vertex_cells'):
            vset = getattr(self.grid, '_vertex_cells')
            if (r, c) not in vset:
                self.msg("Please select a vertex cell from the YAML map.")
                return
        idx = self.current_robot_idx
        color = self.COLORS[idx % len(self.COLORS)]

        if self.current_mode == 'start':
            self.starts[idx] = (r, c)
            self.grid_view.draw_start_marker(r, c, color)
            self.current_mode = 'end'
        else:  # picking end → immediately plan for this robot
            self.goals[idx] = (r, c)
            self.grid_view.draw_end_marker(r, c, color)
            self.plan_robot(idx)
        self.update_mode_label()

    # ---- Candidate cost table helpers ----
    def clear_cost_table(self):
        self.cost_table.setRowCount(0)
        # Also clear selection highlight
        if hasattr(self, 'highlighted_candidate_idx'):
            self.highlighted_candidate_idx = None

    def update_cost_table(self):
        if self.last_planned_idx is None:
            self.clear_cost_table()
            return
        idx = self.last_planned_idx
        details = self.candidate_costs[idx]
        cands = self.candidates[idx]
        preds = self.candidate_pred_costs[idx]
        if not details or not cands:
            self.clear_cost_table()
            return
        # Determine chosen path index
        chosen = self.paths[idx]
        chosen_i = 0
        for i, p in enumerate(cands):
            if p == chosen:
                chosen_i = i
                break
        self.cost_table.setRowCount(len(details))
        for row, ((comp, cost), path) in enumerate(zip(details, cands)):
            pred_str = "-"
            if preds and row < len(preds) and preds[row] is not None:
                pred_str = f"{preds[row]:.3f}"
            cells = [
                QtWidgets.QTableWidgetItem(str(row+1)),
                QtWidgets.QTableWidgetItem(str(int(comp['length']))),
                QtWidgets.QTableWidgetItem(str(int(comp['turns']))),
                QtWidgets.QTableWidgetItem(str(int(comp['cell_overlap']))),
                QtWidgets.QTableWidgetItem(str(int(comp['path_overlap']))),
                QtWidgets.QTableWidgetItem(str(int(comp['edge_overlap']))),
                QtWidgets.QTableWidgetItem(str(int(comp['h2h']))),
                QtWidgets.QTableWidgetItem(str(int(comp['deadlock']))),
                QtWidgets.QTableWidgetItem(str(int(comp['self_cycle']))),
                QtWidgets.QTableWidgetItem(f"{cost:.2f}"),
                QtWidgets.QTableWidgetItem(pred_str),
                QtWidgets.QTableWidgetItem("✓" if row == chosen_i else ""),
            ]
            for col, item in enumerate(cells):
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.cost_table.setItem(row, col, item)
            # Tooltip with full formula (use applied weights if available)
            w = None
            if hasattr(self, 'applied_weights') and 0 <= idx < len(self.applied_weights):
                w = self.applied_weights[idx]
            if w is None:
                w = self._weights()
            contrib = (
                f"Cost = w_len({w.w_len})*len({int(comp['length'])}) + "
                f"w_turn({w.w_turn})*turns({int(comp['turns'])}) + "
                f"w_cell({w.w_overlap_cell})*cellOv({int(comp['cell_overlap'])}) + "
                f"w_path({w.w_path_overlap})*pathOv({int(comp['path_overlap'])}) + "
                f"w_edge({w.w_overlap_edge})*edgeOv({int(comp['edge_overlap'])}) + "
                f"w_h2h({w.w_h2h})*h2h({int(comp['h2h'])}) + "
                f"w_dead({w.w_deadlock})*dead({int(comp['deadlock'])}) + "
                f"w_self({w.w_self_cycle})*self({int(comp['self_cycle'])}) = {cost:.2f}"
            )
            for col in range(self.cost_table.columnCount()):
                self.cost_table.item(row, col).setToolTip(contrib)
        self.cost_table.resizeColumnsToContents()
        # Reflect highlighted selection in the table (default to chosen if none)
        try:
            hi = self.highlighted_candidate_idx
            if hi is None or not (0 <= hi < len(cands)):
                hi = chosen_i
            self.cost_table.clearSelection()
            self.cost_table.selectRow(hi)
        except Exception:
            pass

    def on_cost_row_clicked(self, row: int, _col: int):
        if self.last_planned_idx is None:
            return
        idx = self.last_planned_idx
        if 0 <= row < len(self.candidates[idx]):
            self.highlighted_candidate_idx = row
            self.redraw_all()

    def on_reset_highlight(self):
        # Reset to the selected best path (the one in self.paths)
        if self.last_planned_idx is None:
            return
        idx = self.last_planned_idx
        chosen = self.paths[idx]
        hi = 0
        for i, p in enumerate(self.candidates[idx]):
            if p == chosen:
                hi = i
                break
        self.highlighted_candidate_idx = hi
        self.update_cost_table()
        self.redraw_all()

    def print_costs_to_console(self, idx: int):
        details = self.candidate_costs[idx]
        cands = self.candidates[idx]
        preds = self.candidate_pred_costs[idx]
        if not details or not cands:
            return
        # Reflect weights used for this robot
        w = None
        if hasattr(self, 'applied_weights') and 0 <= idx < len(self.applied_weights):
            w = self.applied_weights[idx]
        if w is None:
            w = self._weights()
        print(f"\nRobot {idx+1} — Candidate cost breakdown (K={len(cands)}):")
        for i, ((comp, cost), path) in enumerate(zip(details, cands), start=1):
            parts = [
                f"w_len*len={w.w_len}*{int(comp['length'])}={w.w_len*comp['length']:.2f}",
                f"w_turn*turns={w.w_turn}*{int(comp['turns'])}={w.w_turn*comp['turns']:.2f}",
                f"w_cell*cellOv={w.w_overlap_cell}*{int(comp['cell_overlap'])}={w.w_overlap_cell*comp['cell_overlap']:.2f}",
                f"w_path*pathOv={w.w_path_overlap}*{int(comp['path_overlap'])}={w.w_path_overlap*comp['path_overlap']:.2f}",
                f"w_edge*edgeOv={w.w_overlap_edge}*{int(comp['edge_overlap'])}={w.w_overlap_edge*comp['edge_overlap']:.2f}",
                f"w_h2h*h2h={w.w_h2h}*{int(comp['h2h'])}={w.w_h2h*comp['h2h']:.2f}",
                f"w_dead*dead={w.w_deadlock}*{int(comp['deadlock'])}={w.w_deadlock*comp['deadlock']:.2f}",
                f"w_self*self={w.w_self_cycle}*{int(comp['self_cycle'])}={w.w_self_cycle*comp['self_cycle']:.2f}",
            ]
            pc = f", L2R_pred={preds[i-1]:.3f}" if preds and preds[i-1] is not None else ""
            print(f"  {i:02d}. cost={cost:.2f}{pc}  [" + ", ".join(parts) + "]")

    # ---- Drawing helpers ----
    def redraw_all(self):
        if not self.grid:
            return
        self.grid_view.reset_visuals()
        # Draw arrows for one-way lanes if available and enabled
        if hasattr(self, 'show_arrows_chk') and self.show_arrows_chk.isChecked():
            arrows = getattr(self.grid, '_oneway_arrows', None)
            if arrows:
                arrow_color = QtGui.QColor(100, 100, 100, 180)
                for u, v in arrows:
                    self.grid_view.draw_arrow(u, v, arrow_color)
        show_alts = self.show_alts_chk.isChecked()
        alt_idx: Optional[int] = self.last_planned_idx if show_alts else None
        # Determine highlighted candidate path (only for last planned robot)
        hi_path: Optional[Path] = None
        if self.last_planned_idx is not None and self.candidates[self.last_planned_idx]:
            hi = self.highlighted_candidate_idx
            if hi is not None and 0 <= hi < len(self.candidates[self.last_planned_idx]):
                hi_path = self.candidates[self.last_planned_idx][hi]
        for i in range(self.nrobots):
            color = self.COLORS[i % len(self.COLORS)]
            if self.starts[i]:
                r, c = self.starts[i]
                self.grid_view.draw_start_marker(r, c, color)
            if self.goals[i]:
                r, c = self.goals[i]
                self.grid_view.draw_end_marker(r, c, color)
            # draw planned path as thick colored line
            if self.paths[i]:
                # If user highlighted a different candidate for the most recent robot,
                # render the committed (best) path as dashed colored to de-emphasize.
                if i == self.last_planned_idx and hi_path is not None and hi_path != self.paths[i]:
                    self.grid_view.draw_dashed_colored(self.paths[i], color)
                else:
                    self.grid_view.draw_planned(self.paths[i], color)
            # only draw alternatives for the most recently planned robot
            if alt_idx is not None and i == alt_idx and self.candidates[i]:
                chosen = self.paths[i]
                for cand in self.candidates[i]:
                    if cand and cand != chosen:
                        self.grid_view.draw_candidate(cand)
        # Optional cost-map overlay (beneath path overlays)
        if hasattr(self, 'show_costmap_chk') and self.show_costmap_chk.isChecked():
            if getattr(self, 'last_costmap', None) is not None:
                self.grid_view.draw_costmap_overlay(self.last_costmap)

        # On top, draw the highlighted candidate (dashed colored) if it differs from chosen
        if self.last_planned_idx is not None and self.candidates[self.last_planned_idx]:
            hi = self.highlighted_candidate_idx
            if hi is not None and 0 <= hi < len(self.candidates[self.last_planned_idx]):
                cand_path = self.candidates[self.last_planned_idx][hi]
                if cand_path != self.paths[self.last_planned_idx]:
                    color = self.COLORS[self.last_planned_idx % len(self.COLORS)]
                    self.grid_view.draw_highlight(cand_path, color)
                # Draw conflict markers for the selected candidate vs committed paths
                committed = [p for j, p in enumerate(self.paths) if j < self.last_planned_idx and p]
                if committed:
                    # align paths for conflict extraction
                    cand_aligned, others_aligned = _align_paths(cand_path, committed)
                    path_ov, h2h_cells, dead_cells = _conflict_cells_for_candidate(cand_aligned, others_aligned)
                    # Colors per conflict type
                    col_path = QtGui.QColor(255, 215, 0)   # gold circle for path overlap
                    col_h2h  = QtGui.QColor(255, 140, 0)   # dark orange X for head-to-head
                    col_dead = QtGui.QColor(186, 85, 211)  # orchid diamond for deadlock
                    for rc in path_ov:
                        self.grid_view.draw_conflict_circle(rc, col_path)
                    for rc in h2h_cells:
                        self.grid_view.draw_conflict_cross(rc, col_h2h)
                    for rc in dead_cells:
                        self.grid_view.draw_conflict_diamond(rc, col_dead)

    # ---- Planning ----
    def plan_robot(self, idx: int):
        if not self.grid or not self.starts[idx] or not self.goals[idx]:
            return
        K = self.k_spin.value()
        weights = self._weights()
        # committed paths from previously planned robots
        committed = [p for j, p in enumerate(self.paths) if j < idx and p]
        # generate candidates (allow entering/leaving racks only at endpoints)
        allow: Set[Coord] = set()
        s_r, s_c = self.starts[idx]
        g_r, g_c = self.goals[idx]
        if self.grid.grid[s_r][s_c] == 1:
            allow.add(self.starts[idx])
        if self.grid.grid[g_r][g_c] == 1:
            allow.add(self.goals[idx])
        # Build optional cost-map(s) via engine
        cost_map = None
        mode_str = "None"
        avg = None
        prefer_learned = bool(self.use_learned_costmap_chk.isChecked()) if hasattr(self, 'use_learned_costmap_chk') else False
        allow_analytic = bool(self.use_costmap_chk.isChecked()) if hasattr(self, 'use_costmap_chk') else False
        if self.cmap_engine is not None:
            try:
                cost_map, mode_str, avg = self.cmap_engine.compute(self.grid, committed, self.goals[idx], prefer_learned, allow_analytic)
            except Exception:
                cost_map, mode_str, avg = None, "None", None
        elif allow_analytic and build_analytic_costmap is not None:
            try:
                cost_map = build_analytic_costmap(self.grid, committed)
                mode_str = "Analytic"
                avg = float(cost_map.mean())
            except Exception:
                cost_map, mode_str, avg = None, "None", None

        # Update UI labels and overlay state
        if mode_str == "Learned" and avg is not None:
            self.costmap_status.setText(f"learned cost-map avg={avg:.3f}")
            if hasattr(self, 'costmap_mode_label'):
                self.costmap_mode_label.setText("Cost shaping: Learned")
            self.last_costmap = cost_map
        elif mode_str == "Analytic" and avg is not None:
            self.costmap_status.setText(f"cost-map avg={avg:.3f}")
            if hasattr(self, 'costmap_mode_label'):
                self.costmap_mode_label.setText("Cost shaping: Analytic")
            self.last_costmap = cost_map
        else:
            self.costmap_status.setText("cost-map: off")
            if hasattr(self, 'costmap_mode_label'):
                self.costmap_mode_label.setText("Cost shaping: None")
            self.last_costmap = None
        # Update cost-map tab if visible
        if self.show_costmap_tab_chk.isChecked():
            self.refresh_costmap_tab()

        cands = yen_k_shortest(self.grid, self.starts[idx], self.goals[idx], K,
                               allow_occupied=allow if allow else None,
                               cost_map=cost_map, turn_penalty=None)
        if not cands:
            # Provide a more helpful reason if endpoints are unreachable from free space
            s_r, s_c = self.starts[idx]
            g_r, g_c = self.goals[idx]
            def free_nbrs(r: int, c: int) -> int:
                cnt = 0
                for rr, cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0 <= rr < self.grid.rows and 0 <= cc < self.grid.cols and self.grid.grid[rr][cc] == 0:
                        cnt += 1
                return cnt
            msg = f"No path found for Robot {idx+1}."
            if self.grid.grid[s_r][s_c] == 1 and free_nbrs(s_r, s_c) == 0:
                msg += " Start rack has no adjacent free cell."
            if self.grid.grid[g_r][g_c] == 1 and free_nbrs(g_r, g_c) == 0:
                msg += " End rack has no adjacent free cell."
            QtWidgets.QMessageBox.warning(self, "Planner", msg)
            return
        # Determine weights to use (Fixed / Learned / Blend)
        used_weights = self._weights()
        wmode = self.weight_mode_box.currentText()
        if wmode != "Fixed" and self.wp is not None and CFMap is not None and compute_context_features is not None:
            try:
                cf = compute_context_features(
                    CFMap(self.grid.grid), committed,
                    self.starts[idx], self.goals[idx], cands,
                    astar_fn=astar
                )
                pred = self.wp.predict(cf)  # dict of weights
                learned_w = ScoreWeights(**pred)
                if wmode == "Learned":
                    used_weights = learned_w
                else:  # Blend 50/50
                    used_weights = ScoreWeights(
                        w_len = 0.5*(used_weights.w_len + learned_w.w_len),
                        w_turn = 0.5*(used_weights.w_turn + learned_w.w_turn),
                        w_overlap_cell = 0.5*(used_weights.w_overlap_cell + learned_w.w_overlap_cell),
                        w_overlap_edge = 0.5*(used_weights.w_overlap_edge + learned_w.w_overlap_edge),
                        w_h2h = 0.5*(used_weights.w_h2h + learned_w.w_h2h),
                        w_deadlock = 0.5*(used_weights.w_deadlock + learned_w.w_deadlock),
                        w_self_cycle = 0.5*(used_weights.w_self_cycle + learned_w.w_self_cycle),
                    )
            except Exception:
                # fall back silently to fixed if compute/predict fails
                pass
        # Mask disabled components regardless of weight source
        used_weights = self._mask_weights(used_weights)
        # compute detailed costs for table/console (heuristic components)
        details: List[Tuple[Dict[str, float], float]] = []
        for cand in cands:
            comp, cost = cost_components(cand, committed, used_weights)
            details.append((comp, cost))
        self.candidate_costs[idx] = details
        # remember applied weights for this robot (for tooltips/console)
        if hasattr(self, 'applied_weights'):
            if 0 <= idx < len(self.applied_weights):
                self.applied_weights[idx] = used_weights

        # optional learned scores (neg predicted cost)
        pred_scores: Optional[List[float]] = None
        mode = self.scorer_box.currentText()
        if mode in ("Learned", "Blend 50/50") and self.l2r and self.l2r.available():
            pred_scores = self.l2r.predict_neg_cost_scores(cands, committed, disabled=self._disabled_components())
            # store predicted scores directly (higher is better)
            self.candidate_pred_costs[idx] = [(float(s) if s != float('-inf') else None) for s in pred_scores]
        else:
            self.candidate_pred_costs[idx] = [None]*len(cands)

        # choose best
        if mode == "Learned" and pred_scores is not None:
            best_i = max(range(len(cands)), key=lambda i: pred_scores[i])
        elif mode == "Blend 50/50" and pred_scores is not None:
            # normalize both channels and average
            heur_scores = [ -cost for (_, cost) in details ]  # higher better
            def norm(v):
                lo, hi = min(v), max(v)
                return [(x - lo) / (hi - lo + 1e-9) for x in v]
            blend = [0.5*a + 0.5*b for a,b in zip(norm(heur_scores), norm(pred_scores))]
            best_i = max(range(len(cands)), key=lambda i: blend[i])
        else:
            # heuristic fallback
            best_i = max(range(len(cands)), key=lambda i: -details[i][1])  # lowest cost

        best = cands[best_i]
        self.paths[idx] = best
        self.candidates[idx] = cands
        self.last_planned_idx = idx  # show alts only for this robot
        # Default highlight: the selected best path
        self.highlighted_candidate_idx = best_i

        # draw & tables & console
        self.redraw_all()
        self.update_cost_table()
        self.print_costs_to_console(idx)

        # advance to next robot or finish
        if idx < self.nrobots - 1:
            self.current_robot_idx = idx + 1
            self.current_mode = 'start'
        else:
            self.current_robot_idx = self.nrobots
            self.current_mode = 'done'
            # simple stats
            total_len = sum(max(0, len(p)-1) for p in self.paths)
            total_turns = sum(count_turns(p) for p in self.paths)
            cell_sets = [set(p) for p in self.paths]
            cell_overlap = 0
            for i in range(len(cell_sets)):
                for j in range(i+1, len(cell_sets)):
                    cell_overlap += len(cell_sets[i] & cell_sets[j])
            QtWidgets.QMessageBox.information(self, "Result", (
                f"""Planned {len(self.paths)} robots.
                Total length: {total_len}
                Total turns:  {total_turns}
                Cell overlaps (pairwise sum): {cell_overlap}"""
            ))

# ==============
# Entry Point
# ==============

def demo_map() -> MapGrid:
    txt = [
        "11111111111111111111",
        "10000000000000000001",
        "10111011011011011101",
        "10000000000000000001",
        "11111111111111111111",
    ]
    grid = [[1 if ch=='1' else 0 for ch in line] for line in txt]
    return MapGrid(grid)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
