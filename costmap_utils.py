from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # only for type hints; avoid runtime import cycles
    from mrpp_selector import MapGrid, Path, Coord  # pragma: no cover


def build_analytic_costmap(grid: 'MapGrid', committed_paths: List['Path']) -> np.ndarray:
    """Construct a lightweight analytic cost-map.
    - Base zeros
    - +0.8 in cells with free-degree ≤ 2 (narrow corridors)
    - +1.2 near committed paths (Chebyshev distance ≤ 1 from any committed cell)
    - Clipped to [0, 3]
    Returns a numpy array with shape (rows, cols).
    """
    rows, cols = grid.rows, grid.cols
    cm = np.zeros((rows, cols), dtype=float)

    # Narrow corridors: free-degree ≤ 2
    for r in range(rows):
        for c in range(cols):
            if grid.grid[r][c] != 0:
                continue
            deg = 0
            for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= rr < rows and 0 <= cc < cols and grid.grid[rr][cc] == 0:
                    deg += 1
            if deg <= 2:
                cm[r, c] += 0.8

    # Proximity to committed paths: Chebyshev distance ≤ 1 (includes 8-neighborhood)
    if committed_paths:
        for p in committed_paths:
            for (r, c) in p:
                for rr in range(r-1, r+2):
                    for cc in range(c-1, c+2):
                        if 0 <= rr < rows and 0 <= cc < cols:
                            cm[rr, cc] += 1.2

    # Clip to [0, 3]
    np.clip(cm, 0.0, 3.0, out=cm)
    return cm


class CostMapEngine:
    """Helper to compute learned/analytic cost-maps with graceful fallback."""

    def __init__(self, weight_path: str = "costmap_unet.pt"):
        self.model = None
        self._torch = None
        try:
            import torch  # type: ignore
            self._torch = torch
            from costmap_model import load_costmap_model  # type: ignore
            self.model = load_costmap_model(weight_path)
        except Exception:
            self.model = None
            self._torch = None

    def compute(
        self,
        grid: 'MapGrid',
        committed_paths: List['Path'],
        goal: 'Coord',
        prefer_learned: bool = True,
        allow_analytic: bool = True,
    ) -> Tuple[Optional[np.ndarray], str, Optional[float]]:
        """Return (cost_map, mode_str, avg) where mode_str in {None, Analytic, Learned}.
        avg is None when cost_map is None.
        """
        # Try learned first if requested and model present
        if prefer_learned and self.model is not None and self._torch is not None:
            try:
                from costmap_features import build_costmap_inputs  # type: ignore
                from costmap_model import postprocess_cost  # type: ignore
                inp = build_costmap_inputs(grid, committed_paths, goal)  # (1,4,H,W)
                with self._torch.no_grad():
                    t = self._torch.from_numpy(inp)
                    logits = self.model(t)  # type: ignore[misc]
                cost_map = postprocess_cost(logits)
                return cost_map, "Learned", float(cost_map.mean())
            except Exception:
                pass

        # Analytic fallback
        if allow_analytic:
            try:
                cm = build_analytic_costmap(grid, committed_paths)
                return cm, "Analytic", float(cm.mean())
            except Exception:
                pass

        return None, "None", None

