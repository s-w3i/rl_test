import os
import heapq
from typing import Tuple, List

import numpy as np


def load_obstacle_map(map_path: str) -> np.ndarray:
    """Load an obstacle map from ``map_path``.

    The file should contain whitespace separated ``0``s and ``1``s, one row per
    line.
    """
    path = map_path
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    try:
        return np.loadtxt(path, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - logging
        raise RuntimeError(f"Failed to load map from {path}: {exc}")


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic for A*"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start: Tuple[int, int], goal: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
    """Simple A* path search on a grid map."""
    rows, cols = grid.shape
    open_list: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start))
    came_from = {start: None}
    g_scores = {start: 0}

    while open_list:
        _, g, current = heapq.heappop(open_list)
        if current == goal:
            path = [current]
            while came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if grid[ny, nx] == 1:
                continue
            tentative_g = g + 1
            neighbor = (nx, ny)
            if tentative_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_list, (f, tentative_g, neighbor))
    return []


MAP_FILE = os.environ.get("MAP_FILE", "map.txt")
obstacle_map = load_obstacle_map(MAP_FILE)


if __name__ == "__main__":
    start = (1, 1)
    goal = (obstacle_map.shape[1] - 2, obstacle_map.shape[0] - 2)
    path = astar(start, goal, obstacle_map)
    if path:
        print("Found path of length", len(path) - 1)
    else:
        print("No path found between", start, "and", goal)
