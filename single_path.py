import os
import heapq
import random
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def pick_start_goal(
    grid: np.ndarray, attempts: int = 100
) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    """Pick two free cells with a valid connecting path."""
    free = [(int(x), int(y)) for y, x in np.argwhere(grid == 0)]
    if len(free) < 2:
        raise ValueError("Map must contain at least two free cells")
    for _ in range(attempts):
        start, goal = random.sample(free, 2)
        path = astar(start, goal, grid)
        if path:
            return start, goal, path
    raise RuntimeError("Failed to find valid start and goal after" f" {attempts} attempts")


def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]]) -> None:
    """Animate the robot moving along ``path`` on ``grid`` using Matplotlib."""
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="gray_r", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, "b--", linewidth=1, alpha=0.5)

    robot, = ax.plot([], [], "ro", markersize=5)

    def init():
        robot.set_data([], [])
        return (robot,)

    def update(frame: int):
        robot.set_data(path_x[frame], path_y[frame])
        return (robot,)

    FuncAnimation(
        fig, update, frames=len(path), init_func=init, interval=300, blit=True, repeat=False
    )
    plt.show()


if __name__ == "__main__":
    start, goal, path = pick_start_goal(obstacle_map)
    print(f"Found path of length {len(path) - 1} between {start} and {goal}")
    visualize_path(obstacle_map, path)
