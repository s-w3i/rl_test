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


def pick_agent_paths(
    grid: np.ndarray, num_agents: int, attempts: int = 1000
) -> List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]]:
    """Pick ``num_agents`` start/goal pairs with valid connecting paths.

    Cells are chosen from the free space of ``grid``.  Start and goal locations
    for different agents are guaranteed to be distinct.
    """
    free = [(int(x), int(y)) for y, x in np.argwhere(grid == 0)]
    if len(free) < num_agents * 2:
        raise ValueError("Map must contain at least two free cells per agent")

    paths: List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]] = []
    used = set()
    for _ in range(attempts):
        if len(paths) == num_agents:
            return paths
        start, goal = random.sample(free, 2)
        if start in used or goal in used or start == goal:
            continue
        path = astar(start, goal, grid)
        if path:
            paths.append((start, goal, path))
            used.update({start, goal})

    raise RuntimeError(
        f"Failed to find paths for {num_agents} agents after {attempts} attempts"
    )


def visualize_paths(
    grid: np.ndarray,
    agents: List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]],
) -> None:
    """Animate multiple agents moving along their paths on ``grid``."""
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="gray_r", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    markers = []
    max_len = 0
    for idx, (start, goal, path) in enumerate(agents):
        color = colors[idx % len(colors)]
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, linestyle="--", color=color, linewidth=1, alpha=0.7)
        ax.plot(start[0], start[1], marker="o", color=color, markersize=4)
        ax.plot(goal[0], goal[1], marker="x", color=color, markersize=4)
        marker, = ax.plot([], [], marker="o", color=color, markersize=5)
        markers.append((marker, path))
        max_len = max(max_len, len(path))

    def init():
        for marker, _ in markers:
            marker.set_data([], [])
        return tuple(m for m, _ in markers)

    def update(frame: int):
        for marker, path in markers:
            step = min(frame, len(path) - 1)
            marker.set_data(path[step][0], path[step][1])
        return tuple(m for m, _ in markers)

    FuncAnimation(
        fig,
        update,
        frames=max_len,
        init_func=init,
        interval=300,
        blit=True,
        repeat=False,
    )
    plt.show()


if __name__ == "__main__":
    agents = pick_agent_paths(obstacle_map, num_agents=6)
    for i, (start, goal, path) in enumerate(agents, 1):
        print(
            f"Agent {i}: path length {len(path) - 1} between {start} and {goal}"
        )
    visualize_paths(obstacle_map, agents)
