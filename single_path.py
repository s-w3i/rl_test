
import argparse
import heapq
import matplotlib.pyplot as plt
from matplotlib import animation


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    width, height = len(grid[0]), len(grid)
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
                next_pos = (nx, ny)
                if next_pos in visited:
                    continue
                new_path = path + [next_pos]
                new_g = g + 1
                new_f = new_g + manhattan(next_pos, goal)
                heapq.heappush(open_set, (new_f, new_g, next_pos, new_path))
    return None


def build_grid(width, height):
    grid = [[0 for _ in range(width)] for _ in range(height)]
    # Vertical wall with a gap at y=5
    for y in range(height):
        if y != 5:
            grid[y][4] = 1
    return grid


def plan_paths(grid, agents):
    for agent in agents:
        path = astar(grid, agent['start'], agent['goal'])
        if path is None:
            raise RuntimeError(f"No path for agent from {agent['start']} to {agent['goal']}")
        agent['path'] = path
    return agents


def animate(grid, agents, save_path=None):
    width, height = len(grid[0]), len(grid)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True)
    ax.set_aspect('equal')

    # draw obstacles
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))

    # draw paths and setup agents
    patches = []
    max_len = 0
    for idx, agent in enumerate(agents):
        path = agent['path']
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linestyle='--', color=colors[idx], alpha=0.5)
        patch, = ax.plot([], [], marker='o', color=colors[idx], markersize=8)
        patches.append(patch)
        if len(path) > max_len:
            max_len = len(path)

    def init():
        for patch in patches:
            patch.set_data([], [])
        return patches

    def update(frame):
        for i, agent in enumerate(agents):
            path = agent['path']
            if frame < len(path):
                x, y = path[frame]
            else:
                x, y = path[-1]
            patches[i].set_data([x], [y])
        return patches

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=max_len, interval=500, blit=True)

    if save_path:
        anim.save(save_path, writer='pillow')
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Multi-agent grid simulation')
    parser.add_argument('--save', type=str, help='Path to save animation (GIF)')
    args = parser.parse_args()

    width = height = 10
    grid = build_grid(width, height)

    agents = [
        {'start': (0, 0), 'goal': (9, 9)},
        {'start': (0, 9), 'goal': (9, 0)},
        {'start': (9, 0), 'goal': (0, 9)},
        {'start': (9, 9), 'goal': (0, 0)},
        {'start': (0, 5), 'goal': (9, 5)},
        {'start': (5, 0), 'goal': (5, 9)},
    ]

    plan_paths(grid, agents)
    animate(grid, agents, save_path=args.save)


if __name__ == '__main__':
    main()