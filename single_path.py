
import argparse
import heapq
import os
import random
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


def load_grid(path):
    """Read a grid map from a text file.

    The file should contain rows of integers separated by spaces. ``0``
    represents a free cell and ``1`` an obstacle.
    """
    with open(path) as f:
        return [list(map(int, line.split())) for line in f if line.strip()]


def random_agents(grid, count):
    """Generate ``count`` agents with random start/goal positions.

    All positions are sampled uniformly from free cells (value ``0``) and are
    unique across all starts and goals.
    """
    width, height = len(grid[0]), len(grid)
    free = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == 0]
    needed = count * 2
    if len(free) < needed:
        raise ValueError("Not enough free cells for the requested number of agents")
    random.shuffle(free)
    agents = []
    for _ in range(count):
        start = free.pop()
        goal = free.pop()
        agents.append({'start': start, 'goal': goal})
    return agents


def plan_paths(grid, agents):
    for agent in agents:
        path = astar(grid, agent['start'], agent['goal'])
        if path is None:
            raise RuntimeError(f"No path for agent from {agent['start']} to {agent['goal']}")
        agent['path'] = path
    return agents


def validate_positions(grid, agents):
    width, height = len(grid[0]), len(grid)
    for agent in agents:
        for key in ['start', 'goal']:
            x, y = agent[key]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"{key} {agent[key]} out of bounds") 
            if grid[y][x] != 0:
                raise ValueError(f"{key} {agent[key]} is on an obstacle")


def animate(grid, agents, save_path=None):
    width, height = len(grid[0]), len(grid)
    colors = plt.get_cmap('tab10', len(agents))

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.8)
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
        ax.plot(xs, ys, linestyle='--', color=colors(idx), alpha=0.5,
                label=f"Agent {idx + 1}")
        # mark start and goal positions
        sx, sy = path[0]
        gx, gy = path[-1]
        ax.plot(sx, sy, marker='^', color=colors(idx), markersize=8)
        ax.plot(gx, gy, marker='s', color=colors(idx), markersize=8)
        patch, = ax.plot([], [], marker='o', color=colors(idx), markersize=8)
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
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save_path:
        anim.save(save_path, writer='pillow')
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Multi-agent grid simulation')
    parser.add_argument('--save', type=str, help='Path to save animation (GIF)')
    parser.add_argument('--agents', type=int, default=6,
                        help='Number of agents to simulate')
    parser.add_argument('--seed', type=int, help='Optional random seed')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    map_file = os.environ.get('MAP_FILE', 'map.txt')
    grid = load_grid(map_file)

    agents = random_agents(grid, args.agents)

    validate_positions(grid, agents)
    plan_paths(grid, agents)
    animate(grid, agents, save_path=args.save)


if __name__ == '__main__':
    main()
