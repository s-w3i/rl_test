import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import time
import heapq
import pandas as pd
import psutil


# Grid Settings
GRIDSIZE = 20
OBSTACLE_RATIO = 0.5
NUM_OBSTACLE = GRIDSIZE*GRIDSIZE*OBSTACLE_RATIO
# Expanded palette for >=8 agents
AGENT_COLORS = ['blue', 'green', 'purple', 'gray', 'orange', 'brown', 'cyan', 'magenta', 'olive', 'navy']

num_agents = 8  # ⬅️ Updated: 8 agents

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create folder if not exists
output_folder = "results/astra3_fac"
os.makedirs(output_folder, exist_ok=True)

# Define the Convolutional Neural Network for Q-learning
class DQN(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 20 * 20, 256)
        self.fc2 = nn.Linear(256, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Replay Memory for Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
    def add_to_buffer(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))  # Assign highest priority initially
        
    def get_probabilities(self, priority_scale):
        # Ensure all priorities are scalar values
        priorities = np.array([p if isinstance(p, (int, float)) else 1.0 for p in self.priorities], dtype=np.float32)
        
        # Scale priorities
        scaled_priorities = priorities ** priority_scale
        
        # Normalize to get probabilities
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    
    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
        
    def sample_minibatch(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        
        # Move tensors from GPU to CPU before creating NumPy array
        samples = [self.buffer[idx] for idx in sample_indices]
        samples = [(s.cpu(), a, r, ns.cpu(), d) if isinstance(s, torch.Tensor) else (s, a, r, ns, d)
                for s, a, r, ns, d in samples]
        
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset  # Update priorities based on TD error


# Function to update the target network
def update_target_network(target_model, policy_model):
    target_model.load_state_dict(policy_model.state_dict())

def euclidean_distance(pos, goal):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

def generate_map_with_positions(obstacle_map, min_distance=10, max_attempts=10000):
    """
    Choose start/goal pairs only on free cells AND guaranteed reachable.
    - Keeps any seeded pairs that are valid and reachable.
    - Fills remaining agents with random pairs from the largest connected free region.
    """
    from collections import deque

    H, W = obstacle_map.shape

    def is_free(p):
        x, y = p
        return 0 <= x < W and 0 <= y < H and obstacle_map[y, x] == 0

    # Simple BFS reachability check (4-neighborhood)
    def reachable_bfs(start, goal):
        if start == goal:
            return True
        q = deque([start])
        seen = {start}
        while q:
            x, y = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                np_ = (nx, ny)
                if 0 <= nx < W and 0 <= ny < H and obstacle_map[ny, nx] == 0 and np_ not in seen:
                    if np_ == goal:
                        return True
                    seen.add(np_)
                    q.append(np_)
        return False

    # Label free-space connected components and keep the largest
    def largest_free_component():
        labels = [[-1] * W for _ in range(H)]
        comp_sizes = []
        comp_cells = []
        cid = 0
        for y in range(H):
            for x in range(W):
                if obstacle_map[y, x] == 0 and labels[y][x] == -1:
                    q = deque([(x, y)])
                    labels[y][x] = cid
                    cells = [(x, y)]
                    while q:
                        cx, cy = q.popleft()
                        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < W and 0 <= ny < H and obstacle_map[ny, nx] == 0 and labels[ny][nx] == -1:
                                labels[ny][nx] = cid
                                q.append((nx, ny))
                                cells.append((nx, ny))
                    comp_cells.append(cells)
                    comp_sizes.append(len(cells))
                    cid += 1
        if not comp_sizes:
            return []
        largest_idx = int(np.argmax(comp_sizes))
        return comp_cells[largest_idx]

    # Candidate pool = largest connected free region
    pool = largest_free_component()
    if not pool:
        raise RuntimeError("No free-space component found. Check your map.")

    # --- seed pairs (only keep valid & reachable) ---
    seed_starts = [(12, 15), (6, 3), (13, 1), (7, 11)]
    seed_goals  = [(18, 5), (6, 17), (6, 13), (3, 1)]

    starts, goals = [], []
    used = set()

    for s, g in zip(seed_starts, seed_goals):
        if is_free(s) and is_free(g) and s in pool and g in pool and euclidean_distance(s, g) >= min_distance and reachable_bfs(s, g):
            starts.append(s); goals.append(g)
            used.add(s); used.add(g)
        # else: silently skip invalid seeded pair (e.g., (13,1)->(6,13))

    # --- fill remaining agents with valid random pairs ---
    attempts = 0
    while len(starts) < num_agents and attempts < max_attempts:
        attempts += 1
        s, g = random.sample(pool, 2)
        if s in used or g in used:
            continue
        if euclidean_distance(s, g) < min_distance:
            continue
        if not reachable_bfs(s, g):
            continue
        starts.append(s); goals.append(g)
        used.add(s); used.add(g)

    if len(starts) < num_agents:
        raise RuntimeError(f"Could not find enough valid start/goal pairs after {max_attempts} attempts.")

    # Build a colored map for visualization
    colored_map = np.zeros((H, W, 3), dtype=np.float32)
    colored_map[obstacle_map == 0] = [1, 1, 1]  # free = white
    colored_map[obstacle_map == 1] = [0, 0, 0]  # obstacles = black
    for i in range(num_agents):
        sx, sy = starts[i]; gx, gy = goals[i]
        colored_map[sy, sx] = [0, 0, 1]  # start = blue
        colored_map[gy, gx] = [1, 0, 0]  # goal  = red

    print("Chosen starts:", starts)
    print("Chosen goals :", goals)
    return colored_map, starts, goals


# Initialize DQNs for agents
action_size = 4
input_channels = num_agents + 1  # Obstacle map + agent positions

dqn_agents = [DQN(input_channels, action_size).to(device) for _ in range(num_agents)]
target_dqn_agents = [DQN(input_channels, action_size).to(device) for _ in range(num_agents)]
optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in dqn_agents]
loss_fn = nn.MSELoss()

for i in range(num_agents):
    update_target_network(target_dqn_agents[i], dqn_agents[i])

# Initialize Replay Memory
memory = PrioritizedReplayBuffer(10000)

# Training Hyperparameters
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99  # Discount factor
batch_size = 64

# Generate environment
obstacle_map = np.array([ 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
                         ], dtype=np.float32)
  # Use the provided static map

colored_map, starts, goals = generate_map_with_positions(obstacle_map)

# Possible actions (Up, Down, Left, Right)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Visualization helpers -----------------------------------------------------
def init_visualization(positions):
    """Create an interactive plot showing the map, initial agent positions and paths."""
    plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(colored_map, origin="upper")
    ax.set_xticks(np.arange(-0.5, GRIDSIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRIDSIZE, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(num_agents)]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    scat = ax.scatter(xs, ys, c=colors, s=100)

    path_lines = []
    paths = [[p] for p in positions]
    for i in range(num_agents):
        (line,) = ax.plot([positions[i][0]], [positions[i][1]], color=colors[i], linewidth=1)
        path_lines.append(line)

    plt.show(block=False)
    return fig, ax, scat, path_lines, paths


def update_visualization(scat, positions, path_lines, paths):
    """Update the scatter plot and path lines with new agent positions."""
    scat.set_offsets(np.array(positions))
    for i, pos in enumerate(positions):
        paths[i].append(pos)
        xs, ys = zip(*paths[i])
        path_lines[i].set_data(xs, ys)
    plt.pause(0.01)

def get_state(agent_positions):
    state = np.zeros((num_agents + 1, 20, 20))
    state[0] = obstacle_map  # First channel: obstacles
    for i, pos in enumerate(agent_positions):
        state[i + 1][pos[1], pos[0]] = 1  # Agent positions
    return torch.FloatTensor(state).unsqueeze(0).to(device)

def step(agent_pos, action, goal):
    new_x = agent_pos[0] + actions[action][0]
    new_y = agent_pos[1] + actions[action][1]

    prev_distance = euclidean_distance(agent_pos, goal)
    new_distance = euclidean_distance((new_x, new_y), goal)

    if 0 <= new_x < GRIDSIZE and 0 <= new_y < GRIDSIZE:
        if obstacle_map[new_y, new_x] == 0:  # Valid move
            new_pos = (new_x, new_y)

            if new_pos == goal:
                return new_pos, 100, False # Goal reward, no obstacle collision

            distance_reward = 5 if new_distance < prev_distance else -5
            return new_pos, distance_reward, False # Normal move, no collision

    return agent_pos, -20, True # Penalize for obstacle collision


def astar(grid, start, goal):
    """Find the shortest path using A*."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse to start → goal order

        for dx, dy in actions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < GRIDSIZE and 0 <= neighbor[1] < GRIDSIZE and grid[neighbor[1], neighbor[0]] == 0:
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found


astar_paths = []
for i in range(num_agents):
    path = astar(obstacle_map, starts[i], goals[i])
    astar_paths.append(path)
print(astar_paths)


def pretrain_with_astar(dqn_agents, astar_paths, num_epochs=5):
    """Boost Q-values for A* paths over multiple epochs to reinforce learning."""
    for epoch in range(num_epochs):  # Train for multiple passes
        print(f"🔁 Pretraining Epoch {epoch+1}/{num_epochs}")

        for i in range(num_agents):
            path = astar_paths[i]
            q_values = None  # Ensure q_values is always initialized
            
            if path:
                path_length = len(path)

                for step in range(path_length - 1):
                    state = get_state([path[step] for _ in range(num_agents)])

                    # Compute action index based on movement direction
                    action = actions.index((path[step + 1][0] - path[step][0], path[step + 1][1] - path[step][1]))

                    q_values = dqn_agents[i](state)

                    # Adaptive reward scaling based on path length
                    boost_value = 10 * (1 - step / path_length)  # Higher boost for earlier steps

                    q_values[0][action] += boost_value  # Encourage A* actions

                    # Smoothed Q-target update
                    target = q_values.clone()
                    target[0][action] = (target[0][action] + 100) / 2  # Weighted update

                    loss = loss_fn(q_values, target)
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()
            else:
                print(f"⚠️ Warning: No valid path found for Agent {i} from {starts[i]} to {goals[i]}")

        print(f"✅ Completed Epoch {epoch+1}/{num_epochs}\n")

    # Visualization remains the same
    plt.imshow(colored_map, origin="upper")
    for i in range(num_agents):
        if astar_paths[i]:
            plt.plot(*zip(*astar_paths[i]), color=AGENT_COLORS[i % len(AGENT_COLORS)], linestyle="dotted", linewidth=2, label=f"Agent {i+1} A*")
            plt.scatter(*starts[i], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black', marker="o", label=f"Agent {i+1} Start")
            plt.scatter(*goals[i], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black', marker="X", label=f"Agent {i+1} Goal")

    plt.grid(visible=True, color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(np.arange(-0.5, colored_map.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, colored_map.shape[0], 1), [])
    plt.title(f"A* paths")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "Astar_paths.png"), dpi=300)  # ensure extension
    plt.close()



def train_dqn(total_episodes=1000, max_steps=1000, visualize=False):
    print("Start training...")
    start_time = time.time()
    cpu_usages = []  # Store CPU usage per episode
    process = psutil.Process(os.getpid())  # Get current process handle
    global epsilon
    scat = path_lines = paths = fig_vis = ax_vis = None
    if visualize:
        fig_vis, ax_vis, scat, path_lines, paths = init_visualization(starts)

    total_rewards_history = []  # Store total rewards per episode
    success_rates = []  # Store percentage of successful episodes
    collision_rates = []  # Stores normalized collision rate per episode
    average_steps_per_episode = []  # New: to track average steps taken

    # Initialize shortest path storage
    shortest_paths = {f"agent_{i+1}": None for i in range(num_agents)}
    shortest_lengths = {f"agent_{i+1}": float("inf") for i in range(num_agents)}

    for episode in range(1, total_episodes + 1):
        start_cpu = process.cpu_percent(interval=None)  # "Prime" the measurement
        print(f"Episode {episode}...")

        agent_positions = starts[:]
        agent_paths = {i: [agent_positions[i]] for i in range(num_agents)}
        terminated_agents = set()

        if visualize and scat is not None:
            for i in range(num_agents):
                paths[i] = [agent_positions[i]]
                path_lines[i].set_data([agent_positions[i][0]], [agent_positions[i][1]])
            update_visualization(scat, agent_positions, path_lines, paths)

        state = get_state(agent_positions)
        total_reward = 0
        step_count = 0
        done = False

        window_size = 50  # Rolling window for success rate

        episode_collision_count = 0  # Count total collision events in this episode
        agent_active_steps = [0 for _ in range(num_agents)]  # Track how many steps each agent is active

        filename = f"map_{episode}.png"
        output_path = os.path.join(output_folder, filename)

        while not done:
            step_count += 1

            actions_selected, next_positions, rewards = [], [], []

            # Predict next positions
            for i in range(num_agents):
                if i not in terminated_agents:
                    action = random.randint(0, 3) if random.random() < epsilon else torch.argmax(dqn_agents[i](state)).item()
                    next_pos, reward, hit_obstacle = step(agent_positions[i], action, goals[i])
                    agent_active_steps[i] += 1
                    if hit_obstacle:
                        episode_collision_count += 1
                else:
                    next_pos, reward = agent_positions[i], 0  # No movement for terminated agents
                    hit_obstacle = False

                actions_selected.append(action)
                next_positions.append(next_pos)
                rewards.append(reward)

            # Store previous positions
            previous_positions = agent_positions.copy()

            # Check for collisions before applying movement
            positions_check = {}  # Track intended positions
            collision_agents = set()  # Agents that need to return

            # First pass: Detect direct collisions
            for i in range(num_agents):
                if next_positions[i] in positions_check:
                    first_agent = positions_check[next_positions[i]]

                    # Both agents revert to their previous positions
                    collision_agents.add(first_agent)
                    collision_agents.add(i)

                    # Small penalty for both agents
                    rewards[first_agent] = -50
                    rewards[i] = -50

                    episode_collision_count += 1

                else:
                    positions_check[next_positions[i]] = i  # Store the agent's index

            # Second pass: Detect swap collisions
            for i in range(num_agents):
                for j in range(i + 1, num_agents):  # Compare each pair
                    if (next_positions[i] == previous_positions[j]) and (next_positions[j] == previous_positions[i]):
                        # Both agents swapped positions → collision
                        collision_agents.add(i)
                        collision_agents.add(j)

                        # Apply penalty
                        rewards[i] = -50
                        rewards[j] = -50

                        episode_collision_count += 1

            # Revert all colliding agents
            for i in collision_agents:
                next_positions[i] = previous_positions[i]  # Move back to the previous step
                print(f"Agent {i+1} is returning to previous position due to collision at step {step_count}.")

            next_state = get_state(next_positions)
            memory.add_to_buffer((state, tuple(actions_selected), tuple(rewards), next_state, done))
            state = next_state
            agent_positions = next_positions.copy()
            if visualize and scat is not None:
                update_visualization(scat, agent_positions, path_lines, paths)
            total_reward += sum(rewards)

            # Update paths
            for i in range(num_agents):
                if i not in terminated_agents:
                    agent_paths[i].append(agent_positions[i])

             # Check termination only if all agents reached their goals
            for i in range(num_agents):
                if agent_positions[i] == goals[i]:
                    terminated_agents.add(i)

            if len(terminated_agents) == num_agents:
                done = True  # End episode only if all agents reach their goals

            if step_count >= max_steps:
                print(f"Episode {episode} ended due to max_steps.")
                done = True

            # Train the agents if enough samples are in the replay buffer
            if len(memory.buffer) > batch_size:
                (states, actions, rewards, next_states, dones), importance, indices = memory.sample_minibatch(batch_size)

                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)

                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).view(len(rewards), num_agents).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).view(len(dones), 1).to(device)
                importance = torch.tensor(importance, dtype=torch.float32).view(len(importance), 1).to(device)  # Convert to tensor

                td_errors = []
                losses = []

                for i in range(num_agents):
                    q_values = dqn_agents[i](states).to(device)
                    next_q_values = target_dqn_agents[i](next_states).detach().to(device)

                    max_next_q_values = torch.max(next_q_values, dim=1)[0].view(batch_size, 1).to(device)
                    target_q_values = rewards[:, i] + gamma * max_next_q_values.squeeze() * (1 - dones.squeeze())

                    td_error = torch.abs(target_q_values.unsqueeze(1) - q_values.gather(1, actions[:, i].unsqueeze(1))).to(device)
                    td_errors.append(td_error)

                    loss = (importance * loss_fn(q_values.gather(1, actions[:, i].unsqueeze(1)), target_q_values.unsqueeze(1))).mean()
                    losses.append(loss)

                    if i not in terminated_agents:
                        optimizers[i].zero_grad()
                        losses[i].backward()
                        optimizers[i].step()

                memory.set_priorities(indices, (sum(td_errors) / num_agents).detach().cpu().numpy())  # Update priorities

                #memory.set_priorities(indices, sum(td_errors) / num_agents)  # Average TD error for all agents

        # Count successful agents (reached goal)
        successful_agents = len([i for i in range(num_agents) if i in terminated_agents])
        success_rate = successful_agents / num_agents  # Success percentage per episode
        success_rates.append(success_rate)

        total_active_steps = sum(agent_active_steps)
        avg_steps = np.mean(agent_active_steps)
        average_steps_per_episode.append(avg_steps)

        if total_active_steps > 0:
            collision_rate = episode_collision_count / total_active_steps
        else:
            collision_rate = 0
        collision_rates.append(collision_rate)

        # Track total rewards
        total_rewards_history.append(total_reward)

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Update target networks every 10 episodes
        if episode % 10 == 0:
            for i in range(num_agents):
                update_target_network(target_dqn_agents[i], dqn_agents[i])
            print(f"Episode {episode}: Total Reward = {total_reward}")

        # Print success rate every 50 episodes
        if episode % 50 == 0:
            avg_success_rate = np.mean(success_rates[-window_size:])  # Rolling average
            print(f"Episode {episode}: Avg Success Rate (last {window_size} eps) = {avg_success_rate:.2f}")
            print(f"Ep {episode} - Collision Rate: {collision_rate:.4f}")

        # Update shortest paths
        for i in range(num_agents):
            agent_key = f"agent_{i+1}"

            if i in terminated_agents:
                final_path = agent_paths[i]  # Already ends at goal

                conflict_found = False
                swap_collision_found = False

                for steps, pos in enumerate(final_path):
                    for other_agent, other_path in shortest_paths.items():
                        if other_agent != agent_key and other_path:

                            # --- Direct conflict check (accounting for post-path presence) ---
                            if steps < len(other_path):
                                pos_b = other_path[steps]
                            else:
                                pos_b = other_path[-1]  # Agent stays at its goal

                            if pos_b == pos:
                                conflict_found = True
                                break

                            # --- Swap conflict check ---
                            if steps > 0:
                                prev_a = final_path[steps - 1]
                                prev_b = other_path[steps - 1] if steps - 1 < len(other_path) else other_path[-1]

                                if pos == prev_b and prev_a == pos_b:
                                    swap_collision_found = True
                                    break

                    if conflict_found or swap_collision_found:
                        break

                if not conflict_found and not swap_collision_found and \
                (shortest_paths[agent_key] is None or len(final_path) < shortest_lengths[agent_key]):
                    shortest_lengths[agent_key] = len(final_path)
                    shortest_paths[agent_key] = final_path[:]

                print(f"Agent {i+1} reached at episode {episode}!")

        # Save an image of every episode's paths (use a separate figure so the live view stays open)
        snap_fig, snap_ax = plt.subplots()
        snap_ax.imshow(colored_map, origin="upper")

        for i in range(num_agents):
            snap_ax.plot(*zip(*agent_paths[i]), color=AGENT_COLORS[i % len(AGENT_COLORS)], linewidth=2, label=f"Agent {i+1} Path")
            snap_ax.scatter(*starts[i][::1], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black', label=f"Agent {i+1} Start")
            snap_ax.scatter(*goals[i][::1], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black', label=f"Agent {i+1} Goal")

        snap_ax.grid(visible=True, color="gray", linestyle="-", linewidth=0.5)
        snap_ax.set_xticks(np.arange(-0.5, colored_map.shape[1], 1))
        snap_ax.set_yticks(np.arange(-0.5, colored_map.shape[0], 1))
        snap_ax.set_xticklabels([])
        snap_ax.set_yticklabels([])
        snap_ax.set_title(f"Episode {episode} Paths")
        snap_fig.savefig(output_path, dpi=300)
        plt.close(snap_fig)

        end_cpu = process.cpu_percent(interval=None)
        cpu_usage = end_cpu  # Percent CPU usage over this episode
        cpu_usages.append(cpu_usage)
    
    print("Plotting Shortest Paths...")
    plt.imshow(colored_map, origin="upper")

    for i in range(num_agents):
        agent_key = f"agent_{i+1}"
        
        if shortest_paths[agent_key]:
            print(shortest_paths[agent_key])

            # Extract positions while detecting wait steps
            x, y = zip(*shortest_paths[agent_key])  # Normal path
            wait_x, wait_y = [], []  # Store waiting positions

            for steps in range(1, len(shortest_paths[agent_key])):
                if shortest_paths[agent_key][steps] == shortest_paths[agent_key][steps - 1]:  # Detect waiting
                    wait_x.append(shortest_paths[agent_key][steps][0])
                    wait_y.append(shortest_paths[agent_key][steps][1])

            # Plot agent's shortest path
            plt.plot(x, y, color=AGENT_COLORS[i % len(AGENT_COLORS)], linewidth=2, label=f"Agent {i+1} Shortest Path")

            # Plot A* path (dotted)
            plt.plot(*zip(*astar_paths[i]), linestyle="dotted", color=AGENT_COLORS[i % len(AGENT_COLORS)], linewidth=2, label=f"Agent {i+1} A* Path")

            # Mark start & goal
            plt.scatter(*starts[i][::1], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black', label=f"Agent {i+1}")
            plt.scatter(*goals[i][::1], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black')

            # Plot waiting steps in red 'X'
            plt.scatter(wait_x, wait_y, color='r', marker="x", s=50, label=f"Agent {i+1} Waiting")

    plt.grid(visible=True, color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(np.arange(-0.5, colored_map.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, colored_map.shape[0], 1), [])
    plt.title("Shortest Paths Found")
    #plt.legend()
    plt.savefig(os.path.join(output_folder, "Shortest_Path.png"), dpi=300)
    plt.close()

    # Plot Success Rate
    plt.figure(figsize=(10, 5))
    #plt.plot(range(1, total_episodes + 1), success_rates, label="Success Rate per Episode", alpha=0.5)
    plt.plot(np.convolve(success_rates, np.ones(window_size)/window_size, mode="valid"), label="Success Rate (Rolling Avg)")
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.title("Training Convergence - Success Rate")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "Success_Rate.png"), dpi=300)
    plt.close()

    # Plot Total Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), total_rewards_history, label="Total Reward per Episode", alpha=0.5)
    #plt.plot(range(1, total_episodes + 1), np.convolve(total_rewards_history, np.ones(window_size) / window_size, mode="same"), label="Smoothed Reward", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.title("Training Convergence - Total Rewards")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "Total_Rewards.png"), dpi=300)
    plt.close()

    # Plot Collision Rates
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), collision_rates, label="Collision Rates per Episode", alpha=0.5)
    #plt.plot(range(1, total_episodes + 1), np.convolve(collision_rates, np.ones(window_size) / window_size, mode="same"), label="Smoothed Collision Rates", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Collision Rates")
    plt.title("Training Convergence - Collision Rates")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "Collision_Rates.png"), dpi=300)
    plt.close()

    # Plot Avr Taking Steps
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), average_steps_per_episode, label="Average Taking Steps per Episode", alpha=0.5)
    #plt.plot(range(1, total_episodes + 1), np.convolve(average_steps_per_episode, np.ones(window_size) / window_size, mode="same"), label="Smoothed Steps", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Average Taking Steps")
    plt.title("Training Convergence - Average Taking Steps per Episode")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "Average_Taking_Steps.png"), dpi=300)
    plt.close()

    end_time = time.time()  # End tracking time
    total_time = end_time - start_time  # Compute total time

    print(f"\n🏆 Training Complete!")
    print(f"⏳ Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # Compute steps for each agent
    agent_data = []
    for agent, path in shortest_paths.items():
        num_steps = len(path) - 1  # Steps taken (excluding start position)
        agent_data.append({"Agent": agent, "Shortest Path": str(path), "Steps Taken": num_steps})

    # Convert to DataFrame
    df_agents = pd.DataFrame(agent_data)

    # Compute overall training metrics
    average_success_rate = np.mean(success_rates)
    average_collisions = np.mean(collision_rates)
    # Get computational cost
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # MB
    average_cpu_usage = np.mean(cpu_usages)

    # Create a DataFrame for general metrics
    df_metrics = pd.DataFrame({
        "Total Time (seconds)": [total_time],
        "Total Time (minutes)": [total_time / 60],
        "Average Success Rate": [average_success_rate],
        "Average Collision Rate": [average_collisions],
        "Memory Usage (MB)": [memory_usage],
        "CPU Usage (%)": [average_cpu_usage]
    })

    # Load existing Excel file (if exists)
    file_path = "algorithm_comparison.xlsx"
    if os.path.exists(file_path):
        with pd.ExcelFile(file_path) as existing_file:
            df_existing_metrics = pd.read_excel(existing_file, sheet_name="Metrics")
            df_existing_agents = pd.read_excel(existing_file, sheet_name="Agent Paths")
    else:
        df_existing_metrics = pd.DataFrame()
        df_existing_agents = pd.DataFrame()

    # Append new data
    df_final_metrics = pd.concat([df_existing_metrics, df_metrics], ignore_index=True)
    df_final_agents = pd.concat([df_existing_agents, df_agents], ignore_index=True)

    # Save both metrics and paths to separate sheets
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        df_final_metrics.to_excel(writer, sheet_name="Metrics", index=False)
        df_final_agents.to_excel(writer, sheet_name="Agent Paths", index=False)

    print("\n📊 Metrics and shortest paths saved to 'algorithm_comparison.xlsx'!")

    if visualize:
        plt.ioff()


if __name__ == "__main__":
    pretrain_with_astar(dqn_agents, astar_paths)
    train_dqn(visualize=True)
