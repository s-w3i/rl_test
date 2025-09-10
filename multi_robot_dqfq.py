#!/usr/bin/env python3
# Edition 2: Right-of-Way rule + Success Buffer + minor schedule tweaks
# Filename suggestion:
# rack_pickplace_dueling_double_per_fixed_pairs_shield_dqfd_v2.py

import os
import time
import heapq
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

# =========================
# Reproducibility & Torch
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# =========================
# Global / Settings
# =========================
GRIDSIZE = 20
AGENT_COLORS = ['blue', 'green', 'purple', 'gray', 'orange', 'brown', 'cyan', 'magenta', 'olive', 'navy']

num_agents = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Results folder for Edition 2
output_folder = "results/rack_pickplace_dueling_double_per_fixed_pairs_shield_dqfd_v2"
os.makedirs(output_folder, exist_ok=True)

# =========================
# Map & Utility
# =========================
def euclidean_distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# 1 = obstacle (rack), 0 = free aisle
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

ACTIONS = [(0,-1),(0,1),(-1,0),(1,0),(0,0)]  # U, D, L, R, STOP
ACTION_SIZE = 5
STOP_IDX = ACTIONS.index((0,0))
FOUR = ((0,-1),(0,1),(-1,0),(1,0))

# =========================
# A* planner (treats start/goal as free)
# =========================
def astar(grid, start, goal):
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came = {}
    g = {start:0}
    f = {start:h(start,goal)}
    H,W = grid.shape
    while open_set:
        _,cur = heapq.heappop(open_set)
        if cur == goal:
            path=[]
            while cur in came:
                path.append(cur); cur=came[cur]
            path.append(start)
            return path[::-1]
        for dx,dy in FOUR:
            nb = (cur[0]+dx, cur[1]+dy)
            if 0<=nb[0]<W and 0<=nb[1]<H and grid[nb[1],nb[0]]==0:
                tg = g[cur] + 1
                if nb not in g or tg < g[nb]:
                    came[nb]=cur; g[nb]=tg; f[nb]=tg+h(nb,goal)
                    heapq.heappush(open_set,(f[nb],nb))
    return []

# =========================
# Start/Goal generation from OBSTACLES (racks)
# =========================
def generate_map_with_positions(min_distance=8, max_attempts=10000):
    H,W = obstacle_map.shape
    dockables = []
    for y in range(H):
        for x in range(W):
            if obstacle_map[y, x] != 1:
                continue
            for dx,dy in FOUR:
                nx,ny = x+dx, y+dy
                if 0<=nx<W and 0<=ny<H and obstacle_map[ny,nx] == 0:
                    dockables.append((x,y))
                    break

    if len(dockables) < 2*num_agents:
        raise RuntimeError(f"Not enough dockable rack cells: {len(dockables)} available; need {2*num_agents}.")

    starts, goals = [], []
    used = set(); attempts = 0
    while len(starts) < num_agents and attempts < max_attempts:
        attempts += 1
        s, g = random.sample(dockables, 2)
        if s in used or g in used: continue
        if euclidean_distance(s, g) < min_distance: continue
        grid_mod = obstacle_map.copy()
        grid_mod[s[1], s[0]] = 0; grid_mod[g[1], g[0]] = 0
        if not astar(grid_mod, s, g): continue
        starts.append(s); goals.append(g); used.add(s); used.add(g)

    if len(starts) < num_agents:
        raise RuntimeError(f"Could not pair {num_agents} agents after {attempts} attempts.")

    colored = np.zeros((H,W,3),dtype=np.float32)
    colored[obstacle_map==0]=[1,1,1]; colored[obstacle_map==1]=[0,0,0]
    for i in range(num_agents):
        sx,sy=starts[i]; gx,gy=goals[i]
        colored[sy,sx]=[0,0,1]; colored[gy,gx]=[1,0,0]
    return colored, starts, goals

# =========================
# Observation (GOAL CHANNEL)
# =========================
INPUT_CHANNELS = 4  # obstacles, me, others, my-goal

def get_state_for_agent(agent_idx:int, positions:List[Tuple[int,int]], goals:List[Tuple[int,int]]):
    s = np.zeros((INPUT_CHANNELS, GRIDSIZE, GRIDSIZE), dtype=np.float32)
    s[0] = obstacle_map
    ax, ay = positions[agent_idx]
    s[1, ay, ax] = 1.0
    for j,(x,y) in enumerate(positions):
        if j==agent_idx: continue
        s[2, y, x] = 1.0
    gx, gy = goals[agent_idx]
    s[3, gy, gx] = 1.0
    return torch.from_numpy(s).unsqueeze(0).to(device)

# =========================
# Dueling DQN with Distance Head
# =========================
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*GRIDSIZE*GRIDSIZE, 256)
        self.val = nn.Linear(256, 1)
        self.adv = nn.Linear(256, action_size)
        self.dist_head = nn.Linear(256, 1)  # predicts normalized BFS distance

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        trunk = self.relu(self.fc(self.flatten(x)))
        V = self.val(trunk)
        A = self.adv(trunk)
        q = V + (A - A.mean(dim=1, keepdim=True))
        dist = self.dist_head(trunk)
        return q, dist

policy = DuelingDQN(INPUT_CHANNELS, ACTION_SIZE).to(device)
target = DuelingDQN(INPUT_CHANNELS, ACTION_SIZE).to(device)
target.load_state_dict(policy.state_dict())
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# =========================
# PER (store dist target as well)
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity:int):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, transition, priority:float):
        # transition = (s, a, r, ns, d, dist_target)
        self.buffer.append(transition)
        self.priorities.append(priority)

    def __len__(self): return len(self.buffer)

    def probabilities(self, alpha:float):
        p = np.array([float(x) for x in self.priorities], dtype=np.float32)
        p = np.power(p, alpha); s = p.sum()
        if s <= 0: p[:] = 1.0/len(p)
        else: p /= s
        return p

    def sample(self, batch_size:int, alpha:float, beta:float):
        n = min(len(self.buffer), batch_size)
        prob = self.probabilities(alpha)
        idxs = random.choices(range(len(self.buffer)), k=n, weights=prob)
        samples = [self.buffer[i] for i in idxs]
        imp = (len(self.buffer) * prob[idxs]) ** (-beta)
        imp = imp / max(imp.max(), 1e-8)
        s  = torch.cat([t[0] for t in samples]).to(device)
        a  = torch.tensor([t[1] for t in samples], dtype=torch.long, device=device).unsqueeze(1)
        r  = torch.tensor([t[2] for t in samples], dtype=torch.float32, device=device).unsqueeze(1)
        ns = torch.cat([t[3] for t in samples]).to(device)
        d  = torch.tensor([t[4] for t in samples], dtype=torch.float32, device=device).unsqueeze(1)
        dist_t = torch.tensor([t[5] for t in samples], dtype=torch.float32, device=device).unsqueeze(1)
        iw = torch.tensor(imp, dtype=torch.float32, device=device).unsqueeze(1)
        return s, a, r, ns, d, dist_t, iw, idxs

    def update_priorities(self, idxs, new_p, eps=1e-3):
        for i,p in zip(idxs, new_p):
            self.priorities[i] = float(abs(p)) + eps

memory = PrioritizedReplayBuffer(70000)
alpha = 0.5
beta_start = 0.4

# =========================
# Success buffer (uniform)
# =========================
class SuccessBuffer:
    def __init__(self, capacity:int=50000):
        self.buf = deque(maxlen=capacity)
    def add_many(self, transitions):
        self.buf.extend(transitions)
    def __len__(self): return len(self.buf)
    def sample(self, n:int):
        if not self.buf: return None
        n = min(n, len(self.buf))
        batch = random.sample(self.buf, n)
        s  = torch.cat([t[0] for t in batch]).to(device)
        a  = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device).unsqueeze(1)
        r  = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
        ns = torch.cat([t[3] for t in batch]).to(device)
        d  = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
        dist_t = torch.tensor([t[5] for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
        return s, a, r, ns, d, dist_t

success_buffer = SuccessBuffer()

# =========================
# Demo buffer + DQfD margin loss
# =========================
class DemoBuffer:
    def __init__(self):
        self.s = []
        self.a = []
    def add_many(self, demos):
        for s,a in demos:
            self.s.append(s)
            self.a.append(a)
    def __len__(self): return len(self.s)
    def sample(self, n):
        if len(self.s) == 0:
            return None, None
        idx = np.random.choice(len(self.s), size=min(n, len(self.s)), replace=False)
        S = torch.cat([self.s[i] for i in idx], dim=0).to(device)
        A = torch.tensor([self.a[i] for i in idx], dtype=torch.long, device=device)
        return S, A

demo_buffer = DemoBuffer()

def dqfd_margin_loss(q, expert_a, margin=0.8):
    one_hot = F.one_hot(expert_a, num_classes=ACTION_SIZE).float()
    q_exp = (q * one_hot).sum(dim=1, keepdim=True)
    q_others = q + margin * (1.0 - one_hot)
    max_others, _ = q_others.max(dim=1, keepdim=True)
    loss = F.relu(max_others - q_exp)
    return loss.mean()

# =========================
# Action Masking (allow entering OWN goal rack)
# =========================
def mask_invalid_actions(q:torch.Tensor, pos:Tuple[int,int], goal:Tuple[int,int]):
    x,y = pos
    q = q.clone()
    for a,(dx,dy) in enumerate(ACTIONS):
        if (dx,dy) == (0,0):  # STOP allowed
            continue
        nx, ny = x+dx, y+dy
        if nx < 0 or nx >= GRIDSIZE or ny < 0 or ny >= GRIDSIZE:
            q[0, a] = -1e9; continue
        if obstacle_map[ny, nx] == 1 and (nx, ny) != goal:
            q[0, a] = -1e9
    return q

# =========================
# Reward Shaping via BFS Distance
# =========================
from collections import deque as _dq
_dist_cache = {}
def distance_map_to(goal:Tuple[int,int]):
    if goal in _dist_cache:
        return _dist_cache[goal]
    gx, gy = goal
    grid = obstacle_map.copy()
    grid[gy, gx] = 0
    H,W = grid.shape
    D = np.full((H,W), np.inf, dtype=np.float32)
    q = _dq([(gx,gy)])
    D[gy,gx] = 0.0
    while q:
        x,y = q.popleft()
        for dx,dy in FOUR:
            nx,ny = x+dx, y+dy
            if 0<=nx<W and 0<=ny<H and grid[ny,nx]==0 and D[ny,nx] > D[y,x] + 1:
                D[ny,nx] = D[y,x] + 1
                q.append((nx,ny))
    _dist_cache[goal] = D
    return D

def normalized_distance(pos:Tuple[int,int], goal:Tuple[int,int]):
    D = distance_map_to(goal)
    d = D[pos[1], pos[0]]
    if not np.isfinite(d):
        d = GRIDSIZE * 2.0
    return float(np.clip(d / (GRIDSIZE*2.0), 0.0, 1.0))

# =========================
# Env Step (with rack semantics + BFS shaping)
# =========================
def step_single(agent_pos, action, goal):
    dx,dy = ACTIONS[action]
    if dx==0 and dy==0:
        return agent_pos, -0.2, False

    nx, ny = agent_pos[0]+dx, agent_pos[1]+dy
    if not (0 <= nx < GRIDSIZE and 0 <= ny < GRIDSIZE):
        return agent_pos, -5.0, True
    if obstacle_map[ny, nx] == 1 and (nx, ny) != goal:
        return agent_pos, -5.0, True

    D = distance_map_to(goal)
    prevd = D[agent_pos[1], agent_pos[0]]
    new_pos = (nx, ny)
    newd = D[new_pos[1], new_pos[0]]

    if new_pos == goal:
        return new_pos, +50.0, False

    shaped = 0.0
    if np.isfinite(prevd) and np.isfinite(newd):
        shaped = float(np.clip(prevd - newd, -1.0, 1.0) * 0.7)
    elif not np.isfinite(prevd) and np.isfinite(newd):
        shaped = +0.2
    elif np.isfinite(prevd) and not np.isfinite(newd):
        shaped = -0.2

    step_penalty = -0.15
    return new_pos, shaped + step_penalty, False

# =========================
# Visualization helpers
# =========================
def init_visualization(colored_map, positions, episode_num:int):
    plt.close('all'); plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(colored_map, origin="upper")
    ax.set_xticks(np.arange(-0.5, GRIDSIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRIDSIZE, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.set_title(f"Episode {episode_num} (running)")
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(num_agents)]
    xs = [p[0] for p in positions]; ys = [p[1] for p in positions]
    scat = ax.scatter(xs, ys, c=colors, s=100)
    path_lines = []; paths = [[p] for p in positions]
    for i in range(num_agents):
        (line,) = ax.plot([positions[i][0]], [positions[i][1]], color=colors[i], linewidth=1)
        path_lines.append(line)
    plt.show(block=False)
    return fig, ax, scat, path_lines, paths

def update_visualization(fig, ax, scat, positions, path_lines, paths, episode_num:int):
    scat.set_offsets(np.array(positions))
    for i, pos in enumerate(positions):
        paths[i].append(pos)
        xs, ys = zip(*paths[i]); path_lines[i].set_data(xs, ys)
    ax.set_title(f"Episode {episode_num} (running)")
    fig.canvas.draw_idle(); plt.pause(0.01)

# =========================
# A* Demos (collection + pretrain)
# =========================
def collect_astar_demos(num_pairs=1500):
    demos = []; tries = 0
    while len(demos) < num_pairs and tries < num_pairs*25:
        tries += 1
        _, starts, goals = generate_map_with_positions(min_distance=10)
        for i in range(num_agents):
            grid_mod = obstacle_map.copy()
            grid_mod[starts[i][1], starts[i][0]] = 0
            grid_mod[goals[i][1], goals[i][0]] = 0
            path = astar(grid_mod, starts[i], goals[i])
            if not path or len(path) < 2: continue
            for t in range(len(path)-1):
                positions = [path[t] for _ in range(num_agents)]
                s = get_state_for_agent(i, positions, goals)
                move = (path[t+1][0]-path[t][0], path[t+1][1]-path[t][1])
                a = ACTIONS.index(move) if move in ACTIONS else STOP_IDX
                demos.append((s, a))
    random.shuffle(demos)
    return demos

def pretrain_with_demos(demos, epochs=12):
    print(f"🔁 Pretraining on {len(demos)} A* samples ...")
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(policy.parameters(), lr=1e-4)
    bs = 256
    for ep in range(epochs):
        random.shuffle(demos); total = 0.0; steps = 0
        for k in range(0, len(demos), bs):
            batch = demos[k:k+bs]
            s = torch.cat([b[0] for b in batch], dim=0).to(device)
            a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
            q, _ = policy(s)
            loss = ce(q, a)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            total += float(loss.item()); steps += 1
        print(f"  epoch {ep+1}/{epochs}  loss={total/max(1,steps):.4f}")
    target.load_state_dict(policy.state_dict())
    print("✅ Pretraining done.\n")

# =========================
# Helpers: A* suggestion + schedules + RoW priority
# =========================
def astar_suggestion(pos, path):
    if not path: return STOP_IDX
    try:
        k = path.index(pos)
        if k < len(path)-1:
            dx = path[k+1][0]-pos[0]; dy = path[k+1][1]-pos[1]
            return ACTIONS.index((dx,dy)) if (dx,dy) in ACTIONS else STOP_IDX
    except ValueError:
        pass
    return STOP_IDX

def linear_decay(start, end, frac):
    frac = float(np.clip(frac, 0.0, 1.0))
    return start + (end - start) * frac

def priority_key(agent_idx:int, current_pos:Tuple[int,int], intended_pos:Tuple[int,int], goal:Tuple[int,int]):
    """Higher is better (winner). Prefer smaller BFS distance after the move; tie-break by index."""
    D = distance_map_to(goal)
    d = D[intended_pos[1], intended_pos[0]]
    if not np.isfinite(d):
        d = 1e9
    # negative distance, then negative index to prefer lower index on tie
    return (-d, -agent_idx)

# =========================
# Training (Edition 2)
# =========================
def train(starts, goals, colored_map, total_episodes=1200, max_steps=1500, visualize=True):
    start_time = time.time()
    cpu_usages=[]
    process = psutil.Process(os.getpid())

    success_rates=[]; total_rewards_history=[]; collision_rates=[]; average_steps_per_episode=[]
    window_size=50

    epsilon_start, epsilon_end = 1.0, 0.05           # lower exploration floor now
    epsilon_decay_episodes = int(total_episodes*0.6)

    # Imitation schedules (persistent)
    demo_ratio_start, demo_ratio_end = 0.20, 0.05
    bc_lambda_start, bc_lambda_end = 0.30, 0.05
    bc_decay_begin = 0.30
    bc_decay_end   = 0.60

    shortest_paths = {f"agent_{i+1}": None for i in range(num_agents)}
    shortest_lengths = {f"agent_{i+1}": float("inf") for i in range(num_agents)}

    # Precompute reference A* once (racks opened at start/goal)
    astar_paths = []
    for i in range(num_agents):
        grid_mod = obstacle_map.copy()
        grid_mod[starts[i][1], starts[i][0]] = 0
        grid_mod[goals[i][1], goals[i][0]] = 0
        astar_paths.append(astar(grid_mod, starts[i], goals[i]))

    base_lr, min_lr = 3e-4, 1e-4  # cosine schedule

    courtesy_reward = 0.05
    violator_penalty = 3.0  # charged to rule violators (not yielders)

    for episode in range(1, total_episodes+1):
        print(f"\n=== Episode {episode}/{total_episodes} (Edition 2) ===")
        start_cpu = process.cpu_percent(interval=None)

        # reset BFS cache per episode
        global _dist_cache
        _dist_cache = {}

        # schedules
        frac_total = episode / total_episodes
        # cosine LR
        cos_factor = 0.5 * (1 + np.cos(np.pi * min(1.0, frac_total)))
        lr = min_lr + (base_lr - min_lr) * cos_factor
        for g in optimizer.param_groups:
            g['lr'] = lr

        # imitation schedules
        demo_ratio = max(demo_ratio_end, linear_decay(demo_ratio_start, demo_ratio_end, frac_total/0.50))
        if frac_total <= bc_decay_begin:
            bc_lambda = bc_lambda_start
        elif frac_total >= bc_decay_end:
            bc_lambda = bc_lambda_end
        else:
            t = (frac_total - bc_decay_begin) / (bc_decay_end - bc_decay_begin)
            bc_lambda = linear_decay(bc_lambda_start, bc_lambda_end, t)

        # PER beta ramp to 1.0 by 40% of training
        beta_progress = min(1.0, frac_total / 0.40)
        beta = beta_start + (1.0 - beta_start) * beta_progress

        # per-episode viz
        fig_vis=ax_vis=scat=path_lines=paths=None
        if visualize:
            fig_vis, ax_vis, scat, path_lines, paths = init_visualization(colored_map, starts, episode)

        positions = starts[:]
        terminated = set()
        agent_paths = {i:[positions[i]] for i in range(num_agents)}
        total_reward=0.0
        step_count=0
        done=False

        episode_collision_count=0
        agent_active_steps=[0]*num_agents

        epsilon = max(epsilon_end, epsilon_start - (epsilon_start-epsilon_end)*((episode-1)/max(1,epsilon_decay_episodes)))

        # Episode transitions for success buffer consideration
        episode_transitions = []

        while not done:
            step_count += 1

            chosen_actions = [STOP_IDX]*num_agents
            intended = [None]*num_agents
            rewards_step = [0.0]*num_agents

            for i in range(num_agents):
                if i in terminated:
                    intended[i]=positions[i]
                    continue

                s_i = get_state_for_agent(i, positions, goals)
                q_raw, _ = policy(s_i)
                q = mask_invalid_actions(q_raw, positions[i], goals[i])

                # A* teacher forcing (decaying)
                a_astar = astar_suggestion(positions[i], astar_paths[i])
                if random.random() < demo_ratio:
                    a = a_astar
                else:
                    a = random.randint(0, ACTION_SIZE-1) if random.random() < epsilon else q.argmax(dim=1).item()

                nxt_pos, rew, _ = step_single(positions[i], a, goals[i])
                chosen_actions[i]=a; intended[i]=nxt_pos; rewards_step[i]=rew
                agent_active_steps[i]+=1

            prev_positions = positions[:]

            # ---------------- Right-of-Way conflict resolution ----------------
            forced_stop = [False]*num_agents  # whether we forced STOP this step (for metrics)

            # Group agents by intended cell
            cell_groups = {}
            for i in range(num_agents):
                if i in terminated: continue
                cell_groups.setdefault(intended[i], []).append(i)

            # (a) Same-cell contention
            for cell, group in cell_groups.items():
                if len(group) <= 1:
                    continue
                # choose winner by priority (based on post-move distance to goal)
                winner = max(group, key=lambda idx: priority_key(idx, prev_positions[idx], intended[idx], goals[idx]))
                for idx in group:
                    if idx == winner:
                        continue
                    # If the agent already chose STOP, it's a correct yield → courtesy
                    if chosen_actions[idx] == STOP_IDX:
                        rewards_step[idx] += courtesy_reward
                    else:
                        # Violated priority: penalize and force yield
                        rewards_step[idx] -= violator_penalty
                    intended[idx] = prev_positions[idx]
                    forced_stop[idx] = True
                    episode_collision_count += 1

            # (b) Head-to-head swap resolution (for pairs not already forced)
            for i in range(num_agents):
                if i in terminated or forced_stop[i]: continue
                for j in range(i+1, num_agents):
                    if j in terminated or forced_stop[j]: continue
                    if intended[i]==prev_positions[j] and intended[j]==prev_positions[i]:
                        # pick winner by priority
                        wi = priority_key(i, prev_positions[i], intended[i], goals[i])
                        wj = priority_key(j, prev_positions[j], intended[j], goals[j])
                        winner = i if wi > wj else j
                        loser  = j if winner == i else i
                        if chosen_actions[loser] == STOP_IDX:
                            rewards_step[loser] += courtesy_reward
                        else:
                            rewards_step[loser] -= violator_penalty
                        intended[loser] = prev_positions[loser]
                        forced_stop[loser] = True
                        episode_collision_count += 1
            # ------------------------------------------------------------------

            next_positions = intended[:]
            for i in range(num_agents):
                if i in terminated: continue
                s_i  = get_state_for_agent(i, positions, goals)
                ns_i = get_state_for_agent(i, next_positions, goals)
                d_i  = float(next_positions[i]==goals[i])
                dist_t_i = normalized_distance(positions[i], goals[i])

                # Demo priority boost if action matches A* suggestion
                exp_priority = 1.0
                a_astar = astar_suggestion(positions[i], astar_paths[i])
                if chosen_actions[i] == a_astar:
                    exp_priority = 2.0

                tr = (s_i, chosen_actions[i], rewards_step[i], ns_i, d_i, dist_t_i)
                memory.add(tr, priority=exp_priority)
                episode_transitions.append(tr)

            positions = next_positions[:]
            if visualize and scat is not None:
                update_visualization(fig_vis, ax_vis, scat, positions, path_lines, paths, episode)

            total_reward += sum(rewards_step)

            for i in range(num_agents):
                if i not in terminated:
                    agent_paths[i].append(positions[i])
                if positions[i]==goals[i]:
                    terminated.add(i)

            if len(terminated)==num_agents or step_count>=max_steps:
                done=True

            # ------------------- Learning step -------------------
            if len(memory) >= 10000:
                states, actions, rewards_b, next_states, dones, dist_targets_b, iw, idxs = memory.sample(
                    batch_size=256, alpha=alpha, beta=beta)

                with torch.no_grad():
                    online_next, _ = policy(next_states)
                    next_actions = online_next.argmax(dim=1, keepdim=True)
                    target_next, _ = target(next_states)
                    target_next = target_next.gather(1, next_actions)
                    targets = rewards_b + 0.99 * target_next * (1.0 - dones)

                q_pred, dist_pred = policy(states)
                q_taken = q_pred.gather(1, actions)

                # TD loss (PER-weighted)
                loss_element = F.smooth_l1_loss(q_taken, targets, reduction='none')
                td_loss_main = (iw * loss_element).mean()

                # Success-buffer TD loss (uniform) ~20% weight
                lambda_success = 0.2
                td_loss_succ = torch.tensor(0.0, device=device)
                succ_batch = success_buffer.sample(64)
                if succ_batch is not None:
                    S_s, A_s, R_s, NS_s, D_s, Dist_s = succ_batch
                    with torch.no_grad():
                        online_next_s, _ = policy(NS_s)
                        next_a_s = online_next_s.argmax(dim=1, keepdim=True)
                        target_next_s, _ = target(NS_s)
                        target_next_s = target_next_s.gather(1, next_a_s)
                        targets_s = R_s + 0.99 * target_next_s * (1.0 - D_s)
                    q_pred_s, dist_pred_s = policy(S_s)
                    q_taken_s = q_pred_s.gather(1, A_s)
                    td_loss_succ = F.smooth_l1_loss(q_taken_s, targets_s)
                    # mix distance supervision too
                    dist_loss_s = F.smooth_l1_loss(dist_pred_s, Dist_s)
                    td_loss_succ = td_loss_succ + 0.1 * dist_loss_s

                # DQfD margin imitation
                bc_loss = torch.tensor(0.0, device=device)
                S_demo, A_demo = demo_buffer.sample(n=128)
                if S_demo is not None:
                    q_demo, _ = policy(S_demo)
                    bc_loss = dqfd_margin_loss(q_demo, A_demo, margin=0.8)

                # Distance head supervision (main batch)
                dist_loss_main = F.smooth_l1_loss(dist_pred, dist_targets_b)

                loss = td_loss_main + lambda_success * td_loss_succ + bc_lambda * bc_loss + 0.1 * dist_loss_main

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()

                # Update PER priorities by TD error magnitude
                td_err = (targets - q_taken).detach().cpu().numpy().flatten()
                memory.update_priorities(idxs, td_err)

                # Soft update target (slower)
                with torch.no_grad():
                    tau = 0.003
                    for tp, p in zip(target.parameters(), policy.parameters()):
                        tp.data.mul_(1.0 - tau).add_(tau * p.data)
            # -----------------------------------------------------

        # Timeout penalty transitions for agents that didn't reach goal
        for i in range(num_agents):
            if i not in terminated:
                s_i = get_state_for_agent(i, positions, goals)
                ns_i = s_i
                dist_t_i = normalized_distance(positions[i], goals[i])
                tr = (s_i, STOP_IDX, -20.0, ns_i, 1.0, dist_t_i)
                memory.add(tr, priority=1.0)
                total_reward -= 20.0
                episode_transitions.append(tr)

        successful_agents = sorted([i+1 for i in range(num_agents) if i in terminated])
        print(f"Robots reached goal this episode: {successful_agents if successful_agents else 'None'}")

        success_rate = len(terminated) / num_agents
        success_rates.append(success_rate)

        # Add the whole episode to success buffer if good
        if success_rate >= 0.75:
            success_buffer.add_many(episode_transitions)

        total_active_steps = sum(agent_active_steps)
        collision_rate = (episode_collision_count / total_active_steps) if total_active_steps>0 else 0.0
        collision_rates.append(collision_rate)
        total_rewards_history.append(total_reward)
        average_steps_per_episode.append(np.mean(agent_active_steps) if agent_active_steps else 0)

        # Track shortest found paths (over episodes)
        for i in range(num_agents):
            agent_key=f"agent_{i+1}"
            if i in terminated:
                path=agent_paths[i]
                if path and (shortest_paths[agent_key] is None or len(path)<shortest_lengths[agent_key]):
                    shortest_lengths[agent_key]=len(path)
                    shortest_paths[agent_key]=path[:]

        # snapshot figure
        snap_fig, snap_ax = plt.subplots()
        snap_ax.imshow(colored_map, origin="upper")
        for i in range(num_agents):
            snap_ax.plot(*zip(*agent_paths[i]), color=AGENT_COLORS[i % len(AGENT_COLORS)], linewidth=2)
            snap_ax.scatter(*starts[i], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black')
            snap_ax.scatter(*goals[i], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black', marker='X')
        snap_ax.grid(visible=True, color="gray", linestyle="-", linewidth=0.5)
        snap_ax.set_xticks(np.arange(-0.5, GRIDSIZE, 1)); snap_ax.set_yticks(np.arange(-0.5, GRIDSIZE, 1))
        snap_ax.set_xticklabels([]); snap_ax.set_yticklabels([])
        snap_ax.set_title(f"Episode {episode} Paths (Edition 2)")
        snap_fig.savefig(os.path.join(output_folder, f"map_{episode}.png"), dpi=300)
        plt.close(snap_fig)

        end_cpu = process.cpu_percent(interval=None)
        cpu_usages.append(end_cpu)
        print(f"Episode {episode} done. Success rate: {success_rate:.2f} | Attempts-to-collide/step: {collision_rate:.3f} | LR={lr:.2e} | bcλ={bc_lambda:.2f} | demo_ratio={demo_ratio:.2f}")

    # ======= Final Plots =======
    print("Plotting Shortest Paths...")
    plt.imshow(np.stack([1-obstacle_map]*3, axis=-1), origin="upper")
    for i in range(num_agents):
        key=f"agent_{i+1}"
        if shortest_paths[key]:
            x,y = zip(*shortest_paths[key])
            plt.plot(x,y, color=AGENT_COLORS[i % len(AGENT_COLORS)], linewidth=2, label=f"A{i+1}")
            plt.scatter(*shortest_paths[key][0], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black')
            plt.scatter(*shortest_paths[key][-1], color=AGENT_COLORS[i % len(AGENT_COLORS)], s=100, edgecolors='black', marker='X')
    plt.grid(visible=True, color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(np.arange(-0.5, GRIDSIZE, 1), []); plt.yticks(np.arange(-0.5, GRIDSIZE, 1), [])
    plt.title("Shortest Paths Found (Edition 2)")
    plt.savefig(os.path.join(output_folder, "Shortest_Path.png"), dpi=300)
    plt.close()

    # Success Rate
    plt.figure(figsize=(10,5))
    if len(success_rates)>=window_size:
        roll = np.convolve(success_rates, np.ones(window_size)/window_size, mode="valid")
        plt.plot(roll, label="Success Rate (Rolling Avg)")
    else:
        plt.plot(success_rates, label="Success Rate")
    plt.xlabel("Episodes"); plt.ylabel("Success Rate"); plt.title("Training Convergence - Success Rate (Edition 2)")
    plt.grid(True); plt.savefig(os.path.join(output_folder, "Success_Rate.png"), dpi=300); plt.close()

    # Rewards
    plt.figure(figsize=(10,5))
    plt.plot(total_rewards_history, alpha=0.7)
    plt.xlabel("Episodes"); plt.ylabel("Total Rewards"); plt.title("Training Convergence - Total Rewards (Edition 2)")
    plt.grid(True); plt.savefig(os.path.join(output_folder, "Total_Rewards.png"), dpi=300); plt.close()

    # Collisions (attempted)
    plt.figure(figsize=(10,5))
    plt.plot(collision_rates, alpha=0.7)
    plt.xlabel("Episodes"); plt.ylabel("Attempted Collisions / Step"); plt.title("Training Convergence - Attempted Collisions (Edition 2)")
    plt.grid(True); plt.savefig(os.path.join(output_folder, "Collision_Rates.png"), dpi=300); plt.close()

    # Steps
    plt.figure(figsize=(10,5))
    plt.plot(average_steps_per_episode, alpha=0.7)
    plt.xlabel("Episodes"); plt.ylabel("Average Taking Steps"); plt.title("Training Convergence - Avg Steps (Edition 2)")
    plt.grid(True); plt.savefig(os.path.join(output_folder, "Average_Taking_Steps.png"), dpi=300); plt.close()

    # ======= Summary Table =======
    end_time = time.time()
    total_time = end_time - start_time
    avg_success = float(np.mean(success_rates)) if success_rates else 0.0
    avg_coll = float(np.mean(collision_rates)) if collision_rates else 0.0
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)
    average_cpu_usage = float(np.mean(cpu_usages)) if cpu_usages else 0.0

    agent_data = []
    for agent, path in shortest_paths.items():
        steps = len(path)-1 if path else -1
        agent_data.append({"Agent": agent, "Shortest Path": str(path), "Steps Taken": steps})
    df_agents = pd.DataFrame(agent_data)

    df_metrics = pd.DataFrame({
        "Total Time (seconds)": [total_time],
        "Total Time (minutes)": [total_time/60],
        "Average Success Rate": [avg_success],
        "Attempted Collisions / Step": [avg_coll],
        "Memory Usage (MB)": [memory_usage],
        "CPU Usage (%)": [average_cpu_usage]
    })

    file_path = "algorithm_comparison.xlsx"
    if os.path.exists(file_path):
        with pd.ExcelFile(file_path) as xf:
            df_existing_metrics = pd.read_excel(xf, sheet_name="Metrics")
            df_existing_agents = pd.read_excel(xf, sheet_name="Agent Paths")
    else:
        df_existing_metrics = pd.DataFrame()
        df_existing_agents = pd.DataFrame()

    df_final_metrics = pd.concat([df_existing_metrics, df_metrics], ignore_index=True)
    df_final_agents = pd.concat([df_existing_agents, df_agents], ignore_index=True)
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        df_final_metrics.to_excel(writer, sheet_name="Metrics", index=False)
        df_final_agents.to_excel(writer, sheet_name="Agent Paths", index=False)

    print("\n🏁 Training Complete (Edition 2)!")
    print(f"⏳ Total Training Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"✅ Avg Success Rate: {avg_success:.3f}   🚧 Attempted Collisions/step: {avg_coll:.3f}")
    print("📊 Saved metrics & paths to algorithm_comparison.xlsx")

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Build demos, pretrain, then train (Edition 2)
    colored_map, starts_init, goals_init = generate_map_with_positions(min_distance=10)
    demos = collect_astar_demos(num_pairs=1500)
    demo_buffer.add_many(demos)
    pretrain_with_demos(demos, epochs=12)
    train(starts_init, goals_init, colored_map, total_episodes=1200, max_steps=1500, visualize=True)
