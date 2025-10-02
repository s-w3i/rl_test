#!/usr/bin/env python3
# multi_agent_dqfq_v2C.py
# - Cooperative space-time A* teacher (reservation table) for demos/stall-bursts
# - Intent-aware collision masking; stronger collision shaping (yield bonus)
# - Stronger BC anchor; downsampled conv net; per-episode path images
# - Artifacts saved under artifacts/<RUN_NAME>/{images,images/episodes,metrics}

import os, heapq, random
from collections import deque
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None

IMAGEIO_AVAILABLE = imageio is not None
_IMAGEIO_WARNED = False

# =========================
# Repro & Torch
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# =========================
# Globals
# =========================
GRIDSIZE = 20
num_agents = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

AGENT_COLORS = ['blue','green','purple','gray','orange','brown','cyan','magenta','olive','navy']

# =========================
# Artifact paths
# =========================
RUN_NAME = "rack_pickplace_cbs_10agents_v1"
BASE_DIR = os.path.join("artifacts", RUN_NAME)
IMG_DIR = os.path.join(BASE_DIR, "images")
EPISODE_IMG_DIR = os.path.join(IMG_DIR, "episodes")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
for d in (BASE_DIR, IMG_DIR, EPISODE_IMG_DIR, METRICS_DIR):
    os.makedirs(d, exist_ok=True)

# =========================
# Map (1 = obstacle, 0 = free)
# =========================
obstacle_map = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
], dtype=np.float32)

ACTIONS = [(0,-1),(0,1),(-1,0),(1,0),(0,0)]  # U,D,L,R,STOP
U_IDX,D_IDX,L_IDX,R_IDX,STOP_IDX = 0,1,2,3,4
FOUR = ((0,-1),(0,1),(-1,0),(1,0))

# =========================
# Utilities
# =========================
def euclid(a,b): return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(grid, start, goal):
    """ Manhattan heuristic; 4-neigh. """
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    H,W = grid.shape
    openq = []
    heapq.heappush(openq,(0,start))
    came={} ; g={start:0}
    while openq:
        _,cur = heapq.heappop(openq)
        if cur==goal:
            path=[]; p=cur
            while p in came:
                path.append(p); p=came[p]
            path.append(start); return path[::-1]
        for dx,dy in FOUR:
            nx,ny = cur[0]+dx, cur[1]+dy
            if 0<=nx<W and 0<=ny<H and grid[ny,nx]==0:
                tg = g[cur]+1
                if (nx,ny) not in g or tg<g[(nx,ny)]:
                    came[(nx,ny)]=cur; g[(nx,ny)]=tg
                    ff = tg + h((nx,ny),goal)
                    heapq.heappush(openq,(ff,(nx,ny)))
    return []

# BFS distance maps cached per goal (treat goal as free)
_distance_cache: Dict[Tuple[int,int], np.ndarray] = {}

def bfs_distance_map(goal: Tuple[int,int]) -> np.ndarray:
    if goal in _distance_cache: return _distance_cache[goal]
    H,W = obstacle_map.shape
    grid = obstacle_map.copy()
    gx,gy = goal; grid[gy,gx] = 0
    dist = np.full((H,W), np.inf, dtype=np.float32)
    q = deque([(gx,gy)])
    dist[gy,gx] = 0.0
    while q:
        x,y = q.popleft()
        for dx,dy in FOUR:
            nx,ny = x+dx, y+dy
            if 0<=nx<W and 0<=ny<H and grid[ny,nx]==0:
                if dist[ny,nx] > dist[y,x] + 1:
                    dist[ny,nx] = dist[y,x] + 1
                    q.append((nx,ny))
    _distance_cache[goal] = dist
    return dist

def normalized_distance(pos: Tuple[int,int], goal: Tuple[int,int]) -> float:
    dist = bfs_distance_map(goal)
    d = dist[pos[1], pos[0]]
    if np.isinf(d): return 1.0
    return float(d / (2*GRIDSIZE))

# =========================
# Cooperative space-time A* teacher
# =========================
WAIT = (0, 0)
MOVE4_W = [(0,-1),(0,1),(-1,0),(1,0), WAIT]
MAX_T = 4000  # safety cap

def is_free_cell(x, y):
    return 0 <= x < GRIDSIZE and 0 <= y < GRIDSIZE and obstacle_map[y, x] == 0

def space_time_neighbors(x, y, t):
    for dx, dy in MOVE4_W:
        nx, ny = x + dx, y + dy
        nt = t + 1
        if is_free_cell(nx, ny):
            yield (nx, ny, nt)

def violates_reservation(x, y, t, nx, ny, nt, reservations_cell, reservations_edge):
    if (nx, ny, nt) in reservations_cell:
        return True
    # forbid swaps
    if ((x, y, t, nx, ny, nt) in reservations_edge) or ((nx, ny, t, x, y, nt) in reservations_edge):
        return True
    return False

def space_time_astar(start, goal, reservations_cell, reservations_edge, t_start=0, t_goal_slack=20):
    """A* in (x,y,t) with reservations. Returns list of (x,y,t)."""
    from heapq import heappush, heappop
    gx, gy = goal
    sx, sy = start
    h = lambda x, y: abs(x - gx) + abs(y - gy)
    openq = []
    gscore = {(sx, sy, t_start): 0}
    parent = {}
    heappush(openq, (h(sx, sy), 0, (sx, sy, t_start)))
    best_goal = None

    while openq:
        f, gc, (x, y, t) = heappop(openq)
        if (x, y) == (gx, gy):
            best_goal = (x, y, t)
            break
        if t > MAX_T:
            continue
        for nx, ny, nt in space_time_neighbors(x, y, t):
            if violates_reservation(x, y, t, nx, ny, nt, reservations_cell, reservations_edge):
                continue
            cost = gc + 1
            node = (nx, ny, nt)
            if node not in gscore or cost < gscore[node]:
                gscore[node] = cost
                parent[node] = (x, y, t)
                heappush(openq, (cost + h(nx, ny), cost, node))

    if best_goal is None:
        return []

    # reconstruct
    path_rev = []
    cur = best_goal
    while cur in parent:
        path_rev.append((cur[0], cur[1], cur[2]))
        cur = parent[cur]
    path_rev.append((cur[0], cur[1], cur[2]))
    path = list(reversed(path_rev))
    # linger at goal so others don't step into it
    for _ in range(t_goal_slack):
        last = path[-1]
        path.append((last[0], last[1], last[2] + 1))
    return path

def build_joint_plan_coop_astar(starts, goals, priority_order=None, t_goal_slack=20):
    """
    Plan agents sequentially with a reservation table to avoid conflicts.
    Returns: list of dicts: joint_plan[i][t] -> (x,y)
    """
    n = len(starts)
    if priority_order is None:
        priority_order = list(range(n))
    reservations_cell = set()
    reservations_edge = set()
    joint_paths = [None] * n
    for i in priority_order:
        s = starts[i]; g = goals[i]
        path_t = space_time_astar(s, g, reservations_cell, reservations_edge, t_start=0, t_goal_slack=t_goal_slack)
        if not path_t:
            # fallback: wait plan
            path_t = [(s[0], s[1], t) for t in range(0, 100)]
        joint_paths[i] = path_t
        # add reservations
        for (x, y, t1), (nx, ny, t2) in zip(path_t[:-1], path_t[1:]):
            reservations_cell.add((nx, ny, t2))
            reservations_edge.add((x, y, t1, nx, ny, t2))
    # convert to dicts of t->(x,y)
    joint_plan = []
    for i in range(n):
        d = {}
        for (x, y, t) in joint_paths[i]:
            d[t] = (x, y)
        joint_plan.append(d)
    return joint_plan

def teacher_action_from_plan(plan_i: Dict[int, Tuple[int,int]], t: int) -> Optional[Tuple[int,int]]:
    """Return movement (dx,dy) implied by teacher plan between t and t+1, else None."""
    if (t in plan_i) and (t+1 in plan_i):
        x0,y0 = plan_i[t]; x1,y1 = plan_i[t+1]
        return (x1-x0, y1-y0)
    return None

# =========================
# Start/Goal generation
# =========================
def generate_map_with_positions(min_distance=10, max_attempts=10000):
    H,W = obstacle_map.shape
    dockables=[]
    for y in range(H):
        for x in range(W):
            if obstacle_map[y,x]!=1: continue
            for dx,dy in FOUR:
                nx,ny=x+dx,y+dy
                if 0<=nx<W and 0<=ny<H and obstacle_map[ny,nx]==0:
                    dockables.append((x,y)); break
    if len(dockables) < 2*num_agents:
        raise RuntimeError("Not enough rack cells for starts/goals.")
    starts=[]; goals=[]; used=set(); attempts=0
    while len(starts)<num_agents and attempts<max_attempts:
        attempts+=1
        s,g = random.sample(dockables,2)
        if s in used or g in used: continue
        if euclid(s,g) < min_distance: continue
        grid = obstacle_map.copy()
        grid[s[1],s[0]]=0; grid[g[1],g[0]]=0
        if not astar(grid,s,g): continue
        starts.append(s); goals.append(g); used.add(s); used.add(g)
    if len(starts)<num_agents: raise RuntimeError("Pairing failed.")
    colored = np.zeros((H,W,3),dtype=np.float32)
    colored[obstacle_map==0]=[1,1,1]; colored[obstacle_map==1]=[0,0,0]
    for i in range(num_agents):
        sx,sy=starts[i]; gx,gy=goals[i]
        colored[sy,sx]=[0,0,1]; colored[gy,gx]=[1,0,0]
    return colored, starts, goals

# =========================
# Observations
# =========================
INPUT_CHANNELS = 4  # [obstacles, me, others, goal]

def get_state_for_agent(i: int, positions: List[Tuple[int,int]], goal: Tuple[int,int], *, as_tensor: bool = True):
    s = np.zeros((INPUT_CHANNELS, GRIDSIZE, GRIDSIZE), dtype=np.float32)
    s[0] = obstacle_map
    ax,ay = positions[i]
    if 0 <= ax < GRIDSIZE and 0 <= ay < GRIDSIZE:
        s[1,ay,ax] = 1.0
    for j,(x,y) in enumerate(positions):
        if j==i: continue
        if 0 <= x < GRIDSIZE and 0 <= y < GRIDSIZE:
            s[2,y,x] = 1.0
    gx,gy = goal; s[3,gy,gx] = 1.0
    if as_tensor:
        return torch.from_numpy(s).unsqueeze(0).to(device)
    return s

# =========================
# Dueling Double DQN (downsampled) + distance head
# =========================
ACTION_SIZE = len(ACTIONS)

class DuelingDQN(nn.Module):
    def __init__(self, in_ch, n_actions):
        super().__init__()
        # 20x20 -> 10x10 -> 5x5
        self.conv1 = nn.Conv2d(in_ch, 32, 3, stride=2, padding=1)  # 20 -> 10
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)     # 10 -> 5
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)    # 5 -> 5
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128*5*5, 256)
        self.val = nn.Linear(256, 1)
        self.adv = nn.Linear(256, n_actions)
        self.dist_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc(self.flat(x)))
        V = self.val(x)
        A = self.adv(x)
        q = V + (A - A.mean(dim=1, keepdim=True))
        dist = torch.sigmoid(self.dist_head(x))
        return q, dist

policy = DuelingDQN(INPUT_CHANNELS, ACTION_SIZE).to(device)
target = DuelingDQN(INPUT_CHANNELS, ACTION_SIZE).to(device)
target.load_state_dict(policy.state_dict())
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# =========================
# PER Buffer (stores demo flag & distance target)
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity:int):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    def add(self, transition, p:float):
        self.buffer.append(transition); self.priorities.append(p)
    def __len__(self): return len(self.buffer)
    def probabilities(self, alpha:float):
        p = np.array([float(x) for x in self.priorities], dtype=np.float32)
        p = np.power(p, alpha)
        s = p.sum()
        if s == 0.0:
            p = np.ones_like(p) / max(len(p), 1)
        else:
            p /= s
        return p
    def sample(self, batch, alpha, beta):
        n = min(len(self.buffer), batch)
        prob = self.probabilities(alpha)
        idxs = random.choices(range(len(self.buffer)), k=n, weights=prob)
        samples = [self.buffer[i] for i in idxs]
        iw = (len(self.buffer) * prob[idxs]) ** (-beta)
        iw = iw / max(iw.max(), 1e-8)

        s = torch.cat([t[0] for t in samples]).to(device)
        a = torch.tensor([t[1] for t in samples], dtype=torch.long, device=device).unsqueeze(1)
        r = torch.tensor([t[2] for t in samples], dtype=torch.float32, device=device).unsqueeze(1)
        ns= torch.cat([t[3] for t in samples]).to(device)
        d = torch.tensor([t[4] for t in samples], dtype=torch.float32, device=device).unsqueeze(1)
        demo = torch.tensor([t[5] for t in samples], dtype=torch.bool, device=device).unsqueeze(1)
        dist_t = torch.tensor([t[6] for t in samples], dtype=torch.float32, device=device).unsqueeze(1)
        iw = torch.tensor(iw, dtype=torch.float32, device=device).unsqueeze(1)
        return s,a,r,ns,d,demo,dist_t,iw,idxs
    def update(self, idxs, td, eps=1e-3):
        for i,t in zip(idxs, td):
            self.priorities[i] = float(abs(t)) + eps

memory = PrioritizedReplayBuffer(80000)
alpha = 0.6
beta_start, beta_end = 0.4, 1.0

# =========================
# Action masks / preferences
# =========================
def mask_invalid_actions(q: torch.Tensor, pos: Tuple[int,int], goal: Tuple[int,int]):
    x,y = pos; q = q.clone()
    for a,(dx,dy) in enumerate(ACTIONS):
        if (dx,dy)==(0,0): continue
        nx,ny=x+dx,y+dy
        if nx<0 or nx>=GRIDSIZE or ny<0 or ny>=GRIDSIZE:
            q[0,a] = -1e9; continue
        if obstacle_map[ny,nx]==1 and (nx,ny)!=goal:
            q[0,a] = -1e9
    return q

def prefer_nonworsening_moves(q: torch.Tensor, pos: Tuple[int,int], goal: Tuple[int,int], margin: float = 1e-3, boost: float = 4.0):
    q = q.clone()
    d0 = normalized_distance(pos, goal)
    nonworsen = False
    worsen_mask = []
    for a,(dx,dy) in enumerate(ACTIONS):
        if (dx,dy)==(0,0):
            worsen_mask.append(False); continue
        nx,ny = pos[0]+dx, pos[1]+dy
        if nx<0 or nx>=GRIDSIZE or ny<0 or ny>=GRIDSIZE:
            worsen_mask.append(False); continue
        if obstacle_map[ny,nx]==1 and (nx,ny)!=goal:
            worsen_mask.append(False); continue
        d1 = normalized_distance((nx,ny), goal)
        w = (d1 - d0) > margin
        worsen_mask.append(w)
        if not w: nonworsen = True
    if nonworsen:
        for a,w in enumerate(worsen_mask):
            if w: q[0,a] -= boost
        q[0,STOP_IDX] -= (boost + 3.0)
    return q

def mask_predicted_collisions(q: torch.Tensor, i: int, positions, chosen_so_far):
    """
    Mask actions that would step into cells reserved by earlier agents this tick.
    Simple, asymmetric reservation to reduce same-cell conflicts.
    """
    q = q.clone()
    my_pos = positions[i]
    reserved = set()
    for j, a in enumerate(chosen_so_far):
        if a is None or j == i:
            continue
        dx,dy = ACTIONS[a]
        reserved.add((positions[j][0]+dx, positions[j][1]+dy))
    for a,(dx,dy) in enumerate(ACTIONS):
        if (dx,dy) == (0,0): 
            continue
        nx,ny = my_pos[0]+dx, my_pos[1]+dy
        if (nx,ny) in reserved:
            q[0,a] = -1e9
    return q

# =========================
# Reward Shaping (Potential-based)
# =========================
GAMMA = 0.99
STEP_COST = -0.05
GOAL_BONUS = +20.0
HIT_PENALTY = -4.0     # wall/illegal
COLLISION_PENALTY = -3.0  # attempt contested move
YIELD_BONUS = +0.15       # chose STOP while someone passed through

def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def phi(pos, goal):  return -float(manhattan(pos, goal))

def step_single(agent_pos, action, goal):
    dx,dy = ACTIONS[action]
    if dx==0 and dy==0:
        base = STEP_COST
        shaped = GAMMA*phi(agent_pos, goal) - phi(agent_pos, goal)
        return agent_pos, base + shaped, False
    nx,ny = agent_pos[0]+dx, agent_pos[1]+dy
    if not (0<=nx<GRIDSIZE and 0<=ny<GRIDSIZE):
        return agent_pos, HIT_PENALTY, True
    if obstacle_map[ny,nx]==1 and (nx,ny)!=goal:
        return agent_pos, HIT_PENALTY, True
    new_pos = (nx,ny)
    base = STEP_COST
    if new_pos==goal:
        base += GOAL_BONUS
    shaped = GAMMA*phi(new_pos, goal) - phi(agent_pos, goal)
    return new_pos, base + shaped, False

# =========================
# Gymnasium environment wrapper
# =========================
class MultiAgentCBSEnv(gym.Env):
    """Gymnasium-compatible wrapper around the cooperative CBS gridworld."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        *,
        num_agents: int = num_agents,
        max_steps: int = 1500,
        starts: Optional[List[Tuple[int, int]]] = None,
        goals: Optional[List[Tuple[int, int]]] = None,
        colored_map: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.max_steps = max_steps
        self._initial_starts = [tuple(p) for p in starts] if starts is not None else None
        self._initial_goals = [tuple(p) for p in goals] if goals is not None else None
        self.colored_map = colored_map

        self.action_space = spaces.MultiDiscrete([ACTION_SIZE] * self.num_agents)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_agents, INPUT_CHANNELS, GRIDSIZE, GRIDSIZE),
            dtype=np.float32,
        )

        self.np_random, _ = seeding.np_random(seed)

        self.positions: List[Tuple[int, int]] = []
        self.goals: List[Tuple[int, int]] = []
        self.teacher_plan: List[Dict[int, Tuple[int, int]]] = []
        self.astar_paths: List[List[Tuple[int, int]]] = []
        self.terminated: Set[int] = set()
        self.agent_active_steps: List[int] = [0] * self.num_agents
        self.episode_collision_count = 0
        self.step_count = 0

    def _ensure_map(self, starts: List[Tuple[int, int]], goals: List[Tuple[int, int]]) -> None:
        if self.colored_map is not None:
            return
        colored = np.zeros((GRIDSIZE, GRIDSIZE, 3), dtype=np.float32)
        colored[obstacle_map == 0] = [1, 1, 1]
        colored[obstacle_map == 1] = [0, 0, 0]
        for i in range(self.num_agents):
            sx, sy = starts[i]
            gx, gy = goals[i]
            colored[sy, sx] = [0, 0, 1]
            colored[gy, gx] = [1, 0, 0]
        self.colored_map = colored

    def _compute_teacher_plan(self) -> List[Dict[int, Tuple[int, int]]]:
        priority = list(range(self.num_agents))
        self.np_random.shuffle(priority)
        return build_joint_plan_coop_astar(self.positions, self.goals, priority_order=priority, t_goal_slack=20)

    def _compute_astar_paths(self) -> List[List[Tuple[int, int]]]:
        refs = []
        for i in range(self.num_agents):
            grid = obstacle_map.copy()
            sx, sy = self.positions[i]
            gx, gy = self.goals[i]
            grid[sy, sx] = 0
            grid[gy, gx] = 0
            refs.append(astar(grid, self.positions[i], self.goals[i]))
        return refs

    def _build_observations(self) -> np.ndarray:
        obs = np.zeros((self.num_agents, INPUT_CHANNELS, GRIDSIZE, GRIDSIZE), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i] = get_state_for_agent(i, self.positions, self.goals[i], as_tensor=False)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        elif self.np_random is None:
            self.np_random, _ = seeding.np_random(None)

        options = options or {}
        starts_opt = options.get("starts")
        goals_opt = options.get("goals")
        color_opt = options.get("colored_map")

        if starts_opt is not None and goals_opt is not None:
            starts = [tuple(p) for p in starts_opt]
            goals = [tuple(p) for p in goals_opt]
            self._initial_starts = starts[:]
            self._initial_goals = goals[:]
            self.colored_map = color_opt if color_opt is not None else self.colored_map
        elif self._initial_starts is not None and self._initial_goals is not None:
            starts = self._initial_starts[:]
            goals = self._initial_goals[:]
        else:
            colored, starts, goals = generate_map_with_positions(min_distance=10)
            self._initial_starts = starts[:]
            self._initial_goals = goals[:]
            self.colored_map = colored

        self._ensure_map(starts, goals)

        self.positions = [tuple(p) for p in starts]
        self.goals = [tuple(g) for g in goals]
        self.teacher_plan = self._compute_teacher_plan()
        self.astar_paths = self._compute_astar_paths()
        self.terminated = set()
        self.agent_active_steps = [0] * self.num_agents
        self.episode_collision_count = 0
        self.step_count = 0

        observations = self._build_observations()
        info = {
            "starts": self.positions[:],
            "goals": self.goals[:],
            "positions": self.positions[:],
            "colored_map": self.colored_map,
            "teacher_plan": self.teacher_plan,
            "astar_paths": self.astar_paths,
        }
        return observations, info

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        elif not isinstance(actions, list):
            actions = list(actions)
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        prev_positions = self.positions[:]
        intended = [None] * self.num_agents
        rewards = [0.0] * self.num_agents
        hits = [False] * self.num_agents

        for i in range(self.num_agents):
            if i in self.terminated:
                intended[i] = self.positions[i]
                rewards[i] = 0.0
                hits[i] = False
                continue
            nxt, rew, hit = step_single(self.positions[i], actions[i], self.goals[i])
            intended[i] = nxt
            rewards[i] = rew
            hits[i] = hit
            self.agent_active_steps[i] += 1

        collision_agents = set()
        cell_to_agent: Dict[Tuple[int, int], int] = {}
        for idx, pos in enumerate(intended):
            if pos in cell_to_agent:
                collision_agents.add(idx)
                collision_agents.add(cell_to_agent[pos])
            else:
                cell_to_agent[pos] = idx

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if intended[i] == prev_positions[j] and intended[j] == prev_positions[i]:
                    collision_agents.add(i)
                    collision_agents.add(j)

        for idx in collision_agents:
            if actions[idx] == STOP_IDX:
                rewards[idx] += YIELD_BONUS
            else:
                rewards[idx] += COLLISION_PENALTY
            intended[idx] = prev_positions[idx]
            hits[idx] = True

        self.episode_collision_count += len(collision_agents)
        self.positions = [tuple(p) for p in intended]

        per_agent_done = [False] * self.num_agents
        for i in range(self.num_agents):
            if self.positions[i] == self.goals[i]:
                self.terminated.add(i)
                per_agent_done[i] = True

        self.step_count += 1
        terminated_episode = len(self.terminated) == self.num_agents
        truncated = self.step_count >= self.max_steps

        observations = self._build_observations()
        rewards_arr = np.array(rewards, dtype=np.float32)
        info = {
            "positions": self.positions[:],
            "hits": hits,
            "per_agent_done": per_agent_done,
            "collisions_this_step": len(collision_agents),
            "teacher_plan": self.teacher_plan,
            "astar_paths": self.astar_paths,
        }
        return observations, rewards_arr, terminated_episode, truncated, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Only rgb_array render mode is supported")
        if self.colored_map is None:
            raise RuntimeError("Environment map is not initialized. Call reset first.")
        img = np.array(self.colored_map, copy=True)
        for i, (x, y) in enumerate(self.positions):
            img[y, x] = [0.0, 1.0, 0.0]
        return img

    def close(self):
        return


def _render_frame(env: MultiAgentCBSEnv, scale: int = 1) -> Optional[np.ndarray]:
    """Return uint8 frame for video logging if imageio is available."""
    if not IMAGEIO_AVAILABLE:
        return None
    frame = env.render()
    if scale > 1:
        # Repeat pixels to magnify the grid for clearer GIFs without extra deps
        ones = np.ones((scale, scale, 1), dtype=frame.dtype)
        frame = np.kron(frame, ones)
    frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame_uint8

# =========================
# Pretraining on A* (single-agent BC)
# =========================
def pretrain_with_astar(env: MultiAgentCBSEnv, epochs=5):
    print("🔁 Pretraining on A* (BC)...")
    ce = nn.CrossEntropyLoss()
    if not env.positions or not env.goals:
        env.reset()
    for ep in range(epochs):
        cnt=0; total=0.0
        for i in range(env.num_agents):
            grid = obstacle_map.copy()
            sx, sy = env.positions[i]
            gx, gy = env.goals[i]
            grid[sy, sx]=0
            grid[gy, gx]=0
            path = astar(grid, env.positions[i], env.goals[i])
            if not path or len(path)<2: continue
            for t in range(len(path)-1):
                positions = [(-1,-1)] * env.num_agents  # others empty
                positions[i] = path[t]
                s = get_state_for_agent(i, positions, env.goals[i])
                move = (path[t+1][0]-path[t][0], path[t+1][1]-path[t][1])
                a = ACTIONS.index(move) if move in ACTIONS else STOP_IDX
                q,_ = policy(s)
                loss = ce(q, torch.tensor([a], device=device))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step(); cnt+=1; total+=float(loss.item())
        print(f"  epoch {ep+1}/{epochs} steps={cnt} loss={total/max(1,cnt):.4f}")
    target.load_state_dict(policy.state_dict())
    print("✅ Pretraining done.\n")

# =========================
# Training
# =========================
STALL_WINDOW = 5
STALL_BURST  = 3

def is_stalled(hist: deque, tol: float = 1e-3):
    if len(hist)<STALL_WINDOW: return False
    return abs(hist[0] - hist[-1]) < tol

def train(
    env: MultiAgentCBSEnv,
    total_episodes=1200,
    save_interval=1,
    record_gif_interval: Optional[int] = 25,
    max_render_frames: Optional[int] = 250,
    render_scale: int = 12,
):
    global _IMAGEIO_WARNED
    process = psutil.Process(os.getpid())

    # logging
    success_rates=[]; rewards_hist=[]; coll_rates=[]; avg_steps_hist=[]
    collisions_per_episode=[]

    # schedules (stronger anchor)
    epsilon_start, epsilon_end = 0.60, 0.05
    eps_decay_episodes = int(total_episodes*0.5)
    demo_ratio_start, demo_ratio_end = 0.40, 0.35
    beta = beta_start

    for episode in range(1, total_episodes+1):
        print(f"\n=== Episode {episode}/{total_episodes} ===")

        _, info = env.reset()
        starts = [tuple(p) for p in info["starts"]]
        goals = [tuple(p) for p in info["goals"]]
        colored_map = info["colored_map"]
        teacher_plan = info["teacher_plan"]
        astar_paths = info["astar_paths"]

        positions = [tuple(p) for p in info["positions"]]
        terminated:Set[int]=set()
        agent_paths = {i:[positions[i]] for i in range(env.num_agents)}
        dist_hist = {i: deque(maxlen=STALL_WINDOW) for i in range(env.num_agents)}
        stall_burst_left = {i: 0 for i in range(env.num_agents)}
        total_reward=0.0; step_count=0; done=False

        episode_collision_count=0
        agent_active_steps=[0]*env.num_agents

        should_record = (
            record_gif_interval is not None
            and record_gif_interval > 0
            and (episode % record_gif_interval == 0)
        )
        scale = max(1, int(render_scale))
        frames: List[np.ndarray] = []
        if should_record:
            if not IMAGEIO_AVAILABLE:
                if not _IMAGEIO_WARNED:
                    print("⚠️ imageio not available; skipping GIF recording.")
                    _IMAGEIO_WARNED = True
                should_record = False
            else:
                first_frame = _render_frame(env, scale=scale)
                if first_frame is not None:
                    frames.append(first_frame)

        frac = (episode-1)/max(1, eps_decay_episodes)
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start-epsilon_end)*frac)
        demo_ratio = max(demo_ratio_end, demo_ratio_start - (demo_ratio_start-demo_ratio_end)*min(1.0, (episode-1)/(0.8*total_episodes)))
        beta = min(1.0, beta + (beta_end-beta_start)/int(total_episodes/3))  # faster IW correction

        while not done:
            step_count+=1

            chosen=[STOP_IDX]*env.num_agents
            used_demo=[False]*env.num_agents

            # randomized ordering to reduce bias
            order = list(range(env.num_agents))
            random.shuffle(order)
            chosen_so_far = [None]*env.num_agents

            for i in order:
                if i in terminated:
                    continue

                s_i = get_state_for_agent(i, positions, goals[i])
                q_raw, _ = policy(s_i)
                q = mask_invalid_actions(q_raw, positions[i], goals[i])
                q = prefer_nonworsening_moves(q, positions[i], goals[i], margin=1e-3, boost=4.0)
                q = mask_predicted_collisions(q, i, positions, chosen_so_far)

                a=None; use_teacher=False

                # cooperative-teacher stall burst
                if stall_burst_left[i] > 0:
                    mv = teacher_action_from_plan(teacher_plan[i], step_count-1)  # last step idx
                    if mv is not None and mv in ACTIONS:
                        a = ACTIONS.index(mv); use_teacher=True
                        stall_burst_left[i] -= 1
                    else:
                        stall_burst_left[i] = 0
                        a = None
                elif is_stalled(dist_hist[i]):
                    mv = teacher_action_from_plan(teacher_plan[i], step_count-1)
                    if mv is not None and mv in ACTIONS:
                        a = ACTIONS.index(mv); use_teacher=True
                        stall_burst_left[i] = STALL_BURST - 1

                # If no stall-override, choose with demo/epsilon
                if a is None:
                    take_demo = (random.random() < demo_ratio)
                    if take_demo:
                        mv = teacher_action_from_plan(teacher_plan[i], step_count-1)
                        if mv is not None and mv in ACTIONS:
                            a = ACTIONS.index(mv); use_teacher=True
                    if a is None and astar_paths[i]:
                        # optional fallback to vanilla A* demo if aligned
                        try:
                            k = astar_paths[i].index(positions[i])
                            if k < len(astar_paths[i]) - 1 and random.random() < 0.1:
                                mv = (astar_paths[i][k+1][0]-positions[i][0],
                                      astar_paths[i][k+1][1]-positions[i][1])
                                if mv in ACTIONS:
                                    a = ACTIONS.index(mv); use_teacher=True
                        except ValueError:
                            pass

                if a is None:
                    if random.random() < epsilon:
                        legal = (q[0]>-1e8).nonzero(as_tuple=True)[0].tolist()
                        a = random.choice(legal) if legal else STOP_IDX
                    else:
                        a = q.argmax(dim=1).item()

                chosen[i] = a
                chosen_so_far[i] = a
                used_demo[i]=use_teacher
                agent_active_steps[i]+=1

            _, rewards_arr, terminated_episode, truncated, step_info = env.step(chosen)
            next_positions = [tuple(p) for p in step_info["positions"]]
            per_agent_done = step_info["per_agent_done"]

            # store transitions
            for i in range(env.num_agents):
                if i in terminated:
                    continue
                s_i = get_state_for_agent(i, positions, goals[i])
                ns_i= get_state_for_agent(i, next_positions, goals[i])
                d_i = float(per_agent_done[i])
                dist_t = normalized_distance(positions[i], goals[i])
                memory.add((s_i, chosen[i], float(rewards_arr[i]), ns_i, d_i, used_demo[i], dist_t), p=1.0)

            positions = next_positions[:]
            total_reward += float(rewards_arr.sum())
            episode_collision_count += step_info["collisions_this_step"]

            for i in range(env.num_agents):
                if i not in terminated:
                    agent_paths[i].append(positions[i])
                if per_agent_done[i]:
                    terminated.add(i)
                dist_hist[i].append(normalized_distance(positions[i], goals[i]))

            if should_record and (max_render_frames is None or len(frames) < max_render_frames):
                frame = _render_frame(env, scale=scale)
                if frame is not None:
                    frames.append(frame)

            done = terminated_episode or truncated

            # learn
            if len(memory) >= 4096:
                states, actions, rewards_b, next_states, dones, demo_flags, dist_targets, iw, idxs = \
                    memory.sample(256, alpha, beta)
                with torch.no_grad():
                    online_next, _ = policy(next_states)
                    na = online_next.argmax(dim=1, keepdim=True)
                    target_next, _ = target(next_states)
                    target_q = target_next.gather(1, na)
                    y = rewards_b + 0.99 * target_q * (1.0 - dones)

                q_all, dist_pred = policy(states)
                q_pred = q_all.gather(1, actions)

                loss_td = F.smooth_l1_loss(q_pred, y, reduction='none')
                loss = (iw * loss_td).mean()
                loss += 0.1 * F.mse_loss(dist_pred, dist_targets)
                if demo_flags.any():
                    ce = nn.CrossEntropyLoss()
                    mask = demo_flags[:,0]
                    loss += 0.10 * ce(q_all[mask], actions.squeeze(1)[mask])

                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()

                td = (y - q_pred).detach().cpu().numpy().flatten()
                memory.update(idxs, td)

                # soft update
                with torch.no_grad():
                    tau=0.005
                    for tp, p in zip(target.parameters(), policy.parameters()):
                        tp.data.mul_(1.0-tau).add_(tau*p.data)

        # episode metrics
        success_rate = len(terminated)/env.num_agents
        success_rates.append(success_rate)
        total_steps = sum(agent_active_steps)
        coll_rate = (episode_collision_count/max(1,total_steps))
        coll_rates.append(coll_rate)
        collisions_per_episode.append(episode_collision_count)
        rewards_hist.append(total_reward)
        avg_steps_hist.append(np.mean(agent_active_steps))

        print(f"Success: {success_rate:.3f} | Coll/step: {coll_rate:.3f} | Coll/Episode: {episode_collision_count} | Reward: {total_reward:.1f}")

        if should_record and frames:
            fps = env.metadata.get("render_fps", 5) or 5
            duration = 1.0 / max(fps, 1e-3)
            gif_path = os.path.join(EPISODE_IMG_DIR, f"episode_{episode:04d}.gif")
            imageio.mimsave(gif_path, frames, duration=duration)


        # --- snapshot plot (save every `save_interval` episodes) ---
        if episode % save_interval == 0:
            fig,ax=plt.subplots()
            ax.imshow(colored_map, origin="upper")
            for i in range(env.num_agents):
                xs,ys = zip(*agent_paths[i])
                ax.plot(xs,ys,color=AGENT_COLORS[i%len(AGENT_COLORS)],lw=1.5)
                ax.scatter(*starts[i], color=AGENT_COLORS[i%len(AGENT_COLORS)], s=40, edgecolors='black')
                ax.scatter(*goals[i], color=AGENT_COLORS[i%len(AGENT_COLORS)], s=40, edgecolors='black', marker='X')
            ax.set_title(f"Episode {episode} paths")
            ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(os.path.join(EPISODE_IMG_DIR, f"map_{episode}.png"), dpi=200)
            plt.close(fig)

    # -------- plots --------
    def save_curve(y, title, ylabel, fname):
        plt.figure(figsize=(10,5))
        plt.plot(y, alpha=0.9)
        plt.grid(True); plt.title(title); plt.xlabel("Episodes"); plt.ylabel(ylabel)
        plt.savefig(os.path.join(IMG_DIR, fname), dpi=220); plt.close()

    save_curve(avg_steps_hist,  "Training Convergence - Avg Steps (v2E-10)",         "Average Taking Steps",      "Average_Taking_Steps.png")
    save_curve(success_rates,   "Training Convergence - Success Rate (v2E-10)",      "Success Rate",              "Success_Rate.png")
    save_curve(rewards_hist,    "Training Convergence - Total Rewards (v2E-10)",     "Total Rewards",             "Total_Rewards.png")
    save_curve(coll_rates,      "Training Convergence - Collisions/Step (v2E-10)",   "Attempted Collisions / Step","Collision_Rates_PerStep.png")
    save_curve(collisions_per_episode, "Training Convergence - Collisions/Episode (v2E-10)", "Collisions per Episode", "Collision_Rates_PerEpisode.png")

    # summary xlsx
    df = pd.DataFrame({
        "SuccessRate": success_rates,
        "AvgSteps": avg_steps_hist,
        "CollisionsPerStep": coll_rates,
        "CollisionsPerEpisode": collisions_per_episode,
        "TotalReward": rewards_hist
    })
    xlsx = os.path.join(METRICS_DIR, f"{RUN_NAME}_metrics.xlsx")
    df.to_excel(xlsx, index=False)
    print(f"\n🏁 Done. Metrics saved to {xlsx}")
    print(f"🖼️ Images in {IMG_DIR}  (episode snapshots in {EPISODE_IMG_DIR})")

# =========================
# Main
# =========================
if __name__ == "__main__":
    colored_map, starts_init, goals_init = generate_map_with_positions(min_distance=10)
    env = MultiAgentCBSEnv(
        num_agents=num_agents,
        max_steps=1500,
        starts=starts_init,
        goals=goals_init,
        colored_map=colored_map,
    )
    env.reset()
    pretrain_with_astar(env, epochs=5)
    # save_interval=1 → save path image every episode
    train(
        env,
        total_episodes=1200,
        save_interval=1,
        record_gif_interval=5,
        max_render_frames=200,
        render_scale=16,
    )
