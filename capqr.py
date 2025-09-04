#!/usr/bin/env python3
"""
CAPQR – Conflict‑Aware Path Quality Ranker (with map.txt support)
=================================================================

Fresh, self‑contained project to evaluate candidate paths for a target robot
in a grid‑based multi‑robot system. The model scores the *conflict risk* of a
candidate path given:
  • static map (obstacles)
  • other robots' spatio‑temporal trajectories
  • the candidate path trajectory of the target robot

It trains with a hybrid objective:
  1) Supervised regression of per‑type conflict counts (overlap & head‑to‑head swap)
  2) Pairwise listwise ranking: lower‑conflict paths should get lower scores

The script supports:
  • Loading a fixed grid from `--map map.txt` (0=free, 1=obstacle)
  • Synthetic data generation (random maps if `--map` is not given)
  • Candidate path generation via randomized, conflict‑aware A*
  • A compact 3D‑CNN model over (Time × Height × Width) rasters
  • Training loop with ranking + regression losses
  • Inference & visualization: choose the best path among K candidates and plot

`map.txt` format:
  Plain text with H rows and W columns of 0/1 characters. Example (5×5):
    00100
    00000
    11100
    00000
    00010

Usage examples
--------------
# Quick demo (generate one scenario on a provided map and plot):
python3 capqr.py --demo --map map.txt --K 8 --num-others 5 --seed 42

# Train on synthetic data using a fixed map:
python3 capqr.py --train --epochs 10 --scenarios-per-epoch 200 --K 6 \
                 --map map.txt --num-others 6 --save ckpt_capqr.pt

# Evaluate a trained checkpoint with a fresh random scenario on the same map:
python3 capqr.py --demo --map map.txt --K 8 --load ckpt_capqr.pt

Notes
-----
• This is a clean start (no dependency on prior PQN work).
• Designed to drop into a ROS2 pipeline later by swapping the data feeder:
  - Replace synthetic "make_scenario" with real map + other paths from topics.
• Time is discretized by steps; other robots *wait at goal* after arrival.
• Conflict types implemented:
  - Path Overlap (same cell, same time)
  - Head‑to‑Head Swap (A at u->v while B at v->u at the same t)
• The model predicts a scalar conflict score and auxiliary per‑type counts.

Author: gpt‑5‑thinking (2025‑09‑04)
"""

import argparse
import heapq
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------
# Reproducibility & Torch setup
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Types & helpers
# -----------------------------
Coord = Tuple[int, int]  # (y, x)


def neighbors(y: int, x: int, H: int, W: int):
    if y > 0: yield (y - 1, x)
    if y + 1 < H: yield (y + 1, x)
    if x > 0: yield (y, x - 1)
    if x + 1 < W: yield (y, x + 1)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# -----------------------------
# A* (with optional per-cell cost)
# -----------------------------

def astar(grid: np.ndarray, start: Coord, goal: Coord,
          cell_cost: Optional[np.ndarray] = None,
          tie_breaker: float = 1e-3) -> Optional[List[Coord]]:
    """4‑connected A*; obstacles are grid==1. Optional per‑cell cost adds to g.
       Returns path as list of (y,x) incl. start & goal; None if unreachable."""
    H, W = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return None
    if start == goal:
        return [start]

    if cell_cost is None:
        cell_cost = np.zeros_like(grid, dtype=np.float32)

    openh = []
    g = {start: 0.0}
    came = {start: None}
    f0 = manhattan(start, goal)
    heapq.heappush(openh, (f0, 0.0, start))

    while openh:
        _, gcur, cur = heapq.heappop(openh)
        if cur == goal:
            # reconstruct
            path = [cur]
            while came[cur] is not None:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        cy, cx = cur
        for ny, nx in neighbors(cy, cx, H, W):
            if grid[ny, nx] == 1:
                continue
            step = 1.0 + float(cell_cost[ny, nx])
            ng = gcur + step
            if (ny, nx) not in g or ng < g[(ny, nx)]:
                g[(ny, nx)] = ng
                came[(ny, nx)] = cur
                # tiny noise tie‑breaker encourages path diversity
                h = manhattan((ny, nx), goal) + tie_breaker * random.random()
                f = ng + h
                heapq.heappush(openh, (f, ng, (ny, nx)))
    return None

# -----------------------------
# Map loader
# -----------------------------

def load_map_from_txt(path: str) -> np.ndarray:
    lines = [line.strip() for line in open(path, 'r') if line.strip()]
    H = len(lines)
    W = len(lines[0])
    for line in lines:
        if len(line) != W:
            raise ValueError("All lines in map.txt must have equal width")
        if not all(ch in ('0', '1') for ch in line):
            raise ValueError("map.txt must contain only '0' and '1'")
    grid = np.zeros((H, W), dtype=np.uint8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            grid[y, x] = 1 if ch == '1' else 0
    return grid

# -----------------------------
# Scenario synthesis & candidate generation
# -----------------------------
@dataclass
class Scenario:
    grid: np.ndarray            # H×W (0=free, 1=obstacle)
    others: List[List[Coord]]   # list of other robots' paths
    start: Coord
    goal: Coord


def random_free_cell(grid: np.ndarray) -> Coord:
    H, W = grid.shape
    for _ in range(H * W * 2):
        y = random.randrange(H)
        x = random.randrange(W)
        if grid[y, x] == 0:
            return (y, x)
    # fallback linear scan
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise RuntimeError("No free cell in map")
    y, x = free[random.randrange(len(free))]
    return int(y), int(x)


def make_random_map(H: int, W: int, obstacle_ratio: float) -> np.ndarray:
    grid = (np.random.rand(H, W) < obstacle_ratio).astype(np.uint8)
    # ensure some holes
    for _ in range(max(H, W)):
        y = random.randrange(H)
        x = random.randrange(W)
        grid[y, x] = 0
    return grid


def extend_wait(path: List[Coord], T: int) -> List[Coord]:
    if not path:
        return []
    if len(path) >= T:
        return path[:T]
    tail = path[-1]
    return path + [tail] * (T - len(path))


def make_scenario(H: int, W: int, obstacle_ratio: float, num_others: int,
                  min_sep: int = 6, max_tries: int = 300,
                  map_grid: Optional[np.ndarray] = None) -> Scenario:
    grid = map_grid.copy() if map_grid is not None else make_random_map(H, W, obstacle_ratio)
    H, W = grid.shape

    # Build other robots' paths
    others: List[List[Coord]] = []
    tries = 0
    while len(others) < num_others and tries < max_tries:
        tries += 1
        s = random_free_cell(grid)
        g = random_free_cell(grid)
        if manhattan(s, g) < min_sep:
            continue
        p = astar(grid, s, g)
        if p is not None:
            others.append(p)
    # Target robot start/goal far apart
    s = random_free_cell(grid)
    g = random_free_cell(grid)
    sep_need = max(8, (H + W) // 4)
    guard = 0
    while manhattan(s, g) < sep_need and guard < 2000:
        s = random_free_cell(grid); g = random_free_cell(grid); guard += 1

    return Scenario(grid=grid, others=others, start=s, goal=g)


def build_cell_cost_from_others(grid: np.ndarray, others: List[List[Coord]],
                                decay: float = 0.9, base_penalty: float = 8.0,
                                temporal: bool = False) -> np.ndarray:
    """Compute a static per‑cell penalty heatmap from others' usage."""
    H, W = grid.shape
    cost = np.zeros((H, W), dtype=np.float32)
    for path in others:
        for t, (y, x) in enumerate(path):
            cost[y, x] += base_penalty * (decay ** t)
    if temporal:
        try:
            from scipy.signal import convolve2d  # optional
            k = np.array([[0.05, 0.1, 0.05],
                          [0.1,  0.4, 0.1 ],
                          [0.05, 0.1, 0.05]], dtype=np.float32)
            cost = convolve2d(cost, k, mode='same', boundary='symm')
        except Exception:
            pass
    cost[grid == 1] = 1e6
    return cost


def gen_candidates(grid: np.ndarray, start: Coord, goal: Coord,
                   others: List[List[Coord]], K: int = 6) -> List[List[Coord]]:
    """Generate K diverse candidates using randomized cost A* against others' heat."""
    base = build_cell_cost_from_others(grid, others, decay=0.9, base_penalty=8.0)

    cands: List[List[Coord]] = []
    seen_paths = set()
    attempts = 0
    while len(cands) < K and attempts < K * 30:
        attempts += 1
        # jitter cost to diversify
        noise = np.random.gamma(shape=2.0, scale=0.5, size=base.shape).astype(np.float32)
        cost = 0.7 * base + 0.3 * noise
        p = astar(grid, start, goal, cell_cost=cost, tie_breaker=1e-2)
        if p is None:
            continue
        key = tuple(p)
        if key in seen_paths:
            # discourage this path and replan
            tweak = base.copy()
            stride = max(1, len(p)//6)
            for (y, x) in p[1:-1: stride]:
                tweak[y, x] += 10.0
            p2 = astar(grid, start, goal, cell_cost=tweak, tie_breaker=5e-3)
            if p2 is None:
                continue
            key = tuple(p2)
            if key in seen_paths:
                continue
            cands.append(p2)
            seen_paths.add(key)
        else:
            cands.append(p)
            seen_paths.add(key)
    return cands

# -----------------------------
# Conflict counting (ground truth)
# -----------------------------
@dataclass
class ConflictCounts:
    overlap: int
    swap: int

    @property
    def total(self) -> int:
        return int(self.overlap + self.swap)


def extend_all(paths: List[List[Coord]], T: int) -> List[List[Coord]]:
    return [extend_wait(p, T) for p in paths]


def count_conflicts(candidate: List[Coord], others: List[List[Coord]]) -> ConflictCounts:
    if len(others) == 0:
        return ConflictCounts(0, 0)
    T = max(len(candidate), max(len(p) for p in others))
    cand = extend_wait(candidate, T)
    overlap = 0
    swap = 0
    others_ext = extend_all(others, T)
    for t in range(T):
        cy, cx = cand[t]
        # overlap: same cell same time
        for o in others_ext:
            if o[t] == (cy, cx):
                overlap += 1
        if t + 1 < T:
            cnext = cand[t + 1]
            for o in others_ext:
                if o[t] == cnext and o[t + 1] == (cy, cx):
                    swap += 1
    return ConflictCounts(overlap=overlap, swap=swap)

# -----------------------------
# Rasterization to (C,T,H,W) tensor
# -----------------------------
@dataclass
class Raster:
    tensor: torch.Tensor  # shape (C, T, H, W)
    T: int


def rasterize_ST(grid: np.ndarray, candidate: List[Coord], others: List[List[Coord]],
                 Tcap: int = 128) -> Raster:
    H, W = grid.shape
    T = min(max(len(candidate), max((len(p) for p in others), default=1)), Tcap)
    # Channels: [obstacles, candidate_occ, others_occ, others_dx, others_dy]
    C = 5
    arr = np.zeros((C, T, H, W), dtype=np.float32)

    # Obstacles broadcast over time
    for t in range(T):
        arr[0, t] = grid

    # Candidate occupancy (clip/pad)
    candT = extend_wait(candidate, T)
    for t in range(T):
        y, x = candT[t]
        arr[1, t, y, x] = 1.0

    # Others occupancy + average flow at each cell/time
    others_ext = extend_all(others, T)
    sum_dx = np.zeros((T, H, W), dtype=np.float32)
    sum_dy = np.zeros((T, H, W), dtype=np.float32)
    cnt = np.zeros((T, H, W), dtype=np.float32)

    for o in others_ext:
        for t in range(T):
            y, x = o[t]
            arr[2, t, y, x] = 1.0
            if t + 1 < T:
                ny, nx = o[t + 1]
                dx = float(nx - x)
                dy = float(ny - y)
            else:
                dx = 0.0
                dy = 0.0
            sum_dx[t, y, x] += dx
            sum_dy[t, y, x] += dy
            cnt[t, y, x] += 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_dx = np.where(cnt > 0, sum_dx / cnt, 0.0)
        avg_dy = np.where(cnt > 0, sum_dy / cnt, 0.0)
    arr[3] = avg_dx
    arr[4] = avg_dy

    ten = torch.from_numpy(arr)
    return Raster(tensor=ten, T=T)

# -----------------------------
# Model: 3D‑CNN ranker
# -----------------------------
class CAPQRNet(nn.Module):
    def __init__(self, in_ch: int = 5, width: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, width, kernel_size=3, padding=1),
            nn.BatchNorm3d(width), nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm3d(width), nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # downsample T,H,W

            nn.Conv3d(width, width*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(width*2), nn.GELU(),
            nn.Conv3d(width*2, width*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(width*2), nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # global avg pool
        )
        hid = width * 2
        self.head_total = nn.Sequential(
            nn.Flatten(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1)
        )
        self.head_overlap = nn.Sequential(
            nn.Flatten(), nn.Linear(hid, hid//2), nn.GELU(), nn.Linear(hid//2, 1)
        )
        self.head_swap = nn.Sequential(
            nn.Flatten(), nn.Linear(hid, hid//2), nn.GELU(), nn.Linear(hid//2, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.enc(x)
        total = self.head_total(feat)
        overlap = self.head_overlap(feat)
        swap = self.head_swap(feat)
        return {"total": total.squeeze(-1),
                "overlap": overlap.squeeze(-1),
                "swap": swap.squeeze(-1)}

# -----------------------------
# Losses: regression + pairwise ranking
# -----------------------------
@dataclass
class LossWeights:
    reg_total: float = 1.0
    reg_types: float = 0.5
    pairwise: float = 1.0


def pairwise_margin_ranking(scores: torch.Tensor, labels_total: torch.Tensor, margin: float = 1.0):
    """Compute pairwise margin ranking loss within a set of candidates.
       labels_total: int counts (lower is better). We want score_A < score_B if A has fewer conflicts.
    """
    n = scores.shape[0]
    if n < 2:
        return scores.new_tensor(0.0)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            yi = labels_total[i].item()
            yj = labels_total[j].item()
            if yi == yj:
                continue
            # target = +1 if score_i < score_j desired (yi < yj)
            target = 1.0 if yi < yj else -1.0
            pairs.append((i, j, target))
    if not pairs:
        return scores.new_tensor(0.0)
    si = torch.stack([scores[i] for (i, _, _) in pairs])
    sj = torch.stack([scores[j] for (_, j, _) in pairs])
    tgt = torch.tensor([t for (_, _, t) in pairs], dtype=scores.dtype, device=scores.device)
    loss = F.margin_ranking_loss(si, sj, tgt, margin=margin)
    return loss

# -----------------------------
# Training over synthetic scenarios
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 10
    scenarios_per_epoch: int = 200
    K: int = 6
    H: int = 20
    W: int = 20
    obstacle_ratio: float = 0.35
    num_others: int = 6
    Tcap: int = 128
    lr: float = 3e-4
    margin: float = 1.0
    seed: int = 42
    map_grid: Optional[np.ndarray] = None


def make_batch_from_scenario(sc: Scenario, K: int, Tcap: int):
    # Generate candidates and labels
    cands = gen_candidates(sc.grid, sc.start, sc.goal, sc.others, K)
    if len(cands) == 0:
        return None
    labels = [count_conflicts(p, sc.others) for p in cands]

    # Build tensors
    tens = []
    for path in cands:
        r = rasterize_ST(sc.grid, path, sc.others, Tcap=Tcap)
        tens.append(r.tensor)
    x = torch.stack(tens, dim=0)  # [K, C, T, H, W]
    y_total = torch.tensor([l.total for l in labels], dtype=torch.float32)
    y_overlap = torch.tensor([l.overlap for l in labels], dtype=torch.float32)
    y_swap = torch.tensor([l.swap for l in labels], dtype=torch.float32)
    return x.to(device), y_total.to(device), y_overlap.to(device), y_swap.to(device), cands, labels


def train_loop(cfg: TrainConfig, save_path: Optional[str] = None):
    set_seed(cfg.seed)
    model = CAPQRNet(in_ch=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    lw = LossWeights()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running_reg = 0.0
        running_pair = 0.0
        for _ in range(cfg.scenarios_per_epoch):
            sc = make_scenario(cfg.H, cfg.W, cfg.obstacle_ratio, cfg.num_others, map_grid=cfg.map_grid)
            batch = make_batch_from_scenario(sc, cfg.K, cfg.Tcap)
            if batch is None:
                continue
            x, yt, yo, ys, _cands, _labels = batch
            out = model(x)
            pred_total = out["total"].float()
            pred_overlap = out["overlap"].float()
            pred_swap = out["swap"].float()
            reg = lw.reg_total * F.mse_loss(pred_total, yt) 
            reg += lw.reg_types * (F.mse_loss(pred_overlap, yo) + F.mse_loss(pred_swap, ys))
            pair = lw.pairwise * pairwise_margin_ranking(pred_total, yt, margin=cfg.margin)
            loss = reg + pair

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_reg += float(reg.item())
            running_pair += float(pair.item())

        print(f"[epoch {ep}/{cfg.epochs}] reg={running_reg:.3f} pair={running_pair:.3f}")

    if save_path:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, save_path)
        print(f"Saved checkpoint to {save_path}")
    return model

# -----------------------------
# Inference & visualization
# -----------------------------
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def draw_grid(ax, grid: np.ndarray):
    H, W = grid.shape
    ax.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])


def plot_paths(ax, grid: np.ndarray, paths: List[List[Coord]], 
               labels: Optional[List[str]] = None, alphas: Optional[List[float]] = None,
               linewidths: Optional[List[float]] = None):
    for i, p in enumerate(paths):
        if len(p) == 0:
            continue
        xs = [x + 0.5 for (_, x) in p]
        ys = [y + 0.5 for (y, _) in p]
        c = COLORS[i % len(COLORS)]
        lw = 2.5 if linewidths is None else linewidths[i]
        a = 0.9 if alphas is None else alphas[i]
        ax.plot(xs, ys, '-', color=c, linewidth=lw, alpha=a)
        ax.scatter([xs[0]], [ys[0]], marker='o', color=c, s=20)
        ax.scatter([xs[-1]], [ys[-1]], marker='s', color=c, s=20)
        if labels is not None:
            mid = len(xs) // 2
            ax.text(xs[mid], ys[mid], labels[i], fontsize=8, color=c,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))


def pick_best(model: nn.Module, sc: Scenario, K: int, Tcap: int,
              return_all: bool = False):
    cands = gen_candidates(sc.grid, sc.start, sc.goal, sc.others, K)
    if len(cands) == 0:
        raise RuntimeError("No candidate paths found")
    tens = [rasterize_ST(sc.grid, p, sc.others, Tcap=Tcap).tensor for p in cands]
    x = torch.stack(tens, dim=0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        s = out["total"].detach().cpu().numpy()
        ov = out["overlap"].detach().cpu().numpy()
        sw = out["swap"].detach().cpu().numpy()
    order = np.argsort(s)  # lower is better
    cands_sorted = [cands[i] for i in order]
    scores_sorted = [(float(s[i]), float(ov[i]), float(sw[i])) for i in order]
    best = cands_sorted[0]
    if return_all:
        return best, cands_sorted, scores_sorted
    return best


def demo_once(model: Optional[nn.Module], sc: Scenario, K: int, Tcap: int):
    if model is None:
        model = CAPQRNet(in_ch=5).to(device)
    best, all_cands, scores = pick_best(model, sc, K=K, Tcap=Tcap, return_all=True)
    gt = [count_conflicts(p, sc.others) for p in all_cands]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_grid(ax, sc.grid)
    # Others as thin gray lines
    if len(sc.others) > 0:
        plot_paths(ax, sc.grid, sc.others, labels=None,
                   alphas=[0.35]*len(sc.others), linewidths=[1.2]*len(sc.others))

    # Candidates: faint; best in bold
    labels = []
    alphas = []
    lws = []
    for i, p in enumerate(all_cands):
        s_tot, s_ov, s_sw = scores[i]
        lab = f"#{i+1} pred={s_tot:.1f} (ov={s_ov:.1f}, sw={s_sw:.1f})\nGT={gt[i].total}"
        labels.append(lab)
        if p is best:
            alphas.append(1.0); lws.append(3.0)
        else:
            alphas.append(0.55); lws.append(2.0)
    plot_paths(ax, sc.grid, all_cands, labels=labels, alphas=alphas, linewidths=lws)

    sx, sy = sc.start[1] + 0.5, sc.start[0] + 0.5
    gx, gy = sc.goal[1] + 0.5, sc.goal[0] + 0.5
    ax.scatter([sx], [sy], marker='*', s=100, color='gold', edgecolor='black', zorder=5)
    ax.scatter([gx], [gy], marker='*', s=100, color='gold', edgecolor='black', zorder=5)
    ax.set_title("CAPQR demo: best (bold) + K candidates with predicted & GT scores")
    plt.tight_layout()
    plt.show()

# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="CAPQR – Conflict‑Aware Path Quality Ranker")
    p.add_argument('--train', action='store_true', help='Run training on synthetic data')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--scenarios-per-epoch', type=int, default=200)
    p.add_argument('--K', type=int, default=6, help='Number of candidate paths')
    p.add_argument('--H', type=int, default=20)
    p.add_argument('--W', type=int, default=20)
    p.add_argument('--num-others', type=int, default=6)
    p.add_argument('--obstacle-ratio', type=float, default=0.35)
    p.add_argument('--Tcap', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--margin', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', type=str, default='', help='Path to save checkpoint')
    p.add_argument('--load', type=str, default='', help='Load checkpoint for demo/eval')
    p.add_argument('--demo', action='store_true', help='Run a single demo scenario and plot')
    p.add_argument('--map', type=str, default='', help='Path to map.txt file (optional)')

    args = p.parse_args()

    set_seed(args.seed)

    # Load map if provided
    map_grid = None
    if args.map:
        map_grid = load_map_from_txt(args.map)
        H, W = map_grid.shape
        args.H, args.W = H, W
        print(f"Loaded map {args.map} of size {H}×{W}")

    # Training path
    if args.train:
        cfg = TrainConfig(epochs=args.epochs, scenarios_per_epoch=args.scenarios_per_epoch,
                          K=args.K, H=args.H, W=args.W, obstacle_ratio=args.obstacle_ratio,
                          num_others=args.num_others, Tcap=args.Tcap, lr=args.lr,
                          margin=args.margin, seed=args.seed, map_grid=map_grid)
        model = train_loop(cfg, save_path=args.save if args.save else None)
        if args.demo:
            sc = make_scenario(args.H, args.W, args.obstacle_ratio, args.num_others, map_grid=map_grid)
            demo_once(model, sc, K=args.K, Tcap=args.Tcap)
        return

    # Inference‑only path
    model = None
    if args.load:
        model = CAPQRNet(in_ch=5).to(device)
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint from {args.load}")

    if args.demo or (not args.train):
        sc = make_scenario(args.H, args.W, args.obstacle_ratio, args.num_others, map_grid=map_grid)
        demo_once(model, sc, K=args.K, Tcap=args.Tcap)


if __name__ == '__main__':
    main()
