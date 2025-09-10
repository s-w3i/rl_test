# ranker_features.py
from typing import List, Tuple, Dict
from mrpp_selector import (
    Path, ScoreWeights, count_turns, undirected_edges,
    _align_paths, _count_path_overlap_full,
    _count_head_to_head_full, _count_deadlock_cycles_involving_candidate,
    _count_self_trivial_cycles
)

# lightweight weights for labeling (same as your UI defaults)
DEFAULT_W = ScoreWeights()

def compute_components(candidate: Path, others: List[Path]) -> Dict[str, float]:
    L = max(1, len(candidate)-1)
    turns = count_turns(candidate)
    cand_aligned, others_aligned = _align_paths(candidate, others)
    # time-synchronized node conflicts
    path_overlap = _count_path_overlap_full(cand_aligned, others_aligned)
    # static cell reuse across any time (unique cells)
    cells_cand = set(candidate)
    cells_others = set().union(*map(set, others)) if others else set()
    cell_overlap = len(cells_cand & cells_others)
    edge_overlap = len(undirected_edges(candidate) &
                       set().union(*map(undirected_edges, others))) if others else 0
    h2h = _count_head_to_head_full(cand_aligned, others_aligned)
    dead = _count_deadlock_cycles_involving_candidate(cand_aligned, others_aligned)
    self_cyc = _count_self_trivial_cycles(candidate)
    return {
        "length": L,
        "turns": turns,
        "cell_overlap": cell_overlap,
        "path_overlap": path_overlap,
        "edge_overlap": edge_overlap,
        "h2h": h2h,
        "deadlock": dead,
        "self_cycle": self_cyc,
    }

def compute_cost(components: Dict[str, float], w: ScoreWeights = DEFAULT_W) -> float:
    return (w.w_len*components["length"]
            + w.w_turn*components["turns"]
            + w.w_overlap_cell*components["cell_overlap"]
            + w.w_overlap_edge*components["edge_overlap"]
            + w.w_h2h*components["h2h"]
            + w.w_deadlock*components["deadlock"]
            + w.w_self_cycle*components["self_cycle"])
