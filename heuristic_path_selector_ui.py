#!/usr/bin/env python3
"""Heuristic-only Path Selector UI with editable weights and cost-map overlay."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from mrpp_selector import (
    GridView,
    MapGrid,
    ScoreWeights,
    _align_paths,
    _conflict_cells_for_candidate,
    astar,
    cost_components,
    demo_map,
)

try:  # Optional cost-map helper (analytic only)
    from costmap_utils import build_analytic_costmap  # type: ignore
except Exception:  # pragma: no cover
    build_analytic_costmap = None  # type: ignore

Coord = Tuple[int, int]
PathT = List[Coord]


class HeuristicSelectorWindow(QtWidgets.QWidget):
    COLORS = [
        QtGui.QColor(220, 20, 60),
        QtGui.QColor(30, 144, 255),
        QtGui.QColor(34, 139, 34),
        QtGui.QColor(238, 130, 238),
        QtGui.QColor(255, 140, 0),
        QtGui.QColor(0, 191, 255),
        QtGui.QColor(154, 205, 50),
        QtGui.QColor(255, 99, 71),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Heuristic Path Selector")
        self.resize(1320, 860)

        self.grid: MapGrid = demo_map()
        self.grid_view = GridView()
        self.grid_view.load_map(self.grid)
        self.grid_view.cellClicked.connect(self._handle_cell_click)

        self.num_robots: int = 0
        self.current_robot: int = 0
        self.current_mode: str = "start"
        self.starts: List[Optional[Coord]] = []
        self.goals: List[Optional[Coord]] = []
        self.paths: List[Optional[PathT]] = []
        self.path_costs: List[Optional[float]] = []
        self.jack_states: List[bool] = []
        self.candidates: List[PathT] = []
        self.components: List[Dict[str, float]] = []
        self.costs: List[float] = []
        self.candidate_conflicts: List[Dict[str, Set[Coord]]] = []
        self.selected_candidate_idx: Optional[int] = None
        self.cost_map = None
        self.last_goal: Optional[Coord] = None

        # Diversity configuration (successive penalty selector)
        self.min_jaccard_sep: float = 0.55
        self.weighted_astar_epsilon: float = 1.3
        self.penalty_lambda: float = 2.0
        self.selector_detour_cap: float = 1.3

        self._build_ui()
        self._reset_planning()

    # ----- UI construction -----
    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        layout.addWidget(self.grid_view, stretch=3)
        side = QtWidgets.QVBoxLayout()
        layout.addLayout(side, stretch=2)

        # Map controls
        map_box = QtWidgets.QGroupBox("Map")
        map_layout = QtWidgets.QHBoxLayout()
        self.btn_load_map = QtWidgets.QPushButton("Load map…")
        self.btn_load_map.clicked.connect(self._load_map)
        self.btn_reset = QtWidgets.QPushButton("Reset planning")
        self.btn_reset.clicked.connect(self._reset_planning)
        map_layout.addWidget(self.btn_load_map)
        map_layout.addWidget(self.btn_reset)
        map_box.setLayout(map_layout)
        side.addWidget(map_box)

        # Robot / selection controls
        robot_box = QtWidgets.QGroupBox("Robots & Selection")
        robot_layout = QtWidgets.QGridLayout()
        self.spin_robots = QtWidgets.QSpinBox()
        self.spin_robots.setRange(1, 12)
        self.spin_robots.setValue(3)
        self.spin_robots.valueChanged.connect(self._on_robot_count_changed)
        self.btn_pick_start = QtWidgets.QPushButton("Pick start")
        self.btn_pick_goal = QtWidgets.QPushButton("Pick goal")
        self.btn_pick_start.clicked.connect(lambda: self._set_mode("start"))
        self.btn_pick_goal.clicked.connect(lambda: self._set_mode("goal"))
        self.btn_undo = QtWidgets.QPushButton("Undo last robot")
        self.btn_undo.clicked.connect(self._undo_last)
        self.chk_jack_up = QtWidgets.QCheckBox("Jack-up engaged")
        self.chk_jack_up.setChecked(False)
        self.chk_jack_up.stateChanged.connect(self._on_jack_state_changed)
        robot_layout.addWidget(QtWidgets.QLabel("Robot count"), 0, 0)
        robot_layout.addWidget(self.spin_robots, 0, 1)
        robot_layout.addWidget(self.btn_pick_start, 1, 0)
        robot_layout.addWidget(self.btn_pick_goal, 1, 1)
        robot_layout.addWidget(self.btn_undo, 2, 0, 1, 2)
        robot_layout.addWidget(self.chk_jack_up, 3, 0, 1, 2)
        robot_box.setLayout(robot_layout)
        side.addWidget(robot_box)

        # Planning controls
        plan_box = QtWidgets.QGroupBox("Planning")
        plan_layout = QtWidgets.QFormLayout()
        self.spin_K = QtWidgets.QSpinBox()
        self.spin_K.setRange(1, 25)
        self.spin_K.setValue(6)
        plan_layout.addRow("K candidates", self.spin_K)

        self.spin_epsilon = QtWidgets.QDoubleSpinBox()
        self.spin_epsilon.setDecimals(2)
        self.spin_epsilon.setMinimum(1.0)
        self.spin_epsilon.setMaximum(sys.float_info.max)
        self.spin_epsilon.setSingleStep(0.05)
        self.spin_epsilon.setValue(self.weighted_astar_epsilon)
        self._bind_selector_spin(self.spin_epsilon, "weighted_astar_epsilon")
        plan_layout.addRow("Weighted A* ε", self.spin_epsilon)

        self.spin_penalty_lambda = QtWidgets.QDoubleSpinBox()
        self.spin_penalty_lambda.setDecimals(2)
        self.spin_penalty_lambda.setMinimum(0.0)
        self.spin_penalty_lambda.setMaximum(sys.float_info.max)
        self.spin_penalty_lambda.setSingleStep(0.25)
        self.spin_penalty_lambda.setValue(self.penalty_lambda)
        self._bind_selector_spin(self.spin_penalty_lambda, "penalty_lambda")
        plan_layout.addRow("Penalty λ", self.spin_penalty_lambda)

        self.spin_min_jaccard = QtWidgets.QDoubleSpinBox()
        self.spin_min_jaccard.setDecimals(2)
        self.spin_min_jaccard.setRange(0.0, 1.0)
        self.spin_min_jaccard.setSingleStep(0.05)
        self.spin_min_jaccard.setValue(self.min_jaccard_sep)
        self._bind_selector_spin(self.spin_min_jaccard, "min_jaccard_sep")
        plan_layout.addRow("Min Jaccard sep", self.spin_min_jaccard)

        self.spin_detour_cap = QtWidgets.QDoubleSpinBox()
        self.spin_detour_cap.setDecimals(2)
        self.spin_detour_cap.setMinimum(0.0)
        self.spin_detour_cap.setMaximum(sys.float_info.max)
        self.spin_detour_cap.setSingleStep(0.05)
        self.spin_detour_cap.setValue(self.selector_detour_cap)
        self._bind_selector_spin(self.spin_detour_cap, "selector_detour_cap")
        plan_layout.addRow("Detour cap", self.spin_detour_cap)

        plan_box.setLayout(plan_layout)
        side.addWidget(plan_box)

        # Heuristic weight controls
        weight_box = QtWidgets.QGroupBox("Heuristic weights (higher = stronger penalty)")
        weight_form = QtWidgets.QFormLayout()
        self.w_len = self._make_weight_spin(0.0, 10.0, 1.0, 0.1)
        self.w_turn = self._make_weight_spin(0.0, 10.0, 0.25, 0.05)
        self.w_cell = self._make_weight_spin(0.0, 50.0, 2.0, 0.5)
        self.w_path = self._make_weight_spin(0.0, 50.0, 0.0, 0.5)
        self.w_follow = self._make_weight_spin(0.0, 200.0, 5.0, 1.0)
        self.w_edge = self._make_weight_spin(0.0, 50.0, 1.0, 0.5)
        self.w_h2h = self._make_weight_spin(0.0, 200.0, 20.0, 1.0)
        self.w_dead = self._make_weight_spin(0.0, 500.0, 50.0, 5.0)
        self.w_self = self._make_weight_spin(0.0, 100.0, 5.0, 1.0)
        weight_form.addRow("w_len", self.w_len)
        weight_form.addRow("w_turn", self.w_turn)
        weight_form.addRow("w_overlap_cell", self.w_cell)
        weight_form.addRow("w_path_overlap", self.w_path)
        weight_form.addRow("w_follow", self.w_follow)
        weight_form.addRow("w_overlap_edge", self.w_edge)
        weight_form.addRow("w_h2h", self.w_h2h)
        weight_form.addRow("w_deadlock", self.w_dead)
        weight_form.addRow("w_self_cycle", self.w_self)
        weight_box.setLayout(weight_form)
        side.addWidget(weight_box)

        # Cost-map controls
        cost_box = QtWidgets.QGroupBox("Cost-map")
        cost_layout = QtWidgets.QVBoxLayout()
        self.chk_use_analytic = QtWidgets.QCheckBox("Enable analytic cost-map")
        self.chk_use_analytic.setChecked(True)
        self.chk_show_overlay = QtWidgets.QCheckBox("Show cost-map overlay")
        self.chk_show_overlay.setChecked(False)
        self.lbl_costmap = QtWidgets.QLabel("Cost-map: off")
        cost_layout.addWidget(self.chk_use_analytic)
        cost_layout.addWidget(self.chk_show_overlay)
        cost_layout.addWidget(self.lbl_costmap)
        cost_box.setLayout(cost_layout)
        side.addWidget(cost_box)

        # Candidate diagnostics
        cand_box = QtWidgets.QGroupBox("Candidates (auto-picks lowest cost)")
        cand_layout = QtWidgets.QVBoxLayout()
        self.btn_evaluate = QtWidgets.QPushButton("Evaluate now")
        self.btn_evaluate.clicked.connect(self._evaluate_candidates)
        self.btn_commit = QtWidgets.QPushButton("Commit selected")
        self.btn_commit.setEnabled(False)
        self.btn_commit.clicked.connect(self._on_commit_clicked)
        cand_buttons = QtWidgets.QHBoxLayout()
        cand_buttons.addWidget(self.btn_evaluate)
        cand_buttons.addWidget(self.btn_commit)
        cand_layout.addLayout(cand_buttons)
        self.table = QtWidgets.QTableWidget(0, 11)
        headers = [
            "#",
            "length",
            "turns",
            "cell_overlap",
            "path_overlap",
            "follow",
            "edge_overlap",
            "h2h",
            "deadlock",
            "self_cycle",
            "total_cost",
        ]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_candidate_clicked)
        cand_layout.addWidget(self.table)
        cand_box.setLayout(cand_layout)
        side.addWidget(cand_box, stretch=1)

        # Committed paths overview
        committed_box = QtWidgets.QGroupBox("Committed paths")
        committed_layout = QtWidgets.QVBoxLayout()
        self.list_committed = QtWidgets.QListWidget()
        committed_layout.addWidget(self.list_committed)
        committed_box.setLayout(committed_layout)
        side.addWidget(committed_box)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        side.addWidget(self.status_label)
        side.addStretch(1)

        self.chk_use_analytic.stateChanged.connect(lambda _: self._maybe_recompute_costmap())
        self.chk_show_overlay.stateChanged.connect(lambda _: self._refresh_scene())

    def _make_weight_spin(
        self,
        lo: float,
        hi: float,
        default: float,
        step: float,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        return spin

    def _bind_selector_spin(
        self, spin: QtWidgets.QDoubleSpinBox, attr_name: str
    ) -> None:
        def _update(value: float) -> None:
            setattr(self, attr_name, float(value))

        spin.valueChanged.connect(_update)

    # ----- State management -----
    def _reset_planning(self) -> None:
        self.num_robots = self.spin_robots.value()
        self.current_robot = 0
        self.current_mode = "start"
        self.starts = [None] * self.num_robots
        self.goals = [None] * self.num_robots
        self.paths = [None] * self.num_robots
        self.path_costs = [None] * self.num_robots
        self.jack_states = [False] * self.num_robots
        self._clear_candidates()
        self.cost_map = None
        self.last_goal = None
        self.list_committed.clear()
        self._clear_table()
        self._refresh_status()
        self._refresh_scene()
        self.lbl_costmap.setText("Cost-map: off")
        self._sync_jack_checkbox()

    def _clear_table(self) -> None:
        self.table.setRowCount(0)
        if hasattr(self, 'btn_commit'):
            self.btn_commit.setEnabled(False)

    def _clear_candidates(self) -> None:
        self.candidates = []
        self.components = []
        self.costs = []
        self.candidate_conflicts = []
        self.selected_candidate_idx = None
    def _refresh_status(self, text: Optional[str] = None) -> None:
        if text is not None:
            self.status_label.setText(text)
            return
        if self.current_robot >= self.num_robots:
            self.status_label.setText("All robots committed. Use Undo to revise a path.")
            return
        mode = "start" if self.current_mode == "start" else "goal"
        self.status_label.setText(f"Robot {self.current_robot + 1}: click on the grid to set the {mode}.")

    def _refresh_scene(self) -> None:
        if not self.grid:
            return
        self.grid_view.reset_visuals()
        # Draw committed paths first so they stay in the background
        for idx, path in enumerate(self.paths):
            if not path:
                continue
            color = self.COLORS[idx % len(self.COLORS)]
            self.grid_view.draw_planned(path, color)
            start = self.starts[idx]
            goal = self.goals[idx]
            if start:
                self.grid_view.draw_start_marker(start[0], start[1], color)
            if goal:
                self.grid_view.draw_end_marker(goal[0], goal[1], color)
        # Current selections (lighter tone) if the robot is still pending
        if self.current_robot < self.num_robots:
            color = self.COLORS[self.current_robot % len(self.COLORS)]
            start = self.starts[self.current_robot]
            goal = self.goals[self.current_robot]
            if start and not self.paths[self.current_robot]:
                self.grid_view.draw_start_marker(start[0], start[1], color.lighter(130))
            if goal and not self.paths[self.current_robot]:
                self.grid_view.draw_end_marker(goal[0], goal[1], color.lighter(130))
        if self.candidates:
            if self.current_robot < self.num_robots:
                highlight_color = self.COLORS[self.current_robot % len(self.COLORS)]
            else:
                highlight_color = QtGui.QColor(200, 200, 200)
            for idx, path in enumerate(self.candidates):
                if idx == self.selected_candidate_idx:
                    self.grid_view.draw_candidate(path)
                    self.grid_view.draw_highlight(path, highlight_color)
                else:
                    self.grid_view.draw_candidate(path)
            if (
                self.selected_candidate_idx is not None
                and 0 <= self.selected_candidate_idx < len(self.candidates)
                and 0 <= self.selected_candidate_idx < len(self.candidate_conflicts)
            ):
                conflict_sets = self.candidate_conflicts[self.selected_candidate_idx]
                path_cells = conflict_sets.get("path_overlap", set())
                edge_cells = conflict_sets.get("h2h", set())
                path_color = QtGui.QColor(255, 215, 0)
                h2h_color = QtGui.QColor(255, 140, 0)
                for rc in path_cells:
                    self.grid_view.draw_conflict_circle(rc, path_color)
                for rc in edge_cells:
                    self.grid_view.draw_conflict_cross(rc, h2h_color)
        if self.cost_map is not None and self.chk_show_overlay.isChecked():
            self.grid_view.draw_costmap_overlay(self.cost_map)

    # ----- Map loading -----
    def _load_map(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select map",
            str(Path.cwd()),
            "Text/YAML files (*.txt *.yaml *.yml)",
        )
        if not path_str:
            return
        try:
            if path_str.lower().endswith((".yaml", ".yml")):
                new_grid = MapGrid.from_yaml(path_str)
            else:
                new_grid = MapGrid.from_txt(path_str)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load map: {exc}")
            return
        self.grid = new_grid
        self.grid_view.load_map(self.grid)
        self._reset_planning()

    # ----- Robot workflow -----
    def _on_robot_count_changed(self, value: int) -> None:
        del value  # derived directly from spin box
        self._reset_planning()

    def _set_mode(self, mode: str) -> None:
        if self.current_robot >= self.num_robots:
            return
        self.current_mode = mode
        self._refresh_status()

    def _handle_cell_click(self, row: int, col: int) -> None:
        if not self.grid or self.current_robot >= self.num_robots:
            return
        if hasattr(self.grid, '_vertex_cells'):
            vset = getattr(self.grid, '_vertex_cells')
            if (row, col) not in vset:
                self.status_label.setText("Select a valid cell from the map.")
                return
        coord = (row, col)
        if self.current_mode == "start":
            self.starts[self.current_robot] = coord
            self.current_mode = "goal"
        else:
            self.goals[self.current_robot] = coord
        self._refresh_status()
        self._refresh_scene()
        self._evaluate_if_ready()

    def _evaluate_if_ready(self) -> None:
        if (
            self.current_robot < self.num_robots
            and self.starts[self.current_robot] is not None
            and self.goals[self.current_robot] is not None
        ):
            self._evaluate_candidates()

    def _collect_weights(self) -> ScoreWeights:
        return ScoreWeights(
            w_len=self.w_len.value(),
            w_turn=self.w_turn.value(),
            w_overlap_cell=self.w_cell.value(),
            w_path_overlap=self.w_path.value(),
            w_follow=self.w_follow.value(),
            w_overlap_edge=self.w_edge.value(),
            w_h2h=self.w_h2h.value(),
            w_deadlock=self.w_dead.value(),
            w_self_cycle=self.w_self.value(),
        )

    def _evaluate_candidates(self) -> None:
        if self.current_robot >= self.num_robots:
            return
        start = self.starts[self.current_robot]
        goal = self.goals[self.current_robot]
        if start is None or goal is None:
            self._refresh_status("Select both start and goal before evaluating candidates.")
            return

        requested_k = max(1, self.spin_K.value())

        jack_up = self.jack_states[self.current_robot] if self.current_robot < len(self.jack_states) else False
        allow: Set[Coord] = set()
        if jack_up:
            if self.grid.grid[start[0]][start[1]] == 1:
                allow.add(start)
            if self.grid.grid[goal[0]][goal[1]] == 1:
                allow.add(goal)
        else:
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.grid[r][c] == 1:
                        allow.add((r, c))

        committed = [path for idx, path in enumerate(self.paths) if idx != self.current_robot and path]
        base_cost_map = self._update_cost_map(goal, committed)

        candidates, diagnostics = self._generate_candidates_penalty(
            start,
            goal,
            allow if allow else None,
            base_cost_map,
            requested_k,
        )
        if not candidates:
            self._clear_table()
            self._clear_candidates()
            self.cost_map = None
            self.last_goal = goal
            self.lbl_costmap.setText("Cost-map: no feasible paths")
            self._refresh_status("No feasible paths found for this start/goal pair.")
            self._refresh_scene()
            return

        weights = self._collect_weights()
        self.candidates = []
        self.components = []
        self.costs = []
        self.candidate_conflicts = []
        for cand in candidates:
            comps, _ = cost_components(cand, committed, weights)
            path_cells: Set[Coord] = set()
            edge_cells: Set[Coord] = set()
            follow_cells: Set[Coord] = set()
            deadlock_cells: Set[Coord] = set()
            if committed:
                cand_aligned, others_aligned = _align_paths(cand, committed)
                (path_cells,
                 edge_cells,
                 follow_cells,
                 deadlock_cells) = _conflict_cells_for_candidate(
                    cand_aligned,
                    others_aligned,
                )
            comps["path_overlap"] = len(path_cells)
            comps["h2h"] = len(edge_cells)
            comps["follow"] = len(follow_cells)
            comps["deadlock"] = len(deadlock_cells)
            cost = (
                weights.w_len * comps.get("length", 0.0)
                + weights.w_turn * comps.get("turns", 0.0)
                + weights.w_overlap_cell * comps.get("cell_overlap", 0.0)
                + weights.w_overlap_edge * comps.get("edge_overlap", 0.0)
                + weights.w_path_overlap * comps.get("path_overlap", 0.0)
                + weights.w_follow * comps.get("follow", 0.0)
                + weights.w_h2h * comps.get("h2h", 0.0)
                + weights.w_deadlock * comps.get("deadlock", 0.0)
                + weights.w_self_cycle * comps.get("self_cycle", 0.0)
            )
            self.candidates.append(cand)
            self.components.append(comps)
            self.costs.append(cost)
            self.candidate_conflicts.append(
                {
                    "path_overlap": path_cells,
                    "h2h": edge_cells,
                    "follow": follow_cells,
                    "deadlock": deadlock_cells,
                }
            )
        if not self.candidates:
            self._clear_table()
            self._clear_candidates()
            self._refresh_scene()
            self._refresh_status("No feasible paths found for this start/goal pair.")
            return

        self.selected_candidate_idx = min(range(len(self.costs)), key=lambda i: self.costs[i])
        self._populate_table()
        if hasattr(self, 'btn_commit'):
            self.btn_commit.setEnabled(True)
        if self.selected_candidate_idx is not None:
            self.table.selectRow(self.selected_candidate_idx)
        self._refresh_scene()
        diag = diagnostics or {}
        diag_msg = (
            f"strategy={diag.get('strategy', 'penalty')}, "
            f"selected={len(self.candidates)}, "
            f"epsilon={diag.get('epsilon', self.weighted_astar_epsilon):.2f}, "
            f"lambda={diag.get('lam', self.penalty_lambda):.2f}, "
            f"min_sep={diag.get('min_jaccard_sep', self.min_jaccard_sep):.2f}, "
            f"detour≤{diag.get('max_detour_ratio', self.selector_detour_cap):.2f}×"
        )
        self._refresh_status(f"Review candidates and commit when ready. ({diag_msg})")

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.candidates))
        for row, (comps, cost) in enumerate(zip(self.components, self.costs)):
            values = [
                row + 1,
                comps.get("length", 0.0),
                comps.get("turns", 0.0),
                comps.get("cell_overlap", 0.0),
                comps.get("path_overlap", 0.0),
                comps.get("follow", 0.0),
                comps.get("edge_overlap", 0.0),
                comps.get("h2h", 0.0),
                comps.get("deadlock", 0.0),
                comps.get("self_cycle", 0.0),
                cost,
            ]
            for col, value in enumerate(values):
                if col == 0:
                    text = str(int(value))
                elif col == 10:
                    text = f"{value:.3f}"
                else:
                    text = str(int(value))
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, col, item)
        if self.selected_candidate_idx is not None:
            self.table.selectRow(self.selected_candidate_idx)

    def _commit_path(self, path: PathT, cost: float, best_idx: int) -> None:
        if self.current_robot >= self.num_robots:
            return
        robot_idx = self.current_robot
        self.paths[robot_idx] = path
        self.path_costs[robot_idx] = cost
        if 0 <= robot_idx < len(self.jack_states):
            self.jack_states[robot_idx] = self.chk_jack_up.isChecked()
        self._update_committed_list()
        self.current_robot += 1
        self.current_mode = "start"
        if self.current_robot < self.num_robots:
            self.starts[self.current_robot] = None
            self.goals[self.current_robot] = None
        self._sync_jack_checkbox()
        self._refresh_scene()
        if self.current_robot < self.num_robots:
            next_robot = self.current_robot + 1
            self._refresh_status(
                f"Committed best path (row {best_idx + 1}) with cost {cost:.3f}. "
                f"Click grid to set start for Robot {next_robot}."
            )
        else:
            self._refresh_status("All robots committed. Use Undo to revise a path.")

    def _on_candidate_clicked(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self.candidates):
            return
        self.selected_candidate_idx = row
        self.table.selectRow(row)
        self._refresh_scene()

    def _on_commit_clicked(self) -> None:
        if not self.candidates:
            return
        idx = self.selected_candidate_idx
        if idx is None or idx < 0 or idx >= len(self.candidates):
            idx = 0
        path = self.candidates[idx]
        cost = self.costs[idx]
        self._commit_path(path, cost, idx)
        self._clear_candidates()
        self._clear_table()
        self._refresh_scene()

    def _edge_set(self, path: PathT, directed: bool = False) -> Set[Tuple[Coord, Coord]]:
        edges: Set[Tuple[Coord, Coord]] = set()
        if not path or len(path) < 2:
            return edges
        for u, v in zip(path[:-1], path[1:]):
            if directed:
                edges.add((u, v))
            else:
                edge = (u, v) if u <= v else (v, u)
                edges.add(edge)
        return edges

    def _jaccard_edges(
        self,
        es_a: Set[Tuple[Coord, Coord]],
        es_b: Set[Tuple[Coord, Coord]],
    ) -> float:
        if not es_a and not es_b:
            return 0.0
        inter = es_a & es_b
        union = es_a | es_b
        if not union:
            return 0.0
        return len(inter) / len(union)

    def _dedupe_by_edges(self, paths: List[PathT], max_jaccard: float = 0.90) -> List[PathT]:
        kept: List[PathT] = []
        kept_edges: List[Set[Tuple[Coord, Coord]]] = []
        for path in paths:
            edges = self._edge_set(path)
            if all(self._jaccard_edges(edges, other) < max_jaccard for other in kept_edges):
                kept.append(path)
                kept_edges.append(edges)
        return kept

    def _generate_candidates_penalty(
        self,
        start: Coord,
        goal: Coord,
        allow: Optional[Set[Coord]],
        cost_map: Optional[object],
        K_target: int,
        *,
        lam: Optional[float] = None,
        epsilon: Optional[float] = None,
        max_detour_ratio: Optional[float] = None,
    ) -> Tuple[List[PathT], Dict[str, object]]:
        if K_target <= 0:
            return [], {"strategy": "penalty", "selected_count": 0}

        allow_set = set(allow) if allow else None

        def _cell_val(arr: Optional[object], rc: Coord) -> float:
            if arr is None:
                return 0.0
            r, c = rc
            try:
                return float(arr[r][c])  # type: ignore[index]
            except Exception:
                try:
                    return float(arr[r, c])  # type: ignore[index]
                except Exception:
                    return 0.0

        def _path_cost(path: PathT) -> float:
            if not path or len(path) < 2:
                return 0.0
            total = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                base_avg = 0.5 * (_cell_val(cost_map, u) + _cell_val(cost_map, v))
                total += 1.0 + base_avg
            return total

        eps = float(epsilon) if epsilon is not None else float(self.weighted_astar_epsilon)
        if eps < 1.0:
            eps = 1.0
        lam = float(lam) if lam is not None else float(self.penalty_lambda)
        detour_cap = (
            float(max_detour_ratio)
            if max_detour_ratio is not None
            else float(self.selector_detour_cap)
        )

        base_path = astar(
            self.grid,
            start,
            goal,
            allow_occupied=allow_set,
            cost_map=cost_map,
            edge_penalty=None,
            epsilon=eps,
        )
        if not base_path:
            return [], {"strategy": "penalty", "selected_count": 0, "error": "no_base_path"}

        base_len = _path_cost(base_path)
        detour_limit = float("inf") if base_len <= 1e-9 else detour_cap * base_len

        penalty: Dict[Tuple[Coord, Coord], float] = {}
        accepted: List[PathT] = []
        accepted_edges: List[Set[Tuple[Coord, Coord]]] = []
        seen: Set[Tuple[Coord, ...]] = set()

        diagnostics: Dict[str, object] = {
            "strategy": "penalty",
            "selected_count": 0,
            "lam": lam,
            "epsilon": eps,
            "max_detour_ratio": detour_cap,
            "min_jaccard_sep": self.min_jaccard_sep,
        }

        for _ in range(K_target):
            cand = astar(
                self.grid,
                start,
                goal,
                allow_occupied=allow_set,
                cost_map=cost_map,
                edge_penalty=penalty,
                epsilon=eps,
            )
            if not cand:
                break
            t = tuple(cand)
            if t in seen:
                edges = list(zip(cand[:-1], cand[1:]))
                if not edges:
                    break
                worst = max(edges, key=lambda e: penalty.get(e, 0.0))
                cand = astar(
                    self.grid,
                    start,
                    goal,
                    blocked_edges={worst},
                    allow_occupied=allow_set,
                    cost_map=cost_map,
                    edge_penalty=penalty,
                    epsilon=eps,
                )
                if not cand or tuple(cand) in seen:
                    break
                t = tuple(cand)

            if _path_cost(cand) > detour_limit:
                break

            cand_edges = self._edge_set(cand, directed=True)
            too_close = False
            for prev_edges in accepted_edges:
                jacc = self._jaccard_edges(prev_edges, cand_edges)
                if jacc > (1.0 - self.min_jaccard_sep):
                    edges = list(zip(cand[:-1], cand[1:]))
                    if not edges:
                        too_close = True
                        break
                    def reuse_score(edge: Tuple[Coord, Coord]) -> int:
                        return sum(1 for ed in accepted_edges if edge in ed)

                    worst = max(edges, key=reuse_score)
                    blocked = {worst}
                    cand = astar(
                        self.grid,
                        start,
                        goal,
                        blocked_edges=blocked,
                        allow_occupied=allow_set,
                        cost_map=cost_map,
                        edge_penalty=penalty,
                        epsilon=eps,
                    )
                    if not cand:
                        too_close = True
                        break
                    cand_edges = self._edge_set(cand, directed=True)
                    if any(
                        self._jaccard_edges(existing, cand_edges) > (1.0 - self.min_jaccard_sep)
                        for existing in accepted_edges
                    ):
                        too_close = True
                    break
            if too_close:
                continue

            accepted.append(cand)
            accepted_edges.append(cand_edges)
            seen.add(t)
            for edge in zip(cand[:-1], cand[1:]):
                penalty[edge] = penalty.get(edge, 0.0) + lam

        diagnostics["selected_count"] = len(accepted)
        return accepted, diagnostics

    def _update_committed_list(self) -> None:
        self.list_committed.clear()
        for idx, (path, cost) in enumerate(zip(self.paths, self.path_costs)):
            if not path or cost is None:
                continue
            length = max(0, len(path) - 1)
            mode = "jack-up" if (0 <= idx < len(self.jack_states) and self.jack_states[idx]) else "free"
            self.list_committed.addItem(f"Robot {idx + 1}: length={length}, cost={cost:.3f}, mode={mode}")

    def _undo_last(self) -> None:
        last_idx = None
        for idx in range(self.num_robots - 1, -1, -1):
            if self.paths[idx]:
                last_idx = idx
                break
        if last_idx is None:
            return
        self.paths[last_idx] = None
        self.path_costs[last_idx] = None
        self.current_robot = last_idx
        if self.goals[last_idx] is not None:
            self.current_mode = "goal"
        elif self.starts[last_idx] is not None:
            self.current_mode = "goal"
        else:
            self.current_mode = "start"
        self.cost_map = None
        self.last_goal = None
        self.lbl_costmap.setText("Cost-map: off")
        self._clear_candidates()
        self._clear_table()
        self._update_committed_list()
        self._refresh_scene()
        self._refresh_status("Re-plan this robot by adjusting the start/goal.")
        self._sync_jack_checkbox()
        self._evaluate_if_ready()

    def _update_cost_map(self, goal: Coord, committed: List[PathT]) -> Optional[object]:
        self.last_goal = goal
        if not self.chk_use_analytic.isChecked() or build_analytic_costmap is None:
            self.cost_map = None
            msg = "Cost-map: disabled" if not self.chk_use_analytic.isChecked() else "Cost-map: unavailable"
            self.lbl_costmap.setText(msg)
            return None
        try:
            cost_map = build_analytic_costmap(self.grid, committed)
        except Exception:
            self.cost_map = None
            self.lbl_costmap.setText("Cost-map: unavailable")
            return None
        self.cost_map = cost_map
        avg = float(cost_map.mean()) if cost_map.size else 0.0
        self.lbl_costmap.setText(f"Cost-map: Analytic avg={avg:.3f}")
        return cost_map

    def _maybe_recompute_costmap(self) -> None:
        if self.last_goal is None:
            self.cost_map = None
            self.lbl_costmap.setText("Cost-map: off")
            self._refresh_scene()
            return
        committed = [path for path in self.paths if path]
        self._update_cost_map(self.last_goal, committed)
        self._refresh_scene()

    def _on_jack_state_changed(self, state: int) -> None:
        if self.current_robot < self.num_robots:
            if self.current_robot >= len(self.jack_states):
                self.jack_states.extend([False] * (self.current_robot + 1 - len(self.jack_states)))
            self.jack_states[self.current_robot] = bool(state)
        self._refresh_scene()

    def _sync_jack_checkbox(self) -> None:
        if not hasattr(self, 'chk_jack_up'):
            return
        state = False
        if self.current_robot < len(self.jack_states):
            state = self.jack_states[self.current_robot]
        block = self.chk_jack_up.blockSignals(True)
        self.chk_jack_up.setChecked(state)
        self.chk_jack_up.blockSignals(block)

    # ----- Qt hooks -----
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: D401 - Qt override
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = HeuristicSelectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()