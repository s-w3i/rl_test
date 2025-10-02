#!/usr/bin/env python3
"""
Interactive Path Evaluation UI
------------------------------
Provide a lightweight PyQt interface to inspect K-shortest paths between a
start/goal pair on a grid map. For each candidate path the tool reports
heuristic costs (using manually tuned weights), optional learned weight costs
(via the weight regressor), and optional Learning-to-Rank model scores.

Usage: python path_evaluator_ui.py

Dependencies: PyQt5, numpy (for models), joblib (if loading models).
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from mrpp_selector import (
    MapGrid,
    ScoreWeights,
    cost_components,
    demo_map,
    yen_k_shortest,
    astar,
)
from ranker_infer import L2RRanker
from weight_infer import WeightPredictor
from context_features import MapGrid as CFMap, compute_context_features
from make_weight_dataset import RANGES as WEIGHT_RANGES, ORDER as WEIGHT_ORDER

Coord = Tuple[int, int]
PathT = List[Coord]


class PathCanvas(QtWidgets.QWidget):
    cellClicked = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._grid: Optional[MapGrid] = None
        self._cell_px = 28
        self._start: Optional[Coord] = None
        self._goal: Optional[Coord] = None
        self._committed: List[PathT] = []
        self._candidates: List[PathT] = []
        self._selected_idx: Optional[int] = None
        self.setMouseTracking(False)
        self.setMinimumSize(400, 400)

    def sizeHint(self) -> QtCore.QSize:
        if not self._grid:
            return QtCore.QSize(400, 400)
        return QtCore.QSize(self._grid.cols * self._cell_px, self._grid.rows * self._cell_px)

    def set_grid(self, grid: MapGrid):
        self._grid = grid
        self.setMinimumSize(self.sizeHint())
        self.update()

    def update_state(
        self,
        start: Optional[Coord],
        goal: Optional[Coord],
        committed: List[PathT],
        candidates: List[PathT],
        selected_idx: Optional[int],
    ):
        self._start = start
        self._goal = goal
        self._committed = committed
        self._candidates = candidates
        self._selected_idx = selected_idx
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: D401 - Qt override
        if not self._grid:
            return
        col = event.x() // self._cell_px
        row = event.y() // self._cell_px
        if row < 0 or col < 0 or row >= self._grid.rows or col >= self._grid.cols:
            return
        self.cellClicked.emit(int(row), int(col))

    def _draw_cell(self, painter: QtGui.QPainter, r: int, c: int):
        rect = QtCore.QRect(c * self._cell_px, r * self._cell_px, self._cell_px, self._cell_px)
        if not self._grid:
            return
        color = QtGui.QColor(55, 55, 55) if self._grid.grid[r][c] else QtGui.QColor(240, 240, 240)
        painter.fillRect(rect, color)
        painter.setPen(QtGui.QPen(QtGui.QColor(210, 210, 210)))
        painter.drawRect(rect)

    def _cell_center(self, rc: Coord) -> QtCore.QPointF:
        r, c = rc
        x = c * self._cell_px + self._cell_px / 2
        y = r * self._cell_px + self._cell_px / 2
        return QtCore.QPointF(x, y)

    def _draw_point(self, painter: QtGui.QPainter, rc: Coord, color: QtGui.QColor):
        center = self._cell_center(rc)
        radius = self._cell_px * 0.3
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(center, radius, radius)

    def _draw_path(self, painter: QtGui.QPainter, path: PathT, color: QtGui.QColor, width: int = 3):
        if len(path) < 2:
            return
        pen = QtGui.QPen(color, width)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        for i in range(len(path) - 1):
            a = self._cell_center(path[i])
            b = self._cell_center(path[i + 1])
            painter.drawLine(a, b)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D401 - Qt override
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        if not self._grid:
            return
        for r in range(self._grid.rows):
            for c in range(self._grid.cols):
                self._draw_cell(painter, r, c)
        # committed paths first (thin lines)
        for path in self._committed:
            self._draw_path(painter, path, QtGui.QColor(255, 165, 0, 220), width=2)
        # candidate overlay (selected thicker)
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._candidates):
            self._draw_path(painter, self._candidates[self._selected_idx], QtGui.QColor(80, 180, 255, 255), width=4)
        # start and goal markers
        if self._start:
            self._draw_point(painter, self._start, QtGui.QColor(0, 200, 0))
        if self._goal:
            self._draw_point(painter, self._goal, QtGui.QColor(200, 0, 0))


class PathEvaluationWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Path Evaluation Explorer")
        self.grid: MapGrid = demo_map()
        self.start: Optional[Coord] = None
        self.goal: Optional[Coord] = None
        self.committed_paths: List[PathT] = []
        self.candidates: List[PathT] = []
        self.selected_candidate: Optional[int] = None
        self.eval_rows: List[Dict[str, float]] = []
        self.click_mode: str = ""
        self.weight_predictor: Optional[WeightPredictor] = None
        self.predicted_weights: Optional[Dict[str, float]] = None
        self.ranker: Optional[L2RRanker] = None
        self.rank_scores: List[float] = []

        self._build_ui()
        self._load_default_weights()
        self._refresh_canvas()

    # ----- UI construction -----
    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # Canvas on the left inside a scroll area (supports bigger maps)
        self.canvas = PathCanvas(self)
        self.canvas.set_grid(self.grid)
        self.canvas.cellClicked.connect(self._handle_canvas_click)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll, stretch=3)

        # Control pane on the right
        ctrl = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl, stretch=2)

        # Map controls
        map_box = QtWidgets.QGroupBox("Map & Selection")
        map_layout = QtWidgets.QGridLayout()
        map_box.setLayout(map_layout)
        self.btn_load_map = QtWidgets.QPushButton("Load Map…")
        self.btn_load_map.clicked.connect(self._load_map_dialog)
        map_layout.addWidget(self.btn_load_map, 0, 0, 1, 2)

        self.btn_pick_start = QtWidgets.QPushButton("Pick Start")
        self.btn_pick_goal = QtWidgets.QPushButton("Pick Goal")
        self.btn_clear = QtWidgets.QPushButton("Clear Start/Goal")
        self.btn_pick_start.clicked.connect(lambda: self._set_click_mode("start"))
        self.btn_pick_goal.clicked.connect(lambda: self._set_click_mode("goal"))
        self.btn_clear.clicked.connect(self._clear_start_goal)
        map_layout.addWidget(self.btn_pick_start, 1, 0)
        map_layout.addWidget(self.btn_pick_goal, 1, 1)
        map_layout.addWidget(self.btn_clear, 2, 0, 1, 2)

        ctrl.addWidget(map_box)

        # Planning controls
        plan_box = QtWidgets.QGroupBox("Planning")
        plan_layout = QtWidgets.QGridLayout()
        plan_box.setLayout(plan_layout)
        plan_layout.addWidget(QtWidgets.QLabel("K candidates"), 0, 0)
        self.spin_K = QtWidgets.QSpinBox()
        self.spin_K.setRange(1, 25)
        self.spin_K.setValue(8)
        plan_layout.addWidget(self.spin_K, 0, 1)
        self.btn_eval = QtWidgets.QPushButton("Evaluate Paths")
        self.btn_eval.clicked.connect(self._evaluate_paths)
        plan_layout.addWidget(self.btn_eval, 1, 0, 1, 2)
        ctrl.addWidget(plan_box)

        # Weight controls
        weight_box = QtWidgets.QGroupBox("Manual Heuristic Weights")
        weight_form = QtWidgets.QFormLayout()
        weight_box.setLayout(weight_form)
        self.weight_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key in WEIGHT_ORDER:
            lo, hi = WEIGHT_RANGES[key]
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setSingleStep((hi - lo) / 50.0 if hi > lo else 0.1)
            spin.setRange(lo, hi)
            weight_form.addRow(key, spin)
            self.weight_spins[key] = spin
        ctrl.addWidget(weight_box)

        # Model loaders
        model_box = QtWidgets.QGroupBox("Models")
        model_layout = QtWidgets.QGridLayout()
        model_box.setLayout(model_layout)
        self.edit_weight_path = QtWidgets.QLineEdit(str(Path("weight_model.joblib")))
        self.btn_weight_load = QtWidgets.QPushButton("Load Weight Model")
        self.btn_weight_load.clicked.connect(self._load_weight_model)
        self.label_weight_status = QtWidgets.QLabel("(none)")
        model_layout.addWidget(QtWidgets.QLabel("Weight model"), 0, 0)
        model_layout.addWidget(self.edit_weight_path, 0, 1)
        model_layout.addWidget(self.btn_weight_load, 0, 2)
        model_layout.addWidget(self.label_weight_status, 0, 3)

        self.edit_ranker_path = QtWidgets.QLineEdit(str(Path("ranker_model.joblib")))
        self.btn_ranker_load = QtWidgets.QPushButton("Load Ranker Model")
        self.btn_ranker_load.clicked.connect(self._load_ranker_model)
        self.label_ranker_status = QtWidgets.QLabel("(none)")
        model_layout.addWidget(QtWidgets.QLabel("Ranker model"), 1, 0)
        model_layout.addWidget(self.edit_ranker_path, 1, 1)
        model_layout.addWidget(self.btn_ranker_load, 1, 2)
        model_layout.addWidget(self.label_ranker_status, 1, 3)

        ctrl.addWidget(model_box)

        # Candidate table
        self.table = QtWidgets.QTableWidget(0, 13)
        headers = [
            "#","length","turns","cell_overlap","path_overlap","follow","edge_overlap",
            "h2h","deadlock","self_cycle","cost_manual","cost_pred","ranker_score"
        ]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.cellClicked.connect(self._on_candidate_clicked)
        ctrl.addWidget(self.table)

        # Committed paths controls
        committed_box = QtWidgets.QGroupBox("Committed Paths")
        committed_layout = QtWidgets.QVBoxLayout()
        committed_box.setLayout(committed_layout)
        self.list_committed = QtWidgets.QListWidget()
        committed_layout.addWidget(self.list_committed)
        btns = QtWidgets.QHBoxLayout()
        self.btn_add_committed = QtWidgets.QPushButton("Add selected")
        self.btn_remove_committed = QtWidgets.QPushButton("Remove selected")
        self.btn_clear_committed = QtWidgets.QPushButton("Clear all")
        self.btn_add_committed.clicked.connect(self._add_committed_path)
        self.btn_remove_committed.clicked.connect(self._remove_committed_path)
        self.btn_clear_committed.clicked.connect(self._clear_committed_paths)
        btns.addWidget(self.btn_add_committed)
        btns.addWidget(self.btn_remove_committed)
        btns.addWidget(self.btn_clear_committed)
        committed_layout.addLayout(btns)
        ctrl.addWidget(committed_box)

        # Status info
        self.label_predicted = QtWidgets.QLabel("Predicted weights: (none)")
        self.label_predicted.setWordWrap(True)
        ctrl.addWidget(self.label_predicted)

        ctrl.addStretch(1)

    def _load_default_weights(self):
        defaults = ScoreWeights()
        for key in WEIGHT_ORDER:
            val = getattr(defaults, key)
            spin = self.weight_spins[key]
            spin.setValue(val)

    # ----- Canvas helpers -----
    def _refresh_canvas(self):
        self.canvas.update_state(
            self.start,
            self.goal,
            self.committed_paths,
            self.candidates,
            self.selected_candidate,
        )
        self.canvas.update()

    def _set_click_mode(self, mode: str):
        self.click_mode = mode
        if mode:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Click on map to set {mode}")

    def _handle_canvas_click(self, row: int, col: int):
        if not self.grid or self.grid.grid[row][col]:
            return  # ignore obstacles
        coord = (row, col)
        if self.click_mode == "start":
            self.start = coord
        elif self.click_mode == "goal":
            self.goal = coord
        self.click_mode = ""
        self._refresh_canvas()

    def _clear_start_goal(self):
        self.start = None
        self.goal = None
        self.candidates = []
        self.selected_candidate = None
        self.eval_rows = []
        self.table.setRowCount(0)
        self._refresh_canvas()

    # ----- Map loading -----
    def _load_map_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select map file",
            str(Path.cwd()),
            "Text Maps (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = [list(line.strip()) for line in f if line.strip()]
            grid = [[1 if ch == "1" else 0 for ch in row] for row in rows]
            self.grid = MapGrid(grid)
            self.canvas.set_grid(self.grid)
            self._clear_start_goal()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Map load failed", str(exc))

    # ----- Evaluation -----
    def _manual_weights(self) -> ScoreWeights:
        vals = {key: self.weight_spins[key].value() for key in WEIGHT_ORDER}
        return ScoreWeights(**vals)

    def _cf_astar(self, cf_grid: CFMap, start: Coord, goal: Coord) -> Optional[PathT]:
        mg = MapGrid(cf_grid.grid)
        return astar(mg, start, goal)

    def _evaluate_paths(self):
        if not self.start or not self.goal:
            QtWidgets.QMessageBox.information(self, "Missing", "Pick both start and goal cells first.")
            return
        K = self.spin_K.value()
        self.candidates = yen_k_shortest(self.grid, self.start, self.goal, K=K)
        if not self.candidates:
            QtWidgets.QMessageBox.warning(self, "No paths", "Unable to find any feasible paths between start and goal.")
            self.eval_rows = []
            self.table.setRowCount(0)
            self._refresh_canvas()
            return

        manual_w = self._manual_weights()

        pred_w_obj: Optional[ScoreWeights] = None
        self.predicted_weights = None
        if self.weight_predictor:
            try:
                cf_features = compute_context_features(
                    CFMap(self.grid.grid),
                    self.committed_paths,
                    self.start,
                    self.goal,
                    self.candidates,
                    astar_fn=self._cf_astar,
                )
                self.predicted_weights = self.weight_predictor.predict(cf_features)
                pred_w_obj = ScoreWeights(**self.predicted_weights)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Weight prediction failed", str(exc))
                self.predicted_weights = None
                pred_w_obj = None
        if self.predicted_weights:
            weights_text = ", ".join(f"{k}={self.predicted_weights[k]:.3f}" for k in WEIGHT_ORDER)
            self.label_predicted.setText(f"Predicted weights: {weights_text}")
        else:
            self.label_predicted.setText("Predicted weights: (none)")

        self.rank_scores = []
        if self.ranker:
            try:
                self.rank_scores = self.ranker.score_candidates(self.candidates, self.committed_paths)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Ranker failed", str(exc))
                self.rank_scores = []

        rows: List[Dict[str, float]] = []
        for idx, path in enumerate(self.candidates):
            comp, cost_manual = cost_components(path, self.committed_paths, manual_w)
            row = {
                "index": idx,
                "length": comp["length"],
                "turns": comp["turns"],
                "cell_overlap": comp["cell_overlap"],
                "path_overlap": comp["path_overlap"],
                "follow": comp["follow"],
                "edge_overlap": comp["edge_overlap"],
                "h2h": comp["h2h"],
                "deadlock": comp["deadlock"],
                "self_cycle": comp["self_cycle"],
                "cost_manual": cost_manual,
                "cost_pred": float("nan"),
                "ranker_score": float("nan"),
            }
            if pred_w_obj:
                _, cost_pred = cost_components(path, self.committed_paths, pred_w_obj)
                row["cost_pred"] = cost_pred
            if self.rank_scores and idx < len(self.rank_scores):
                row["ranker_score"] = self.rank_scores[idx]
            rows.append(row)

        self.eval_rows = rows
        self._populate_table()
        self.selected_candidate = 0
        if rows:
            self.table.selectRow(0)
        self._refresh_canvas()

    def _populate_table(self):
        self.table.setRowCount(len(self.eval_rows))
        for r, row in enumerate(self.eval_rows):
            vals = [
                row["index"] + 1,
                row["length"],
                row["turns"],
                row["cell_overlap"],
                row["path_overlap"],
                row["follow"],
                row["edge_overlap"],
                row["h2h"],
                row["deadlock"],
                row["self_cycle"],
                row["cost_manual"],
                row["cost_pred"],
                row["ranker_score"],
            ]
            for c, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem()
                if isinstance(val, (float, int)):
                    if isinstance(val, float) and (val != val):  # NaN
                        item.setText("-")
                    else:
                        item.setText(f"{val:.3f}")
                        item.setData(QtCore.Qt.UserRole, float(val))
                else:
                    item.setText(str(val))
                if c == 0:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(r, c, item)
        self.table.resizeRowsToContents()

    # ----- Candidate interaction -----
    def _on_candidate_clicked(self, row: int, col: int):  # noqa: D401 - slot
        if 0 <= row < len(self.candidates):
            self.selected_candidate = row
            self._refresh_canvas()

    # ----- Committed path management -----
    def _add_committed_path(self):
        if self.selected_candidate is None or self.selected_candidate >= len(self.candidates):
            QtWidgets.QMessageBox.information(self, "Select candidate", "Select a candidate row first.")
            return
        path = list(self.candidates[self.selected_candidate])
        self.committed_paths.append(path)
        self.list_committed.addItem(f"#{len(self.committed_paths)} len={len(path)-1}")
        self._refresh_canvas()

    def _remove_committed_path(self):
        row = self.list_committed.currentRow()
        if row < 0 or row >= len(self.committed_paths):
            return
        self.list_committed.takeItem(row)
        self.committed_paths.pop(row)
        self._refresh_canvas()

    def _clear_committed_paths(self):
        self.list_committed.clear()
        self.committed_paths.clear()
        self._refresh_canvas()

    # ----- Model loading -----
    def _load_weight_model(self):
        path = Path(self.edit_weight_path.text()).expanduser()
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", f"Weight model not found: {path}")
            return
        try:
            self.weight_predictor = WeightPredictor(str(path))
            self.label_weight_status.setText("loaded")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            self.weight_predictor = None
            self.label_weight_status.setText("(none)")

    def _load_ranker_model(self):
        path = Path(self.edit_ranker_path.text()).expanduser()
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", f"Ranker model not found: {path}")
            return
        try:
            self.ranker = L2RRanker(str(path))
            self.label_ranker_status.setText("loaded")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            self.ranker = None
            self.label_ranker_status.setText("(none)")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PathEvaluationWindow()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
