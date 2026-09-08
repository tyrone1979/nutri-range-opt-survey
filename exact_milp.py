"""
Exact MILP baseline for small RCDA instances using PuLP + CBC.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pulp
    PULP_OK = True
except ImportError:
    PULP_OK = False


def solve_rcda_milp(
    matrix: np.ndarray,
    target_array: np.ndarray,
    col_idx: List[int],
    meal_tags: Optional[np.ndarray] = None,
    min_foods: int = 3,
    max_foods: int = 30,
    has_meal_constraint: bool = False,
    time_limit: int = 60,
    candidate_limit: int = 400,
    seed: int = 42,
) -> Dict:
    """
    Binary selection MILP on a candidate subset (size <= candidate_limit).
    Objective: minimize nutrient violation soft slack (should be 0 if feasible),
    then maximize trivial 0 (feasibility focus).
    """
    t0 = time.time()
    out = {
        "status": "unavailable",
        "feasible": False,
        "optimal": False,
        "objective": None,
        "n_selected": 0,
        "runtime": 0.0,
        "violation": None,
        "message": "",
    }
    if not PULP_OK:
        out["message"] = "pulp not installed"
        return out

    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    if n > candidate_limit:
        # Prefer diversified sample
        idx = rng.choice(n, size=candidate_limit, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    A = matrix[idx][:, col_idx].astype(float)
    lo = target_array[:, 0].astype(float)
    hi = target_array[:, 1].astype(float)
    tags = meal_tags[idx] if meal_tags is not None else None

    prob = pulp.LpProblem("RCDA", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(idx))]
    # slacks for infeasibility detection
    s_lo = [pulp.LpVariable(f"slo_{j}", lowBound=0) for j in range(len(col_idx))]
    s_hi = [pulp.LpVariable(f"shi_{j}", lowBound=0) for j in range(len(col_idx))]

    for j in range(len(col_idx)):
        expr = pulp.lpSum(A[i, j] * x[i] for i in range(len(idx)))
        prob += expr + s_lo[j] >= lo[j]
        prob += expr - s_hi[j] <= hi[j]

    prob += pulp.lpSum(x) >= min_foods
    prob += pulp.lpSum(x) <= max_foods

    if has_meal_constraint and tags is not None:
        for m in (0, 1, 2):
            members = [i for i in range(len(idx)) if tags[i] == m]
            if members:
                prob += pulp.lpSum(x[i] for i in members) >= 1
            else:
                out["message"] = f"no foods for meal {m} in candidate set"
                out["runtime"] = time.time() - t0
                return out

    prob += pulp.lpSum(s_lo) + pulp.lpSum(s_hi)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, options=[f"randomSeed {seed}"])
    status = prob.solve(solver)
    runtime = time.time() - t0
    out["runtime"] = runtime
    out["status"] = pulp.LpStatus.get(status, str(status))

    if status not in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved, 1, 0):
        # CBC may return Optimal=1
        pass

    selected = [int(idx[i]) for i in range(len(idx)) if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]
    out["n_selected"] = len(selected)
    viol = float(sum(pulp.value(v) or 0.0 for v in s_lo) + sum(pulp.value(v) or 0.0 for v in s_hi))
    out["violation"] = viol
    out["objective"] = pulp.value(prob.objective)
    out["feasible"] = viol < 1e-5 and min_foods <= len(selected) <= max_foods
    out["optimal"] = out["status"] == "Optimal" and out["feasible"]
    out["selected"] = selected
    out["message"] = out["status"]
    return out
