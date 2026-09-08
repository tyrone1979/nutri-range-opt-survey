"""
NutriOpt-Bench core utilities: seeds, budgets, metadata, status logging.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


def make_seed(base: int, exp: str, scale: int, target_id: str, algorithm: str, rep: int) -> int:
    payload = f"{base}|{exp}|{scale}|{target_id}|{algorithm}|{rep}".encode()
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    import random
    random.seed(seed)


@dataclass
class RunBudget:
    pop_size: int = 500
    max_evaluations: int = 100_000
    n_generations: int = 10_000  # soft cap; evals dominate
    early_stop_feasible: Optional[int] = None  # None = disabled


@dataclass
class RunMeta:
    experiment: str
    dataset: str
    scale: int
    dim: int
    sparsity: float
    algorithm: str
    target_id: str
    rep: int
    seed: int
    pop_size: int
    max_evaluations: int
    status: str = "ok"
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    software: Dict[str, str] = field(default_factory=dict)


def collect_software_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for mod in ("pandas", "scipy", "pymoo", "yaml", "pulp", "sklearn"):
        try:
            m = __import__(mod if mod != "yaml" else "yaml")
            if mod == "sklearn":
                import sklearn
                versions[mod] = sklearn.__version__
            elif mod == "yaml":
                versions["pyyaml"] = getattr(m, "__version__", "unknown")
            else:
                versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "missing"
    return versions


def write_env(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "software": collect_software_versions(),
                "cpu": platform.processor(),
            },
            f,
            indent=2,
        )


def append_log(log_dir: str, message: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "runs.log")
    with open(path, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def empty_metrics() -> Dict[str, Any]:
    return {
        "FHR": 0.0,
        "BestFit": 0.0,
        "FSC": 0,
        "Diversity": 0.0,
        "Contrib": 0.0,
        "CV": 1.0,
        "Entropy": 0.0,
        "FFG": 10**9,
        "FFG_evals": 10**9,
        "n_evals": 0,
        "n_foods": 0.0,
        "runtime": 0.0,
        "status": "error",
    }
