# REPRODUCIBILITY.md

## Environment (fill/refresh by running)

```bash
python3 -c "from benchmark_core import write_env; write_env('results/reproducibility/env.json')"
```

See `results/reproducibility/env.json` after runs.

## Key commands

```bash
# Data validation
python3 validate_data.py

# Target coverage
python3 analyze_target_coverage.py

# Feasible-region density (E2 geometry; sanitized bounds)
python3 estimate_feasibility.py --config configs/e2_density.yaml

# E2 sparsity (fair budget)
python3 -u run_e2.py --config configs/e2_pilot.yaml --stage pilot
python3 -u run_e2.py --config configs/e2_primary_lite.yaml --stage main
python3 -u run_e2.py --config configs/e2_full.yaml --stage main

# Search-dynamics probe (BMOEA vs Random trajectories)
python3 run_dynamics_probe.py --scale 2000 --evals 2400 --target-index 3

# Exact MILP (small)
python3 run_exact.py configs/exact_small.yaml

# Profiling
python3 run_benchmark.py --config configs/profile.yaml --stage profile

# Pilot
python3 run_benchmark.py --config configs/e1_pilot.yaml --stage pilot

# Full E1 (expensive — only after pilot OK; use process pool)
python3 -u run_benchmark.py --config configs/e1_full.yaml --stage main --workers 24

# E3 dimension PRIMARY
python3 -u run_e3.py --config configs/e3_full.yaml --stage main --workers 24

# After E1+E3: E2 full (or use watch_then_e2_full.py)
python3 -u run_e2.py --config configs/e2_full.yaml --stage main --workers 32
```

## TEVC Phase II docs

| File | Role |
|------|------|
| `docs/TEVC_RECONSTRUCTION_AUDIT.md` | STEP 1 audit |
| `docs/RCFA_FORMULATION.md` | Frozen problem regime |
| `docs/ALGORITHM_MECHANISM_MATRIX.md` | Mechanism taxonomy |
| `docs/E2_SPARSITY_PROTOCOL.md` | E2 sanitized sparsity rules |
| `docs/PHASE2_STEP3_NOTE.md` | STEP 3 completion note |


## Configs

| File | Role |
|------|------|
| `config_debug.yaml` / `config.yaml` | Legacy debug (preserved) |
| `configs/profile.yaml` | Stage 1 |
| `configs/e1_pilot.yaml` | Stage 2 |
| `configs/e1_full.yaml` | Stage 3 primary |
| `configs/e2_full.yaml` / `e3_full.yaml` | Sparsity / dimension |
| `configs/exact_small.yaml` | MILP |
| `configs/ablation.yaml` | Ablations |

## Fairness invariants

- Identical `population_size` and `max_evaluations` across algorithms
- `early_stop_feasible: null` in production configs
- Per-run seeds via `benchmark_core.make_seed`
- Failed runs logged under `results/logs/` with `status=error`

## Manuscript numbers

Do **not** hand-type experimental numbers. Prefer CSV → table scripts → LaTeX `\input` or generated `results/tables/*.tex`.
