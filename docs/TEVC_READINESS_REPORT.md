# TEVC Readiness Report

**Date:** 2026-09-09  
**Basis:** Complete E1–E4 PRIMARY + reconstructed manuscript

| Dimension | Score /10 | Notes |
|-----------|----------:|-------|
| 1. Scientific contribution | **8** | Clear difficulty-vs-ranking separation; strong E2/E4 empirics |
| 2. Evolutionary-computation relevance | **8** | Fair-budget MOEA/baseline comparison on combinatorial constrained search |
| 3. Problem formulation | **8** | Frozen RCFA; MILP calibration; soft/hard separation |
| 4. Benchmark design | **8** | Orthogonal axes; recipe/NHANES split; fairness audit |
| 5. Experimental rigor | **8** | 100k×10 PRIMARY; process-pool; checkpoint/resume |
| 6. Statistical rigor | **7** | Friedman + Kendall verified; limited multiple-testing discussion; no full Nemenyi in main |
| 7. Mechanistic insight | **5** | Rank patterns + limited probe; **no ablations** |
| 8. Reproducibility | **8** | Configs, runners, CSVs, reports; env logging |
| 9. Generalizability | **5** | Analogies stated; not experimentally transferred |
| 10. Manuscript quality | **7** | Evidence-aligned draft; survey still seed-level; bib sparse |

## Overall TEVC readiness: **72 / 100**

### Interpretation
Competitive as a **benchmark + empirical EC study** draft after PRIMARY completion.
Not submission-final until: denser EC related work, optional ablations/dynamics, camera-ready stats presentation, and completed literature screening if survey claims are strengthened.

### Do not inflate
Mechanistic insight and generalizability remain the weakest dimensions; the paper’s strength is controlled empirical separation of difficulty vs ranking, not a new algorithm or universal law.
