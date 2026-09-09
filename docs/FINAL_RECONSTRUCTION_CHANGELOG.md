# FINAL RECONSTRUCTION CHANGELOG (TEVC)

Date: 2026-09-09  
Manuscript: `manuscript/nutriopt_bench.tex` → `manuscript/nutriopt_bench.pdf` (≈15 pp.)

---

## 1. Title / positioning

**New title:**
> Range-Constrained Binary Subset Selection:  
> A Fair-Budget Benchmark for Sparse Feasible-Region Search

- Food/nutrition removed from title; RCFA retained as instantiation in abstract/body.
- Central framing: sparse feasible-region search under coupled interval constraints (not MOEA survey / diet paper).

## 2. Section-level changes

| Section | Change |
|---------|--------|
| Abstract | Rebuilt: gap → RCFA → NutriOpt-Bench → axes → fair-budget → FHR → findings → synthetic → budget → methodological implication |
| Intro | 8-paragraph logic + **C1–C6** hierarchy |
| Related Work | A constrained EA / B combinatorial subset / C food as application + cautious gap; `tab:position` kept |
| Formulation | General problem vs RCFA vs NutriOpt-Bench; controlled axes (not orthogonal); FHR ≠ nutrition quality |
| Protocol | Computational contract table retained; mechanism families |
| Results | Reordered by questions: E1 scale → E2 tightness+density → E3 dim → E4 ranks → **E5 synthetic** → **budget** → calibration |
| Discussion | Q1–Q5 + external validity |
| Guidelines | Compressed |
| Limitations | Strengthened 7-point list |
| Conclusion | 7 findings + methodological message; **no “MA is best”** |

## 3. Experiment / result integration (verified against CSVs)

| Source | Key numbers used |
|--------|------------------|
| E1 | NHANES band ≈0.76–0.82; recipes FHR=1.0 |
| E2 | 0.172 / 0.761 / 0.962 |
| E3 | 0.992 / 0.983 / 0.946 / 0.795 |
| E4 | τ = 0.909–0.982 |
| E5 | f: 0.667/0.794/0.917; m: 0.930 vs 0.656; N: 0.806/0.783/0.789 |
| Budget | means 0.139/0.165/0.193/0.208; per-algo table; 100k from PRIMARY E2 (disclosed) |
| Density | MC diagnostic; censored \(\hat p_F\); mean_V tracks FHR |
| Exact | pool 500: 8/4; multi-pool 250/500/1000: 9/3, 8/4, 10/2 |

## 4. Budget-sensitivity integration

- Primary protocol remains **100k**.
- Budget panel = protocol validation (not algorithm ranking).
- Message: measurable gains 25k→100k; modest 100k→200k (+0.015 aggregate); Random near floor; 100k defensible.

## 5. Error handling (`low >= high`)

- **Root cause:** `np.random.randint(min_foods, min(max_foods, len(combined))+1)` when unique crossover pool `< min_foods`.
- **Fix:** `_safe_card_randint` in `baseline_algorithms.py` (all cardinality sample sites).
- **Re-run:** BMOEA@50k/rep_001/r8 and MA@200k/rep_008/r5 completed with **FHR=0** (ordinary search failure).
- **Aggregates unchanged** to reported precision; status now `ok`; manuscript discloses the incident.

## 6. Claims removed / weakened

- “Orthogonal axes” → controlled / one-factor-at-a-time
- Universal champion / “MA is best”
- Phase transition / scaling law / N*
- Causal “causes” without qualification
- Clinical / diet-recommendation language
- Budget as ranking section

## 7. Figures / tables

| Asset | Role |
|-------|------|
| fig_e1… | Scale → FHR |
| fig_e2… | Interval tightness → FHR |
| fig_density_vs_fhr | Empirical sparsity diagnostic |
| fig_e3… | Dimensionality |
| fig_e4 / heatmap | Ranking concordance |
| **fig_e5_synthetic_fhr** | Structural external validation |
| **fig_budget_vs_fhr** | Budget adequacy (25k–200k) |
| tab:position, tab:contract, tab:axes, tab:params, tab:ranks | Retained |

## 8. Remaining TEVC review risks

| Risk | Mitigation / residual |
|------|----------------------|
| “Is this food optimization?” | Title/abstract/intro now EC-first; food = instantiation |
| Generality of findings | E5 qualitative only; absolute FHR differs; stated explicitly |
| E3 nutrient-order dependence | Listed in Limitations |
| Residual pymoo/native asymmetries | Disclosed in contract/limitations |
| Empirical density heavily censored at tight f | Framed as diagnostic + violation proxy, not exact volume |
| 100k vs 200k seed-stream mix | Disclosed (E2 reuse for 100k) |
| No 30-seed robustness panel | Optional; not blocking if reviewers accept 10-seed primary |
| Page length (~15) | May need further compression of guidelines/discussion for some TEVC length norms |

## 9. Reproducibility preserved

- pop=500, primary evals=100000, early-stop off, 10 hashed seeds, shared predicate
- PRIMARY E1–E4 CSVs untouched
- Budget/E5 are additive arms
