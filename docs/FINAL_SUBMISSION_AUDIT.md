# Final Pre-Submission Revision Audit

Date: 2026-09-09  
Manuscript: `manuscript/nutriopt_bench.tex` (PDF ~14 pp.)

## Identity (stable)

**Problem-driven structured review** of evolutionary search for sparse interval-constrained binary subset selection, **supported by a unified empirical analysis**.

Not: benchmark methodology / new algorithm / leaderboard / SOTA paper.

## A. Terminology

- `systematic review` → `structured review` / `problem-driven structured review` (except “less systematic / systematically categorized”)
- NutriOpt-Bench: **once** (Intro reproducibility alias)
- No fairness / fair-budget / computational contract / hardness grammar / benchmark methodology
- “leaderboard” only as negation
- Algorithm language: relatively strong / comparatively weaker (not winner/best)

## B. Section titles (unchanged from Round 2, verified)

I Intro → II Literature (scope + organizing lens) → III Problem Setting → IV Empirical Comparison Protocol → V Empirical Evidence → VI Synthesis → VII Limitations → VIII Conclusion

## C. Literature

- **Organizing lens** sentence added in §II.A
- Unknown ref fixed: `anon2025…` → `cork2025landscapeawarese8d63`  
  Authors (from IS PDF): Cork, Andova, Krömer, Tušar, Filipič; IS 2025; DOI 10.70314/is.2025.skui.3280
- Ref [27]-class DOI `10.1162/evco_a_00320` (Aishwaryaprajna & Rowe) retained with underscore form (correct MIT Press DOI); no space form
- No `[NEEDS LITERATURE SUPPORT]`
- No fabricated citations

## D. Statistics (code-verified)

PRIMARY Friedman (`analyze_e2.py` / analogous PRIMARY pipelines):

1. Average FHR over seeds (reps) within each target  
2. Build targets × algorithms matrix  
3. Friedman treats **targets** as matched blocks  

Manuscript wording updated to match.  
E3 Kendall τ still restricted to shared algorithm subset.

## E. Figures

- **Deleted** reconstructed best-CV trajectory figure (`fig:conv` / `convergence_sparsity.pdf`) and all related text  
- Replaced with: terminal CV + FFG complementary evidence  
- Remaining figures continuous: concept, e1, e2, density, e3, box16, e4ranks, heat, e5, budget  
- Figure 1 regenerated: title **Problem difficulty versus relative algorithmic behavior**; NutriOpt-Bench removed from visual

## F. Tables

- Table I: structural families; last row = problem class in this review  
- Table II: Common conditions for empirical comparison  
- Table III: Structural factors examined  
- Table V: Cross-regime behavioral summary… (no “1 = better” in caption)

## G. Claim tightening

- FHR = empirical success rate (not theoretical probability)  
- MC \(\hat p_F=0\) = none observed in finite sample  
- Orthotope volume ≠ binary feasible-set density  
- E3 under canonical-prefix construction  
- E4/VI.D: **Yes, relative behavior largely stable** (contradiction fixed)  
- 100k = practical ceiling, not sufficiency  
- E5 = structural replication, not universality  

## H. Experiments unchanged

All PRIMARY numbers preserved (E1–E5, τ, budget, calibration, seeds, 100k).

## I. Consistency checklist (§46)

- [x] Title Review + Empirical Analysis  
- [x] Abstract structured review + empirical analysis  
- [x] Intro problem-driven; experiments test/refine review patterns  
- [x] Contributions ≠ benchmark methodology  
- [x] §II literature-derived + organizing lens  
- [x] No Unknown; DOI checked  
- [x] No NEEDS markers  
- [x] Fig5 deleted; numbering OK; Fig1 de-branded  
- [x] Friedman unit matches code  
- [x] No universal best / SOTA / 100k-sufficient / E5-universality claims  

## Remaining minor risks

1. Name NutriOpt-Bench still appears once (intentional).  
2. Table V still shows numeric ranks (needed for concordance; disclaimers present).  
3. Review is structured/problem-driven, not PRISMA—correctly disclosed.
