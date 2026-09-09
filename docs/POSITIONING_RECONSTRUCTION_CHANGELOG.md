# Positioning Reconstruction CHANGELOG
# Benchmark methodology → Systematic review + empirical analysis

Date: 2026-09-09  
Manuscript: `manuscript/nutriopt_bench.tex` (PDF ~17 pp.)

## Mapping (old → new)

| Old | New |
|-----|-----|
| Fair-Budget Benchmark title | **Evolutionary Algorithms for Sparse Interval-Constrained Binary Subset Selection: A Review and Empirical Analysis** |
| Related Work | II. Evolutionary Algorithms for … (mechanism taxonomy A–I + synthesis) |
| Experimental Protocol | IV. Unified Empirical Analysis Framework |
| Experimental Results | V. Empirical Comparison of Algorithmic Search Behavior |
| Discussion + Benchmarking Guidelines | VI. Cross-Algorithm Synthesis and General Insights |
| Conclusion (benchmark) | VIII. Conclusion (review + empirical patterns + open questions) |

## P0 changes executed

- Title / Abstract / Keywords / Introduction / RQ1–RQ4 / Contributions (4 review-oriented)
- Related Work expanded to mechanism taxonomy with strengths/weaknesses
- NutriOpt-Bench redefined as **unified empirical setting** (not methodological novelty)
- RCFA kept as **application instantiation**
- “fair-budget methodology” → **common evaluation ceiling**
- No “benchmark methodology” / “fair-budget benchmark” identity language remaining (lowercase `benchmark` scrubbed)
- Discussion renamed to synthesis; guidelines merged into implications
- Conclusion is review conclusion, not “NutriOpt-Bench provides…”

## Numbers preserved (verified)

E2 0.172/0.761/0.962; E1 ~0.76–0.82; E3 0.992/0.983/0.946/0.795; τ 0.909–0.982;  
E5 0.667/0.794/0.917; m 0.930 vs 0.656; N 0.806/0.783/0.789;  
Budget means 0.139→0.165→0.193→0.208; exact pools 9/3, 8/4, 10/2.

## [NEEDS LITERATURE SUPPORT] markers (intentional; no fabricated cites)

Intro census of penalty/repair; §II-A constraint-handling taxonomies; §II-B MKP/set-cover under two-sided boxes; §II-C penalty vs repair; §II-D feasibility-rules lineages; §II-F decomposition with FHR endpoint; §II-G constrained-archive vs dominance; §II-H memetic surveys for sparse intervals; Limitations (8) points to these markers.

## Self-check (user §22)

- [x] Title is Review / Empirical Analysis  
- [x] Abstract review-oriented  
- [x] Intro literature-gap (not benchmark gap)  
- [x] RQ1–RQ4 review + synthesis  
- [x] Contributions not benchmark methodology  
- [x] Related Work taxonomy present  
- [x] Mechanism-oriented synthesis in §II  
- [x] RCFA = instantiation  
- [x] NutriOpt-Bench not primary contribution  
- [x] Fair-budget not computational-fairness identity  
- [x] E1–E5 as empirical evidence  
- [x] FHR = feasibility-discovery metric  
- [x] No universal best algorithm  
- [x] E3 canonical-prefix caveat  
- [x] E5 not universal validation  
- [x] Synthesis section is core  
- [x] Conclusion = domain patterns  
- [x] Numbers unchanged  
- [x] No fabricated citations  
- [x] Unsupported claims marked [NEEDS LITERATURE SUPPORT]

## Remaining TEVC risks

1. Review depth still limited by current bibliography seed set (markers flag gaps).  
2. ~17 pages may be long for some TEVC review formats.  
3. Name “NutriOpt-Bench” still appears (as empirical setting only)—reviewers may still hear “benchmark”; consider further reducing occurrences if needed.  
4. Filling [NEEDS LITERATURE SUPPORT] with verified references is the next literature task (not inventing).
