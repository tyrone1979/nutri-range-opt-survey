# Round-2 Revision Audit
# Review identity strengthening + benchmark-language suppression + literature credibility

Date: 2026-09-09  
Manuscript: `manuscript/nutriopt_bench.tex` (PDF ~14 pp after rebuild)

## A. Changed terminology

| From | To / action |
|------|-------------|
| fair-budget / fairness audit / fairness residuals / fairness contract | removed as identity language; → implementation audit / residual evaluation-condition records / common evaluation conditions |
| computational contract / PRIMARY contract | common evaluation conditions / common experimental conditions |
| hardness grammar | controlled variation of problem structure |
| Campaigns E1–E5 | Empirical analyses E1–E5 |
| hardness proxy / hardness factor / hardness sweep | difficulty / structural wording |
| ranking stability (as identity) | cross-regime behavioral consistency / concordance |
| NutriOpt-Bench (repeated) | **one** reproducibility alias in Intro; thereafter “unified empirical setting” |
| supports the adequacy of 100k | supports use of 100k as a *practical* primary ceiling in the present setting |
| `\nocite{*}` | **removed** (was dumping ~150 mostly diet/peripheral refs) |

“Leaderboard” retained only as an explicit *negation* (“not a universal/standalone leaderboard”).

## B. Changed section titles

| Old | New |
|-----|-----|
| III. Problem Formulation | III. Problem Setting and Empirical Design |
| IV. Unified Empirical Analysis Framework | IV. Empirical Comparison Protocol |
| V. Empirical Comparison of Algorithmic Search Behavior | V. Empirical Evidence on Algorithmic Search Behavior |
| Controlled difficulty axes | Structural factors examined… |
| Why classical…hardness geometry | Relationship to classical constrained suites |
| Common evaluation ceiling… | Common experimental conditions |
| Algorithm suite | Representative algorithm suite |
| Campaign overview | Overview of empirical analyses E1–E5 |
| Feasible Hit Rate | Feasible Hit Rate and feasibility-search diagnostics |
| VI Q1–Q5 labels | Synthesis questions (no “Is 100k enough?” as Q5) |

Table captions: Table I structural families; Table II common experimental conditions; Table III structural factors; Table V descriptive ranks (not winner table).

## C. Literature-review improvements

- Added **Scope and organization of the review** (problem-driven; not PRISMA; not exhaustive).
- Each mechanism subsection now follows: what studied → structures/mechanisms → established conclusions → comparatively less systematic → sparse interval-box outlook.
- Expanded EC citations from existing `references.bib` (constraint handling, knapsack, sparse large-scale, repair, memetic, two-archive, etc.).
- Diet/menu refs confined mainly to application/hybrid context.
- Removed all `[NEEDS LITERATURE SUPPORT]` markers by citing existing bib keys or softening claims.
- Bibliography now **42 cited items** (was ~150 via `\nocite{*}`).

## D. Claims that were weakened

- 100k “adequacy” → practical primary ceiling under present setting (not universal sufficiency).
- E4 “ranking” framing → behavioral consistency / concordance.
- Classical suites “under-represent hardness grammar” → complements suites; not a universal suite.
- Algorithm suite → mechanism sampling, not SOTA exhaustiveness.
- Conclusion: no benchmark/fairness/contract identity language.

## E. Claims that still require literature support (soft risk; no fabricated cites)

- Classical stochastic ranking / ε-constrained *lineages* are not cited by name (bib lacked those exact classics); coverage uses hybrid CH / feasible-archive / repair papers instead.
- “Comparatively less systematic” statements on two-sided interval boxes remain qualitative judgments of the reviewed seed set, not a proven empty-set claim.
- Deeper TEVC review depth may still benefit from adding verified classics (Runarsson & Yao; Takahama; Coello constraint surveys) in a *future* literature pass—without inventing them here.

## F. Experimental content left unchanged

All PRIMARY numbers preserved: E1 band ~0.76–0.82; E2 0.172/0.761/0.962; E3 0.992/0.983/0.946/0.795; τ 0.909–0.982; E5 0.667/0.794/0.917; budget means; exact pools 9/3, 8/4, 10/2; seeds; 100k/200k; algorithms; datasets.

## G. Remaining potential reviewer risks

1. Name **NutriOpt-Bench** still appears once (Intro alias)—some reviewers may still hear “benchmark.”
2. Table V still shows mean ranks (necessary for concordance); mitigated by caption/body disclaimers.
3. Review is **problem-driven / structured**, not PRISMA-exhaustive—correctly disclosed, but may invite “missing classic CH survey” comments.
4. Application refs remain in bib where cited for RCFA context; keep them out of mechanism-core claims.
5. PDF shortened (~14 pp) mainly via bibliography pruning; verify journal length expectations.

## 30-second cognitive test (target answers)

1. Problem: sparse interval-constrained binary subset selection  
2. Type: problem-driven systematic review + unified empirical analysis  
3. Reviewed: evolutionary search mechanisms for this class  
4. Why experiments: empirically synthesize mechanism behavior under controlled structural difficulty  
5. Main finding: interval-induced sparsity strongest observed difficulty factor vs scale alone  
6. Also: dimensionality selective; relative behavior concordant across regimes  
7. Benchmark paper?: No
