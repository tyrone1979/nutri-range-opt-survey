# Build (IEEE TEVC / IEEEtran)

Template sources: `templates/` (official IEEEtran journal sample).

```bash
cd manuscript
# IEEEtran.cls and IEEEtran.bst must be on TeX path (copied here)
pdflatex nutriopt_bench.tex
bibtex nutriopt_bench
pdflatex nutriopt_bench.tex
pdflatex nutriopt_bench.tex
```

Class: `\documentclass[lettersize,journal]{IEEEtran}`  
Bib style: `IEEEtran`  
Keywords: `IEEEkeywords` → Index Terms

## Supplemental Material

Regenerate tables/figures from PRIMARY CSVs, then compile:

```bash
cd /path/to/nutri-range-opt-survey
python3 scripts/build_supplement.py
cd manuscript
pdflatex supplement.tex
pdflatex supplement.tex
```

Outputs: `manuscript/supplement.pdf`, `manuscript/supplement_tables/`, `manuscript/supplement_figs/`.
The supplement is an analysis appendix (no code listings); code/raw CSVs remain on GitHub.
