# docs

- **`paper/sicilian-nmt.tex`**, the paper (arXiv:2610.30374), with its tables in `paper/tables/`
  and the bibliography in `paper/references.bib`.
- **`slides/sicilian-nmt-hlt.tex`**, the Human Language Technologies talk (lamalunga theme,
  LuaLaTeX); `slides/sicilian-nmt-hlt-it.tex` builds the same deck in Italian from the same
  source, and `slides/make_charts.py` draws the charts in both languages.
- `slides/sicilian-nmt-slides.tex`, an older and shorter Beamer deck.
- `low-resource-nmt-methods.md`, survey and ranked experiment plan.
- `sicilian-standardization.md`, `sicilian-contractions.md`, Standard-Sicilian references.

Build:

```bash
cd docs/paper && pdflatex sicilian-nmt.tex && bibtex sicilian-nmt && pdflatex sicilian-nmt.tex && pdflatex sicilian-nmt.tex
cd docs/slides && python3 make_charts.py && latexmk sicilian-nmt-hlt.tex && latexmk sicilian-nmt-hlt-it.tex
```

The slides need a checkout of the lamalunga theme in `~/GDrive/lamalunga` (see `slides/latexmkrc`).
