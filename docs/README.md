# docs

- **`paper/sicilian-nmt.tex`** — method write-up (article). Skeleton with our results so far
  and `\TODO{}`/`\num{}` placeholders to fill once the 22k, BPE-dropout and NLLB runs finish.
- **`slides/sicilian-nmt-slides.tex`** — Beamer presentation (same structure, shorter).
- `paper/references.bib` — shared bibliography (Wdowiak's + modern methods).
- `low-resource-nmt-methods.md` — survey and ranked experiment plan.
- `sicilian-standardization.md`, `sicilian-contractions.md` — Standard-Sicilian references.

Build:

```bash
cd docs/paper   && pdflatex sicilian-nmt.tex && bibtex sicilian-nmt && pdflatex sicilian-nmt.tex && pdflatex sicilian-nmt.tex
cd docs/slides  && pdflatex sicilian-nmt-slides.tex && bibtex sicilian-nmt-slides && pdflatex sicilian-nmt-slides.tex
```

Both compile today; the results tables are filled incrementally as experiments complete.
