# legacy — original (upstream) Perl pipeline

The original Sicilian-Translator data pipeline, taken from upstream
[`ewdowiak/Sicilian_Translator`](https://github.com/ewdowiak/Sicilian_Translator)
(latest version), kept here for reference and for comparing extraction methods.

- `extraction/` — PDF → text → align: `01_extract-text.pl` (pdflatex on hand-picked
  page ranges), `02_wrap_text.pl`, `03_align.sh` (hunalign + Dieli dictionary),
  `04_sort-parallel-text.pl` (quality filter).
- `perl-module/Napizia/` — tokenizer/detokenizer + helpers used by the pipeline.
- `dataset/` — tokenization, ASCII-ification, subword and m2m combination scripts.

Not fully runnable from this repo alone: it needs `hunalign`, the Dieli dictionary,
the hand-picked per-issue page ranges (in `01_extract-text.pl`), and external assembly
files. The active pipeline is the Python one under `experiments/` (PyMuPDF + LaBSE),
which already extracts from the same PDFs. See the extraction comparison in the commit
history / docs.
