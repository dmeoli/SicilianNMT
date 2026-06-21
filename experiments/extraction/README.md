# Extraction experiments (Phase 3)

Modern, CPU-only replacement for the legacy PDF extraction pipeline
(`extract-text/01_extract-text.pl` → pdflatex rotate/crop → pdftotext → hunalign).

## Findings (as30 prototype)

- **PyMuPDF extracts clean text directly** from the original Arba Sicula PDFs:
  no LaTeX, no 180° rotation, no viewport crop, no custom-font garbage, accents intact.
- Layout is **facing pages: even = Sicilian, odd = English**.
- A stopword-ratio **page-language classifier** recovers all 29 hand-picked gold
  Sicilian pages *and* finds genuine parallel pairs the manual process skipped
  (e.g. pp.30-31 poem). So automation can produce a **larger** corpus than the original.
- Facing-page pairing **over-generates** (~58 candidates vs 29 gold), including false
  positives (bilingual cover, tables of contents) → must be filtered by the next stage.

## `extract_pages.py`

```
python extract_pages.py extract-text/as-issues/as30.pdf --out out/as30
```
Writes `sc.txt`, `en.txt`, `pairs.tsv` (candidate facing SC/EN page pairs).

## Next stage (not here yet)

Cross-lingual **sentence-embedding alignment** (LaBSE / SONAR — `scn_Latn` is
supported — + vecalign) to (1) confirm which candidate page pairs are real mutual
translations and (2) align at the sentence level. Benchmarked against the
`extract-text/AS27-31_aligned_set01.csv` gold set vs the legacy hunalign output.
