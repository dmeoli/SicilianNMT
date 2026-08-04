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

## Web scrapers (Napizia sources — collaboration with E. Wdowiak, 2026-08)

Additional parallel text beyond the Arba Sicula PDFs. Stdlib-only (no ML deps), so
they run anywhere; output feeds the same LaBSE sentence-alignment step.

### `scrape_magazine.py`

```
python scrape_magazine.py --out data/processed/napizia_magazine
```
Crawls `magazine.napizia.com`, downloads each article's Sicilian / English / Italian
pages and pairs body paragraphs positionally (the site keeps versions aligned 1:1;
count mismatches are written out and flagged for LaBSE). Writes per-article
`<vol>-<slug>.{scn,en,it}`, combined `magazine.{scn,en}`, and `manifest.json`.

**Copyright filter:** Maria Anna Manzella's poetry is excluded (agreed with Eryk) —
hard-blocked by slug (`mulinazzu`) *and* by author line (any `<h2>` containing
"Manzella"). Do not remove either guard.

### TODO (need a live inspection pass first)

- **Napizia Dictionary** (`dizziunariu.napizia.com`) — example sentences from poetry /
  proverbs / prose. Search-based, no word list; the raw Dieli vocab we already have in
  `vocab/`. Lower priority.
- **Young Sicilian Manifesto** — locate the page and its structure, then scrape.
