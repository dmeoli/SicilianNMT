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

## `analyze_standard_vs_nonstandard.py`

Tests Eryk's hypothesis (2026-08) that our extractor should agree with his
hand-alignments more on **standard** Sicilian (esp. Cipolla) than on dialect poetry
/ folk tales. Over his AS41-42 gold (18 texts, 1,373 pairs) it relates two objective
non-standardness proxies (OOV vs the Dieli standard lexicon; orthographic divergence
under `normalize_scn` std) to the alignment-difficulty signals already in his gold
(`se_score` = his Sockeye model score; `score` = hunalign confidence). Pure stdlib.

```
python experiments/extraction/analyze_standard_vs_nonstandard.py   # needs gold in data/external/eryk/
```

**First-pass finding (2026-08):** the hypothesis is **not** strongly supported. Vocabulary
OOV is essentially uncorrelated with difficulty (Spearman +0.07 vs se_score), and
sentence length is not a confound (~0). Orthographic divergence shows only a modest
negative correlation (Spearman −0.52) over a tiny dynamic range, and the two difficulty
signals disagree in sign — i.e. weak effects. Per-genre means are anecdotal (n=1 for
Cipolla and for Pitrè) and, if anything, run opposite to the intuition. **Caveat:**
se_score/ha_score are his model's signals, not our LaBSE extractor's recall; the
definitive per-text LaBSE-recall test is the follow-up (needs the ML venv rebuilt).

## `labse_by_text.py` (the DIRECT test — supersedes the first pass)

The first pass used HIS model's signals; this uses OURS. LaBSE is the encoder our
aligner scores with, so the LaBSE cosine of an already-correct (hand-aligned) pair
measures how confidently our extractor could recover it. Reports per text the mean
LaBSE cosine of the gold pairs and the recall at our production threshold (0.40).

```
python experiments/extraction/labse_by_text.py     # needs the ML .venv + gold in data/external/eryk/
```

**Finding (2026-08): with the RIGHT signal, Eryk's hypothesis largely HOLDS.** Genre
means (LaBSE cos / recall@0.40): **standard-prose (Cipolla) 0.759 / 0.91 = easiest**,
prose 0.666 / 0.85, poetry 0.635 / 0.80, folk-tale 0.607 / 0.88. Standard Sicilian
aligns best and dialect poetry worst — the opposite of the his-model first pass, because
se_score measures translatability under his small model (Cipolla's ornate literary prose
is hard to *model*) whereas LaBSE measures alignability (clean standard orthography maps
cleanly to English). Nuance: vocabulary OOV is NOT a smooth predictor (Spearman −0.11);
the losses concentrate in a few strongly dialectal texts (e.g. "Giuseppe Sciacca" LaBSE
0.359, recall 0.38) and in poetry. **Corrective measures suggested:** genre-aware / lower
thresholds for poetry, a Sicilian-adapted encoder, and flagging low-mean-LaBSE texts for
review. n=1 for the standard-prose and folk-tale genres (single texts) — treat those
means as anecdotal; the standard > prose > poetry gradient is the robust signal.

## `labse_why_sicilian.py` — why does vanilla LaBSE work on Sicilian at all?

Eryk's question (LaBSE has no official Sicilian support). On FLORES-200 (multi-parallel,
includes scn_Latn): **scn→en retrieval P@1 = 99%**, trailing SUPPORTED Italian (100%) by
1% — it genuinely works. Mechanism: raw cosines don't give a clean distance ladder
(English is LaBSE's pivot, so scn-en cosine is highest), but ignoring English, each
Sicilian sentence's nearest true translation is **Italian 18% vs French 8%** (2× the next
Romance) — LaBSE places Sicilian in its Italian/Romance region and the transfer rides on
that lexical/Romance proximity.

```
python experiments/extraction/labse_why_sicilian.py    # needs ML .venv + data/external/flores
```

## `poetry_rolling.py` — test Eryk's "roll verses into one line"

Simulates rolling k consecutive hand-aligned pairs into one unit before scoring. **Eryk was
right, especially for poetry.** Mean LaBSE cos / recall@0.40, k=1 → k=3:

```
  poetry          0.651/0.84 -> 0.722/0.96 -> 0.764/0.99   (+0.11 cos, +0.15 recall)
  prose           0.707/0.90 -> 0.761/0.94 -> 0.793/0.95   (+0.05 recall)
  standard-prose  0.759/0.91 -> 0.799/0.96 -> 0.830/1.00
  folk-tale       0.607/0.88 -> 0.674/0.95 -> 0.714/1.00
```
Rolling helps every genre but **disproportionately poetry** (+0.15 recall vs +0.05 for
prose), closing poetry's gap to prose by k=3. Caveat: recall here = clearance of the 0.40
threshold; part of the gain is mechanical (fewer, longer, more distinctive units). Precision
is plausibly up too (longer segments match less spuriously) but not directly measured here.

## TODO (need a live inspection pass first)

- **Napizia Dictionary** (`dizziunariu.napizia.com`) — example sentences from poetry /
  proverbs / prose. Search-based, no word list; the raw Dieli vocab we already have in
  `vocab/`. Lower priority.
- **Young Sicilian Manifesto** — locate the page and its structure, then scrape.
