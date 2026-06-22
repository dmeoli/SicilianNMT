# Extraction methods compared: legacy Perl vs ours

On Arba Sicula **as30** (scn→en), comparing the original Perl/hunalign pipeline output
(`extract-text/aligned/as30_ha.csv`) with our PyMuPDF+LaBSE pipeline
(`experiments/extraction/build_issue.py`):

| method | pairs | notes |
|---|---|---|
| Perl (hunalign) | 546 (399 with score > 0.5) | more segments, more short/noisy fragments |
| ours (LaBSE) | 351 | fewer, high-confidence (sim ≥ 0.5) |

- **202** Sicilian sentences match exactly (normalized); ~236 of ours are contained in the
  Perl text. Perl-only: 344, ours-only: 144.
- Complementary, neither dominates: Perl gets more quantity (with noise); ours is cleaner
  and recovers coherent prose the manual page-selection skipped (e.g. the "history of
  Arba Sicula" passages).

## Merge (union)

`experiments/extraction/merge_perl.py` adds the Perl pairs (as27-31, score > 0.5) that our
pipeline did **not** already produce, deduped on normalized Sicilian:
**+862 Perl-unique pairs** → Arba Sicula corpus 9.3k → 10.2k.

After the unified re-assembly that is **+840 train pairs (+3.7%)** on a 22k train already
dominated by NLLB (16k). So the enrichment is **marginal** — not worth a dedicated ~6 h CPU
retrain to measure in isolation, but it is kept in `data/dataset` and benefits future
training (NLLB fine-tune, BPE-dropout, etc.).

`assemble.py` was fixed so test/valid are drawn only from the original LaBSE-extracted pairs
(`heldout` flag), so adding Perl-merged or future data never moves the held-out sets.

The Perl pipeline is preserved under `legacy/` (not fully runnable without hunalign + the
Dieli dictionary + the hand-picked page ranges).
