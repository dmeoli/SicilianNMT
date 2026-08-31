# Research ideas & data sources (mined from the project history)

Actionable ideas, data sources and techniques to push Sicilian MT further. Several come
from the long discussion with Eryk Wdowiak; the rest from our own results.

## Data sources to add (beyond what we use)

We currently use: auto-extracted Arba Sicula (44 PDFs) + a filtered slice of
`Good-Sicilian-in-NLLB` + WikiMatrix it-scn. Still untapped:

- **Eryk's curated sets (HF / his sites):** `Good-Sicilian-from-WikiMatrix` (514 it-scn,
  hand-edited), the best of `Good-Sicilian-in-NLLB` (we use a slice), and **scrapes of**
  the Napizia **Dictionary**, the Napizia **Magazine**, and the **Young Sicilian Manifesto**.
- **Textbook / trilingual homework exercises** — Cipolla's *Mparamu lu sicilianu* and
  Bonner's *Grammar*. These are **trilingual** (scn/en/it), so they also feed the Italian
  bridge. Eryk reports these "few-shot each grammatical element" and noticeably help.
- **Dieli's translations of Pitrè's *Folk Tales*** (folklore prose).
- **Marco Scalabrino's** Sicilian translations of American songs; **David Massaro's**
  Sicilian Bible (Eryk was gathering these).
- **More from the Legas bookstore** — ask Gaetano Cipolla; many Legas books are bilingual.
- **WikiMatrix en-scn** (we only extracted it-scn) and the rest of **NLLB** via
  back-translation (treat as monolingual → translate → score → keep best).

## Modelling / training techniques to try

- **Linguistic input features (source/target factors)** — Sennrich's "Linguistic Input
  Features" (aclanthology W16-2209): concatenate lemmas + POS tags onto each token. We have
  `vocab/lemma_dict_scn.json`; POS/lemmas also come from the Dieli dictionary and
  Wikidizionario. Sockeye supports `--source-factors`/`--target-factors`; for NLLB we'd
  prepend factor tags. **Untried, and it's literally one of the paper's research questions.**
- **Theoretic (desinence-biased) subword splitting** — done (lever B, +1.7 BLEU). Quantify
  its contribution as an ablation.
- **Reverse-training strategy** (Eryk's best): design the fine-tune set first, then build
  pre-training data backward (forward-translate synthetic → pre-train on internet text →
  fine-tune on curated). His scn→en hit ~48 BLEU on his test set this way.
- **Zero-shot + bridging (multilingual)**: a multilingual model gives zero-shot scn↔it;
  back-translate scn-it; translate the trilingual homework into en/it → a trilingual
  "bridge". We have 11k it-scn (WikiMatrix) to seed this.
- **Back-translation for pre-training** (Eryk offered to help): identify good Sicilian in
  the rest of NLLB and back-translate it.
- **Newer Sockeye / SSRU decoder** (arXiv 2207.05851 §4.2) — better speed/accuracy than the
  2017 Sockeye; relevant if we keep a Sockeye baseline.
- **Grid search + k-fold CV** (Ray on Colab) — the model is small; systematic
  hyperparameter search is feasible and was on the agreed agenda.
- **Bigger NLLB** (1.3B / 3.3B) — in progress.

## Representation-analysis studies (probing)

- **Bidirectional vs unidirectional training and encoder clustering** — *Eryk's idea
  (2026-08), offered to us to "steal", credit him.* Question: does bidirectional training
  pull translated sentence pairs closer together in the encoder than unidirectional training?
  His proposed setup: take an ~8M It–En corpus and cut it in half; the **bidirectional** model
  trains on the *same* 4M sentences on both the source and target sides (discarding the other
  4M), so the encoder sees each sentence in both languages; the **unidirectional** model trains
  one 4M half for it→en and the other 4M half for en→it, so the encoder sees each sentence only
  once. Hypothesis: the bidirectional encoder places translated pairs closer. Also **vary depth**
  (3 vs 6 layers) to see whether pairs converge as depth grows. Measure with SVCCA / cosine, cf.
  Kudugunta et al. 2019 (encoder reps cluster by linguistic similarity and depend on both source
  and target) and Verma et al. 2026 (surface form over structure). **Cheap first
  cut for us:** our trilingual NLLB adapter is *already* bidirectionally fine-tuned, so we can
  probe its encoder geometry (scn/en/it translated-pair distances) and compare against a
  unidirectional-only fine-tune — half the study is set up already.

## Extraction improvements (our pipeline)

- **Caption removal** — captions getting mixed into the body text was Eryk's biggest manual
  pain. Add layout-aware caption/figure detection (PyMuPDF blocks + font/position) so we
  don't need hand-cleaning.
- **Old issues (1–18)** extract poorly (evolved orthography + captions; early issues were
  **phototypeset**). Try **modern OCR** (Surya/Docling, or Google Lens) and a **regex
  normalizer** to map non-standard → Standard Sicilian, then measure the BLEU effect
  with/without normalization.
- **Issue DATE is a quality/standardness signal** (Eryk, 2026-08). Arba Sicula's Sicilian
  improves *steadily over time* as written norms matured: Pitrè's 1875 grammar (descriptive of
  diatopic variation, not a style guide) → Privitera 1998 → **Kirk Bonner's *Introduction to
  Sicilian Grammar* (2001)**, which supplied the missing rules and became a de-facto style guide
  → 2000s Google Books digitization of 19th-c dictionaries → Cipolla's textbooks (2013, 2021).
  So **weight/prioritise post-2001 issues** (cleaner, more standard), and reserve the OCR +
  aggressive-normalization path for the older, lower-quality issues. Worth an ablation: does
  restricting training data to newer issues, or weighting by issue year, raise BLEU?
  - **Refinement (Eryk, 2026-09):** go finer than year and **weight Prof. Cipolla's texts**
    more heavily. His stylistic variation (even between *Learn Sicilian* 1 (2013) and 2 (2021))
    is narrower than the *combined* variation of all Arba Sicula contributors, so a
    Cipolla-heavy training mix should be more internally consistent. **Open problem:**
    identifying which AS articles Cipolla wrote or translated — there is no clean byline
    metadata, so this needs a detector (TOC "di/by" parsing where present, plus a stylometric
    or authorship classifier for the rest). Harder than it looks; a good sub-project.
- **Don't discard unaligned text** — Eryk's trick: text that can't be sentence-aligned still
  goes into a monolingual pool for back-translation.
- We avoid poetry too (hard to align); keep that, but the glossary/nomenclature pages in old
  issues are bilingual term lists → a `term:gloss` parser would harvest dictionary-style data.

## The paper (framing)

The agreed research question: **what makes the most efficient use of low-resource data?**
A clean ablation — pretrained model (NLLB) vs from-scratch, ± theoretic subwords,
± homework exercises, ± linguistic features, ± back-translation — on one held-out literary
test set, with a fully public, automatically-built dataset. That's the contribution.

## People / leads

- **Gaetano Cipolla** (Arba Sicula / Legas) — gave data permission; can point to more
  bilingual Legas materials. Acknowledge Arba Sicula + Legas in any output.
- **Donato's NLP advisor (Pisa)** — happy to advise.
- Note from Eryk: a Google contact once wanted to put Sicilian in Google Translate and
  fine-tune a multilingual model, but never followed up — Sicilian still isn't in Google
  Translate. (Gap worth filling.)
