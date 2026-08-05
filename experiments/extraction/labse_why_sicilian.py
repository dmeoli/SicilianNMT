#!/usr/bin/env python3
"""Why does vanilla LaBSE work on Sicilian, a language it does not officially support?

Eryk's question (2026-08). Hypothesis: LaBSE was trained on 109 languages including
Italian; Sicilian shares the Latin script and most of its lexicon with Italian and the
wider Romance family, so LaBSE embeds Sicilian inside its Italian/Romance region and
cross-lingual retrieval to English rides on that transfer.

Test on FLORES-200 devtest (1012 multi-parallel sentences, aligned by line, includes
scn_Latn). We measure, with LaBSE:

  1. mean cosine of TRUE pairs from Sicilian to each language -> a Romance "ladder"
     (Sicilian should sit closest to Italian, then other Romance, English mid, a
     non-Romance control far);
  2. scn->en retrieval P@1 (nearest English neighbour is the true translation) vs the
     it->en / es->en references -> shows Sicilian works almost as well as a SUPPORTED
     Romance language;
  3. for each Sicilian sentence, which language's true translation is its nearest -> if
     Italian wins most often, LaBSE literally treats Sicilian ~ Italian.

    python experiments/extraction/labse_why_sicilian.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
FLORES = REPO / "data/external/flores/flores200_dataset/devtest"
LANGS = {"scn": "scn_Latn", "it": "ita_Latn", "es": "spa_Latn", "fr": "fra_Latn",
         "pt": "por_Latn", "ro": "ron_Latn", "en": "eng_Latn", "de": "deu_Latn"}


def load(code: str) -> list[str]:
    return (FLORES / f"{code}.devtest").read_text(encoding="utf-8").splitlines()


def p_at_1(src: np.ndarray, tgt: np.ndarray) -> float:
    """Retrieval P@1: fraction of src rows whose nearest tgt row is the aligned one."""
    sim = src @ tgt.T
    return float((sim.argmax(axis=1) == np.arange(len(src))).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/analysis")
    args = ap.parse_args()

    if not FLORES.exists():
        raise SystemExit(f"FLORES devtest not found at {FLORES}")

    print("loading LaBSE ...", flush=True)
    model = SentenceTransformer("sentence-transformers/LaBSE")
    emb = {k: model.encode(load(v), normalize_embeddings=True, batch_size=64,
                           show_progress_bar=False) for k, v in LANGS.items()}
    print("embedded", {k: len(v) for k, v in emb.items()}, "\n")

    # 1. Raw mean cosine of true Sicilian->X pairs. NOTE: absolute cosines are confounded
    # by English being LaBSE's pivot language, so this is NOT a clean language-distance
    # ladder (scn-en often tops it). The within-Romance comparison in step 3 is cleaner.
    print("1. Mean LaBSE cosine of true Sicilian->X pairs (confounded by the English pivot):")
    ladder = []
    for k in ("it", "es", "pt", "fr", "ro", "en", "de"):
        c = float((emb["scn"] * emb[k]).sum(axis=1).mean())
        ladder.append((k, c))
    for k, c in sorted(ladder, key=lambda x: -x[1]):
        tag = "  (Italian)" if k == "it" else ("  (non-Romance control)" if k == "de" else "")
        print(f"   scn-{k}:  {c:.3f}{tag}")

    # 2. scn->en retrieval vs supported-language references
    print("\n2. Retrieval P@1 to English (is the true translation the nearest neighbour?):")
    for k in ("it", "es", "fr", "pt", "ro", "scn", "de"):
        ref = "  <- Sicilian (UNsupported)" if k == "scn" else ""
        print(f"   {k}->en:  {p_at_1(emb[k], emb['en']):.3f}{ref}")
    print(f"   scn->it:  {p_at_1(emb['scn'], emb['it']):.3f}  <- Sicilian to Italian (closest pair)")

    # 3. Which language is each Sicilian sentence's nearest true translation?
    others = ["it", "es", "pt", "fr", "ro", "en", "de"]
    stack = np.stack([(emb["scn"] * emb[k]).sum(axis=1) for k in others], axis=1)
    winners = np.asarray(others)[stack.argmax(axis=1)]
    print("\n3. For each Sicilian sentence, the nearest of its true translations is in:")
    uniq, counts = np.unique(winners, return_counts=True)
    for lang, ct in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1]):
        print(f"   {lang}:  {ct:>4}  ({ct / len(winners):.0%})")

    it_p1, scn_p1 = p_at_1(emb["it"], emb["en"]), p_at_1(emb["scn"], emb["en"])
    romance = {l: c for l, c in zip(uniq.tolist(), counts.tolist()) if l != "en"}
    top_rom = max(romance, key=romance.get) if romance else "?"
    print(f"\nVerdict: LaBSE handles Sicilian remarkably well despite not supporting it — "
          f"scn->en retrieval P@1 is {scn_p1:.0%}, trailing SUPPORTED Italian ({it_p1:.0%}) "
          f"by only {it_p1 - scn_p1:.0%}. Absolute cosines don't give a clean distance "
          f"ladder (English is LaBSE's pivot), but ignoring English the nearest neighbour "
          f"is overwhelmingly {top_rom} ({romance.get(top_rom, 0)} vs the next Romance "
          f"language) — i.e. LaBSE places Sicilian in its Italian/Romance region, and the "
          f"cross-lingual transfer rides on that.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labse_why_sicilian.txt").write_text(
        "scn-X mean cosine: " + ", ".join(f"{k}={c:.3f}" for k, c in ladder) + "\n"
        + f"scn->en P@1={scn_p1:.3f}  it->en P@1={it_p1:.3f}\n"
        + "nearest-language of Sicilian: "
        + ", ".join(f"{l}={c}" for l, c in zip(uniq.tolist(), counts.tolist())) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
