# Sicilian/English tokenization (pure Python)

`sicilian_tok.py` is a faithful Python port of the Napizia Perl tokenizer
(`perl-module/Napizia/Translator.pm`: `rm_malice`, `sc_tokenizer`, `en_tokenizer`
and helpers) plus the `rm_morejunk` step. It removes the Perl dependency from the
data pipeline (the paper's recipe: ASCII-fold accents + expand Sicilian
contractions/conjunctive pronouns — ô→a lu, dû→di lu, mû→mi lu, ...).

```bash
python experiments/tokenization/sicilian_tok.py sc < in.scn > out.scn
python experiments/tokenization/sicilian_tok.py en < in.en  > out.en
```

## Fidelity

The port was verified byte-for-byte against the original Perl (commit that added
this module): core logic **identical** (0 differences on the test set). The Python
additionally normalizes em/en-dashes and ellipses, which the Perl's byte-mode
`rm_morejunk` silently skips (`\x{2026}` cannot match UTF-8 bytes) — a latent Perl
bug the port fixes. Note: Perl `lc()` here is byte-mode (ASCII-only); the port
replicates that with `_ascii_lower`, then handles accents via explicit maps.

This is now the canonical tokenizer (`experiments/baseline/lever_b_prep.sh` uses it);
the Perl module has been removed from the repo.

**TODO:** port the inverse (detokenizer) to Python for serving. The original Perl
`en_detokenizer` was buggy (dropped first/last words), so evaluation is done in
tokenized space anyway.
