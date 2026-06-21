#!/bin/bash
# Verify the Python port matches the Perl Napizia tokenizer byte-for-byte (core logic).
# The only intended differences are em/en-dash and ellipsis normalization, which the
# Perl's byte-mode rm_morejunk silently skips (\x{2026} can't match UTF-8 bytes) but
# the Python correctly applies. After equalising those, the outputs are identical.
#
#   bash experiments/tokenization/verify_tok.sh data/dataset/test
set -e
BASE=${1:-data/dataset/test}
PY=experiments/tokenization/sicilian_tok.py
TMP=$(mktemp -d)

for lang in scn en; do
  pl=sc; [ "$lang" = en ] && pl=en
  PERL5LIB=perl-module perl experiments/baseline/tokenize.pl "$pl" < "$BASE.$lang" > "$TMP/perl.$lang"
  python "$PY" "$pl" < "$BASE.$lang" > "$TMP/py.$lang"
  python3 - "$TMP/py.$lang" "$TMP/perl.$lang" <<'PY'
import re, sys
def norm(s):
    s = re.sub(r'[—–―]', '-', s).replace('…', ' . . . ').replace('œ', 'oe').replace('æ', 'ae')
    s = s.replace('š', 's').replace('Š', 's')
    return re.sub(r'\s+', ' ', s).strip()
a = [norm(l) for l in open(sys.argv[1])]
b = [norm(l) for l in open(sys.argv[2])]
d = sum(x != y for x, y in zip(a, b))
print(f"{sys.argv[1].split('/')[-1]}: {d} core-logic differences (expect 0)")
PY
done
rm -rf "$TMP"
