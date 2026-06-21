#!/usr/bin/env python3
"""Pure-Python port of the Napizia Perl Sicilian/English tokenizer.

Faithful reimplementation of perl-module/Napizia/Translator.pm (rm_malice,
sc_tokenizer, en_tokenizer + helpers) plus the rm_morejunk step from
experiments/baseline/tokenize.pl. Lets us drop the Perl dependency.

IMPORTANT: the Perl runs on bytes (no `use utf8`), so its lc() lowercases ASCII
A-Z ONLY and accents are handled by explicit maps. We replicate that with
`_ascii_lower` — using Python's unicode str.lower() would diverge.

Verified byte-for-byte against the Perl output (see verify_tok.sh).

    python sicilian_tok.py sc < in > out
    python sicilian_tok.py en < in > out
"""
from __future__ import annotations
import re
import sys

_ASCII = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _ascii_lower(s: str) -> str:
    return s.translate(_ASCII)


def _sub_map(s: str, pairs) -> str:
    for a, b in pairs:
        s = s.replace(a, b)
    return s


# acute -> grave (incl. uppercase acute -> lowercase grave)
_SWAP = [("á", "à"), ("é", "è"), ("í", "ì"), ("ó", "ò"), ("ú", "ù"),
         ("Á", "à"), ("É", "è"), ("Í", "ì"), ("Ó", "ò"), ("Ú", "ù")]
# grave/acute/diaeresis + ç -> ASCII (keeps circumflex)
_RID = [("à", "a"), ("è", "e"), ("ì", "i"), ("ò", "o"), ("ù", "u"),
        ("À", "a"), ("È", "e"), ("Ì", "i"), ("Ò", "o"), ("Ù", "u"),
        ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
        ("Á", "a"), ("É", "e"), ("Í", "i"), ("Ó", "o"), ("Ú", "u"),
        ("ä", "a"), ("ë", "e"), ("ï", "i"), ("ö", "o"), ("ü", "u"),
        ("Ä", "a"), ("Ë", "e"), ("Ï", "i"), ("Ö", "o"), ("Ü", "u"),
        ("Ç", "c"), ("ç", "c")]
# circumflex -> ASCII
_CIRC = [("â", "a"), ("ê", "e"), ("î", "i"), ("ô", "o"), ("û", "u"),
         ("Â", "a"), ("Ê", "e"), ("Î", "i"), ("Ô", "o"), ("Û", "u")]


def rm_malice(s: str) -> str:
    s = s.replace("@", " ")
    s = re.sub(r"([$%&])", r"\1 ", s)
    s = s.replace("`", "'")
    s = _sub_map(s, [("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("«", '"'), ("»", '"')])
    return _sub_map(s, [("{", "("), ("}", ")"), ("[", "("), ("]", ")")])


def rid_accents(s: str) -> str:
    return _sub_map(s, _RID)


def rid_circum(s: str) -> str:
    return _sub_map(s, _CIRC)


def swap_accents(s: str) -> str:
    return _sub_map(s, _SWAP)


_KEEP = {"sì", "si'", "è", "e'", "n'è", "n'e'", "c'è", "c'e'"}
_ARTICLE = {"'u": "lu", "'a": "la", "'i": "li", "'n": "in", "n": "in"}
_REPL = {"cchiu": "chiu", "cci": "ci", "dopu": "doppu", "libru": "libbru",
         "non": "nun", "peggiu": "peju", "pir": "pi", "pri": "pi",
         "pirchi": "picchi", "soccu": "zoccu", "sunu": "sunnu"}
# mistaken contractions (no accent) + proper (circumflex) ones
_UNC = {
    "co": "cu lu", "che": "cu li", "do": "di lu", "de": "di li",
    "pu": "pi lu", "pa": "pi la", "pe": "pi li",
    "nno": "nni lu", "nnu": "nni lu", "nne": "nni li",
    "nto": "nta lu", "ntu": "nta lu", "nte": "nta li",
    "ntro": "ntra lu", "ntre": "ntra li",
    "on": "a un", "cun": "c'un", "dun": "d'un", "pun": "p'un",
    "nnun": "nni un", "ntun": "nta un", "ntrun": "ntra un",
    "he": "haiu a", "hanna": "hannu a",
    "ô": "a lu", "cû": "cu lu", "cô": "cu lu", "câ": "cu la", "chî": "cu li", "chê": "cu li",
    "dû": "di lu", "dô": "di lu", "dâ": "di la", "dî": "di li", "dê": "di li",
    "pû": "pi lu", "pô": "pi lu", "pâ": "pi la", "pî": "pi li", "pê": "pi li",
    "nnû": "nni lu", "nnô": "nni lu", "nnâ": "nni la", "nnî": "nni li", "nnê": "nni li",
    "ntû": "nta lu", "ntô": "nta lu", "ntâ": "nta la", "ntî": "nta li", "ntê": "nta li",
    "ntrû": "ntra lu", "ntrô": "ntra lu", "ntrâ": "ntra la", "ntrî": "ntra li", "ntrê": "ntra li",
    "ôn": "a un", "cûn": "c'un", "dûn": "d'un", "pûn": "p'un",
    "nnûn": "nn'un", "ntûn": "nta un", "ntôn": "nta un", "ntrôn": "ntra un", "ntrûn": "ntra un",
    "hê": "haiu a", "hannâ": "hannu a", "hâ": "havi a",
}


def uncontract(word: str, nxt: str) -> str:
    if word not in ("â", "ê"):
        return _UNC.get(word, word)
    if re.search(r"ari$", nxt) or re.search(r"iri$", nxt):
        return {"â": "havi a", "ê": "haiu a"}[word]
    return {"â": "a la", "ê": "a li"}[word]


def sc_tokenizer(line: str) -> str:
    line = _ascii_lower(line)
    line = _sub_map(line, [("È", "è"), ("É", "è"), ("Ì", "ì"), ("Í", "ì")])
    line = re.sub(r'([-".,:;!?()])', r" \1 ", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = " " + line + " "
    line = _sub_map(line, [(" cu un ", " c'un "), (" di un ", " d'un ")])
    line = _sub_map(line, [(" po' ", " po "), (" vo' ", " vo "), (" me' ", " me "),
                           (" to' ", " to "), (" so' ", " so ")])
    line = _sub_map(line, [("«", ' " '), ("»", ' " ')])
    line = re.sub(r'"\s+"', ' " ', line)
    line = re.sub(r"\s+", " ", line)
    line = line.replace("' ", "'")
    line = _sub_map(line, [("'è ", "' e' "), ("'e'", "' e' "), (" c' e' ", " c'e' ")])
    # conjunctive-pronoun contractions
    pron = []
    for v, art in [("û", "lu"), ("â", "la"), ("î", "li")]:
        for p, full in [("m", "mi"), ("t", "ti"), ("ci", "ci"), ("cci", "ci"),
                        ("s", "si"), ("n", "ni"), ("v", "vi")]:
            pron.append((f" {p}{v} ", f" {full} {art} "))
            pron.append((f" {p}'{v} ", f" {full} {art} "))
    # the Perl uses ' c'î ' / ' cc'î ' (not ci'/cci') in the î apostrophe block; patch those
    line = _sub_map(line, pron)
    line = _sub_map(line, [(" c'î ", " ci li "), (" cc'î ", " ci li ")])
    line = re.sub(r"\s+", " ", line).strip()
    line = swap_accents(line)

    words = line.split(" ")
    out = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i != len(words) - 1 else ""
        if w not in _KEEP:
            nw = rid_accents(w)
            nw = _ARTICLE.get(nw, nw)
            nw = _REPL.get(nw, nw)
            nw = uncontract(nw, nxt)
            out.append(nw)
        else:
            out.append({"sì": "si'", "è": "e'", "c'è": "c'e'", "n'è": "n'e'"}.get(w, w))
    line = " ".join(out)
    line = rid_accents(line)
    line = rid_circum(line)
    line = line.replace("'", "' ")
    line = _ascii_lower(line)
    return re.sub(r"\s+", " ", line).strip()


def en_tokenizer(line: str) -> str:
    line = _ascii_lower(line)
    line = re.sub(r'([-".,:;!?()])', r" \1 ", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = " " + line + " "
    line = rid_accents(line)
    line = rid_circum(line)
    line = _ascii_lower(line)
    line = re.sub(r"([a-z])'s ", r"\1 ~~'s ", line)
    return re.sub(r"\s+", " ", line).strip()


def rm_morejunk(line: str) -> str:
    line = re.sub(r"[—–―]", "-", line)
    line = re.sub(r"[šŠ]", "s", line)
    line = line.replace("…", " . . . ").replace("œ", "oe").replace("æ", "ae")
    return re.sub(r"\s+", " ", line).strip()


def tokenize(line: str, lang: str = "sc") -> str:
    line = rm_malice(line).replace("~~~", " ")
    line = sc_tokenizer(line) if lang == "sc" else en_tokenizer(line)
    return rm_morejunk(line)


def main() -> None:
    lang = sys.argv[1] if len(sys.argv) > 1 else "sc"
    for line in sys.stdin:
        print(tokenize(line.rstrip("\n"), lang))


if __name__ == "__main__":
    main()
