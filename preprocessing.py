import argparse
import json
import re
from pathlib import Path
import pandas as pd


def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if
                    line.strip() and not line.startswith('#')])


def load_lemma_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_fixes(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_fixes(text, fixes):
    for src, tgt in fixes.items():
        text = text.replace(f" {src} ", f" {tgt} ")
    return text


def remove_stopwords(text, stopwords):
    words = text.split()
    return ' '.join([w for w in words if w not in stopwords])


def lemmatize(text, lemma_dict):
    return ' '.join([lemma_dict.get(w, w) for w in text.split()])


def normalize_spaces(text):
    text = re.sub(r"[^a-zA-Z0-9'~ ]+", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess(text, fixes, stopwords, lemmas):
    text = text.lower()
    text = " " + text + " "
    text = apply_fixes(text, fixes)
    text = normalize_spaces(text)
    text = remove_stopwords(text, stopwords)
    text = lemmatize(text, lemmas)
    return text.strip()


def run_preprocessing(input_csv):
    input_path = Path(input_csv)
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = load_stopwords('vocab/stopwords_scn.txt')
    lemma_dict = load_lemma_dict('vocab/lemma_dict_scn.json')
    fixes = load_fixes('vocab/unigramify_scn.json')

    df = pd.read_csv(input_path)
    assert 'scn' in df.columns and 'en' in df.columns, "CSV must have 'scn' and 'en' columns."

    src_lines = []
    tgt_lines = []

    for scn, en in zip(df['scn'], df['en']):
        if pd.isna(scn) or pd.isna(en):
            continue
        src = preprocess(str(scn), fixes, stopwords, lemma_dict)
        tgt = str(en).strip()
        if src and tgt:
            src_lines.append(src)
            tgt_lines.append(tgt)

    with open(output_dir / 'train.src', 'w', encoding='utf-8') as fsrc, \
            open(output_dir / 'train.tgt', 'w', encoding='utf-8') as ftgt:
        for s, t in zip(src_lines, tgt_lines):
            fsrc.write(s + '\n')
            ftgt.write(t + '\n')

    print(
        f"✅ Preprocessing complete. {len(src_lines)} examples written to data/processed/.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str,
                        help='Path to input CSV with \"scn\" and \"en\" columns')
    args = parser.parse_args()
    run_preprocessing(args.input_csv)
