"""The 2022 baseline of E. Wdowiak, re-run with his Sockeye configuration on three training
sets (A his hand-aligned sheets, B our automatic alignment, C both; see recipe_data.py), and
scored on his own test in the space of the Napizia tokenizer, where he scored (§7.12).

His configuration (dataset/sockeye_n30_sw3000 in the history of his repository): Napizia
tokenization (accents folded, contractions undone), separate subword vocabularies per language
with 3000 merges, the Sicilian one learned with the Dieli and Chiu da Palora inflections
appended once each, and a Transformer with 3 layers, model size 256, 4 heads, feed-forward
1024, dropout 0.5 on the embeddings and 0.25 elsewhere, label smoothing 0.1, Adam with learning
rate 1.5e-4, batches of 20 sentences, at most 20 epochs, early stopping after 4 checkpoints
without improvement, a checkpoint every 725 updates. Two deliberate differences: he stopped on
his test set, we stop on 500 lines of our validation set, and he started each model from the
previous one (c00), we start from scratch.

Sockeye 3 pins an old torch, so it runs in its own environment (`sockeye_bin`); tokenization,
subwords and scoring run in the notebook's Python.

    from recipe import run
    run('A', f'{OUT}/recipe', 'scn', 'en', sockeye_bin='/content/sk/bin')
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'tokenization'))
from sicilian_tok import tokenize  # noqa: E402

DESINENCES = os.path.join(_HERE, '..', '..', 'vocab', 'dieli-cchiu-vocab.txt')
TOK = {'scn': 'sc', 'en': 'en'}
TRAIN_ARGS = ['--batch-size', '20', '--batch-type', 'sentence',
              '--max-num-epochs', '20', '--max-num-checkpoint-not-improved', '4',
              '--checkpoint-interval', '725', '--max-seq-len', '200',
              '--encoder', 'transformer', '--decoder', 'transformer',
              '--num-layers', '3', '--num-embed', '256',
              '--transformer-model-size', '256', '--transformer-attention-heads', '4',
              '--transformer-feed-forward-num-hidden', '1024',
              '--embed-dropout', '0.5', '--transformer-dropout-attention', '0.25',
              '--transformer-dropout-act', '0.25', '--transformer-dropout-prepost', '0.25',
              '--label-smoothing', '0.1', '--optimizer', 'adam',
              '--initial-learning-rate', '0.00015', '--seed', '13']


def _read(p):
    return open(p, encoding='utf-8').read().splitlines()


def _write(p, lines):
    open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


def _sh(*cmd):
    """Run a command; on failure show the tail of its output, which Colab would not print."""
    print('$', ' '.join(cmd)[:160])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        print('\n'.join((p.stdout + p.stderr).splitlines()[-60:]))
        raise RuntimeError(f'{os.path.basename(cmd[0])} failed with exit status {p.returncode}')


def prepare(name, root):
    """Tokenize the set, learn one BPE per language (3000 merges) and apply it."""
    w = f'{root}/work-{name}'
    if os.path.exists(f'{w}/done'):
        return w
    os.makedirs(w, exist_ok=True)
    for split, stem in (('train', name), ('valid', 'valid'), ('test', 'test_eryk')):
        for lang in ('scn', 'en'):
            _write(f'{w}/{split}.tok.{lang}', [tokenize(x, TOK[lang]) for x in _read(f'{root}/{stem}.{lang}')])
    des = sorted({tokenize(x, 'sc') for x in _read(DESINENCES)} - {''})
    _write(f'{w}/bpe_input.scn', _read(f'{w}/train.tok.scn') + des)
    _write(f'{w}/bpe_input.en', _read(f'{w}/train.tok.en'))
    from subword_nmt.apply_bpe import BPE
    from subword_nmt.learn_bpe import learn_bpe
    for lang in ('scn', 'en'):
        with open(f'{w}/bpe_input.{lang}', encoding='utf-8') as i, \
                open(f'{w}/codes.{lang}', 'w', encoding='utf-8') as o:
            learn_bpe(i, o, 3000)
        with open(f'{w}/codes.{lang}', encoding='utf-8') as c:
            bpe = BPE(c)
        for split in ('train', 'valid', 'test'):
            _write(f'{w}/{split}.bpe.{lang}', [bpe.process_line(x) for x in _read(f'{w}/{split}.tok.{lang}')])
    open(f'{w}/done', 'w').close()
    return w


def bleu_tokenized(hyp, ref):
    """Corpus BLEU on whitespace tokens of already tokenized text (as his GluonNLP bleu.py)."""
    from sacrebleu.metrics import BLEU
    return round(BLEU(tokenize='none', force=True).corpus_score(hyp, [ref]).score, 2)


def run(name, root, src, tgt, sockeye_bin):
    """Train (or reuse) the model for one set and direction; return its BLEU on his test."""
    res_path = f'{root}/results_recipe.json'
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    key = f'{name} {src}->{tgt}'
    if key in res:
        print(key, res[key]); return res[key]
    w = prepare(name, root)
    prep, model = f'{w}/prep-{src}2{tgt}', f'{w}/model-{src}2{tgt}'
    sk = lambda tool: os.path.join(sockeye_bin, tool)
    if not (os.path.isdir(prep) and os.listdir(prep)):
        _sh(sk('sockeye-prepare-data'), '--source', f'{w}/train.bpe.{src}', '--target', f'{w}/train.bpe.{tgt}',
            '--max-seq-len', '200', '--output', prep)
    if not os.path.exists(f'{model}/params.best'):
        if os.path.isdir(model) and not os.path.isdir(f'{model}/training_state'):
            shutil.rmtree(model)       # a failed start, not a resumable run: Sockeye would refuse it
        _sh(sk('sockeye-train'), '--prepared-data', prep, '--output', model,
            '--validation-source', f'{w}/valid.bpe.{src}', '--validation-target', f'{w}/valid.bpe.{tgt}',
            *TRAIN_ARGS)
    out = f'{w}/test.hyp.{src}2{tgt}'
    _sh(sk('sockeye-translate'), '--models', model, '--input', f'{w}/test.bpe.{src}', '--output', out,
        '--beam-size', '5')
    hyp = [l.replace('@@ ', '').replace('@@', '').strip() for l in _read(out)]
    score = bleu_tokenized(hyp, _read(f'{w}/test.tok.{tgt}'))
    res[key] = score
    json.dump(res, open(res_path, 'w'), indent=2)
    print(key, score)
    return score
