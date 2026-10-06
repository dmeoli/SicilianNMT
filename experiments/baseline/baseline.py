"""Our from-scratch Sockeye baseline and its two levers, re-run on Colab on the current web set
(the upper rows of Table 2 of the paper), so that every reported number comes from one script.

The model is the small high-dropout Transformer of train.sh (3 layers, model size 256, 4 heads,
feed-forward 1024, shared vocabulary, dropout 0.4 on the embeddings and 0.2 elsewhere, label
smoothing 0.1, Adam with learning rate 2e-4, batches of 1024 words, at most 30 epochs, early
stopping after 8 checkpoints without improvement, a checkpoint every 500 updates, sequences of at
most 100 subwords), trained for scn->en on the web set (data/train.*) and stopped on its 1,000
validation pairs. The variants are cumulative:

    baseline  raw text, joint BPE of 4000 merges (01_subword.sh)
    leverB    Napizia tokenization, joint BPE of 4000 merges learned with the Dieli and Chiu da
              Palora inflections appended to the Sicilian side (lever_b_prep.sh)
    leverD    leverB plus a lemma source factor summed into the source embedding (build_factors.py,
              train_factors.sh)

All of them are scored on our test set in the space of the Napizia tokenizer (the hypothesis and
the reference are both tokenized, BLEU on whitespace tokens and chrF), together with the floor
obtained by copying the source, which is also given on raw text.

    from baseline import run
    run('baseline', f'{OUT}/baseline', DATA, sockeye_bin='/content/sk/bin')
"""
from __future__ import annotations
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'tokenization'))
sys.path.append(_HERE)
from sicilian_tok import tokenize  # noqa: E402
from recipe import _read, _write, _sh, DESINENCES  # noqa: E402

TRAIN_ARGS = ['--shared-vocab', '--seed', '13', '--max-seq-len', '100',
              '--batch-size', '1024', '--batch-type', 'word',
              '--max-num-epochs', '30', '--checkpoint-interval', '500',
              '--max-num-checkpoint-not-improved', '8',
              '--encoder', 'transformer', '--decoder', 'transformer',
              '--num-layers', '3', '--num-embed', '256',
              '--transformer-model-size', '256', '--transformer-attention-heads', '4',
              '--transformer-feed-forward-num-hidden', '1024',
              '--embed-dropout', '0.4', '--transformer-dropout-attention', '0.2',
              '--transformer-dropout-act', '0.2', '--transformer-dropout-prepost', '0.2',
              '--label-smoothing', '0.1', '--optimizer', 'adam', '--initial-learning-rate', '0.0002']
FACTOR_ARGS = ['--source-factors-num-embed', '256', '--source-factors-combine', 'sum']


def _scores(hyp, ref):
    """BLEU and chrF in the space of the Napizia tokenizer (English side)."""
    from sacrebleu.metrics import BLEU, CHRF
    h, r = [tokenize(x, 'en') for x in hyp], [tokenize(x, 'en') for x in ref]
    return [round(BLEU(tokenize='none', force=True).corpus_score(h, [r]).score, 2),
            round(CHRF().corpus_score(h, [r]).score, 2)]


def floor(data):
    """Copy-source floor on our test set, on raw text and in the tokenized space."""
    from sacrebleu.metrics import BLEU, CHRF
    src, ref = _read(f'{data}/test.scn'), _read(f'{data}/test.en')
    raw = [round(BLEU().corpus_score(src, [ref]).score, 2), round(CHRF().corpus_score(src, [ref]).score, 2)]
    return {'raw': raw, 'tokenized': _scores([tokenize(x, 'sc') for x in src], ref)}


def prepare(variant, root, data):
    """Tokenize (levers only), learn the joint BPE and apply it; factors for leverD."""
    w = f'{root}/work-{variant}'
    if os.path.exists(f'{w}/done'):
        return w
    os.makedirs(w, exist_ok=True)
    tok = variant != 'baseline'
    for split in ('train', 'valid', 'test'):
        for lang, t in (('scn', 'sc'), ('en', 'en')):
            lines = _read(f'{data}/{split}.{lang}')
            _write(f'{w}/{split}.tok.{lang}', [tokenize(x, t) for x in lines] if tok else lines)
    scn = _read(f'{w}/train.tok.scn')
    if tok:
        scn += sorted({tokenize(x, 'sc') for x in _read(DESINENCES)} - {''})
    _write(f'{w}/bpe_input', scn + _read(f'{w}/train.tok.en'))
    from subword_nmt.apply_bpe import BPE
    from subword_nmt.learn_bpe import learn_bpe
    with open(f'{w}/bpe_input', encoding='utf-8') as i, open(f'{w}/codes', 'w', encoding='utf-8') as o:
        learn_bpe(i, o, 4000)
    with open(f'{w}/codes', encoding='utf-8') as c:
        bpe = BPE(c)
    for split in ('train', 'valid', 'test'):
        for lang in ('scn', 'en'):
            _write(f'{w}/{split}.bpe.{lang}', [bpe.process_line(x) for x in _read(f'{w}/{split}.tok.{lang}')])
    if variant == 'leverD':
        from build_factors import lemma_of
        for split in ('train', 'valid', 'test'):
            fac = []
            for t, b in zip(_read(f'{w}/{split}.tok.scn'), _read(f'{w}/{split}.bpe.scn')):
                words, wi, f = t.split(), 0, []
                for s in b.split():
                    f.append(lemma_of(words[wi] if wi < len(words) else ''))
                    if not s.endswith('@@'):
                        wi += 1
                fac.append(' '.join(f))
            _write(f'{w}/{split}.fac.scn', fac)
    open(f'{w}/done', 'w').close()
    return w


def run(variant, root, data, sockeye_bin):
    """Train (or reuse) one variant for scn->en; return its scores on our test set."""
    os.makedirs(root, exist_ok=True)
    res_path = f'{root}/results_baseline.json'
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if 'floor' not in res:
        res['floor'] = floor(data)
    if variant in res:
        print(variant, res[variant]); return res[variant]
    w = prepare(variant, root, data)
    fac = variant == 'leverD'
    prep, model = f'{w}/prep', f'{w}/model'
    sk = lambda tool: os.path.join(sockeye_bin, tool)
    if not (os.path.isdir(prep) and os.listdir(prep)):
        _sh(sk('sockeye-prepare-data'), '--source', f'{w}/train.bpe.scn', '--target', f'{w}/train.bpe.en',
            *(['--source-factors', f'{w}/train.fac.scn'] if fac else []),
            '--shared-vocab', '--max-seq-len', '100', '--output', prep)
    if not os.path.exists(f'{model}/params.best'):
        if os.path.isdir(model) and not os.path.isdir(f'{model}/training_state'):
            shutil.rmtree(model)       # a failed start, not a resumable run: Sockeye would refuse it
        _sh(sk('sockeye-train'), '--prepared-data', prep, '--output', model,
            '--validation-source', f'{w}/valid.bpe.scn', '--validation-target', f'{w}/valid.bpe.en',
            *(['--validation-source-factors', f'{w}/valid.fac.scn'] if fac else []),
            *TRAIN_ARGS, *(FACTOR_ARGS if fac else []))
    out = f'{w}/test.hyp'
    _sh(sk('sockeye-translate'), '--models', model, '--input', f'{w}/test.bpe.scn', '--output', out,
        *(['--input-factors', f'{w}/test.fac.scn'] if fac else []), '--beam-size', '5')
    hyp = [l.replace('@@ ', '').replace('@@', '').strip() for l in _read(out)]
    res[variant] = _scores(hyp, _read(f'{data}/test.en'))
    json.dump(res, open(res_path, 'w'), indent=2)
    print(variant, res[variant])
    return res[variant]
