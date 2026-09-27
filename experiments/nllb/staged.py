"""Staged NLLB+LoRA training (the recipe of §7.8) as one parametrised, resumable run.

Stage 2 trains the adapter on the forward-scored back-translations of E. Wdowiak (the `topk`
lowest-scoring pairs of each file, after the filters below) plus natural WikiMatrix it-en;
stage 3 continues it on the curated set (scn<->en, scn<->it). Each stage checkpoints on Drive
and resumes after a disconnection; stage 3 keeps one checkpoint per epoch, the epoch is chosen
on the 1,000-pair validation set (held out of both stages), and only that one is scored on the
test set. Results go to OUT/results_<name>.json, so that the variants of the §7.10 ablation
can be compared with one another and with results_full.json.

    from staged import run
    run('base', OUT, DATA)                                   # §7.8's settings
    run('bt300', OUT, DATA, topk=300000)                     # + more back-translations
"""
from __future__ import annotations
import json
import math
import os
import re
from difflib import SequenceMatcher
from glob import glob

from nllb_pipeline import ATTN, load_base, attach_lora, build_dataset, finetune, translate, score
from normalize_scn import normalize

FILES = [('nllb-as-mono-eng_bt-s2e.scores.txt', 'scn', 'en'),
         ('nllb-as-mono-scn_bt-e2s.scores.txt', 'en', 'scn'),
         ('nllb-as-mono-scn_bt-i2s.scores.txt', 'it', 'scn'),
         ('wm-million-as-mono-ita_bt-s2i.scores.txt', 'scn', 'it'),
         ('as-mono-scn_bt-e2s.scores.txt', 'en', 'scn'),
         ('as-mono-scn_bt-i2s.scores.txt', 'it', 'scn')]
_TAG = re.compile(r'^<2[a-z]{3}>\s*')
_W = re.compile(r"[a-zàèéìòùâêîôû']+")


def read(p):
    return open(p, encoding='utf-8').read().splitlines()


def sicilianish(t):
    """Italian leaks into the low-score end: at most 1 word in -o, and no more than words in -u."""
    ws = [w for w in _W.findall(t.lower()) if len(w) >= 3]
    o = sum(w.endswith(('o', 'ò')) for w in ws)
    return o <= 1 and o <= sum(w.endswith(('u', 'ù')) for w in ws)


def load_scored(path, src, tgt, k, ban, minw=5, copy=0.9):
    """The k lowest-scoring pairs of one score file (column 1 is Sockeye's negative log
    probability that the target translates the source, so lower is better), dropping <unk>
    lines, short fragments, near-copies, Italian passing as Sicilian and anything in `ban`."""
    rows = []
    for line in read(path):
        if '<unk>' in line:
            continue
        p = line.split('\t')
        if len(p) < 3:
            continue
        try:
            sc = float(p[0])
        except ValueError:
            continue
        s = _TAG.sub('', p[1]).strip(); t = p[2].strip()
        if s and t and s != t and s not in ban and t not in ban:
            rows.append((sc, s, t))
    rows.sort(key=lambda r: r[0])
    kept = []
    for sc, s, t in rows:
        if len(s.split()) < minw or len(t.split()) < minw:
            continue
        if not sicilianish(t if tgt == 'scn' else s if src == 'scn' else 'u'):
            continue
        sm = SequenceMatcher(None, s.lower(), t.lower())
        if sm.quick_ratio() >= copy and sm.ratio() >= copy:
            continue
        kept.append((sc, s, t))
        if len(kept) == k:
            break
    cut = kept[-1][0] if kept else float('nan')
    print(f'  {os.path.basename(path)}: kept {len(kept)} ({src}->{tgt}), '
          f'worst kept score {cut:.3f} = p {100 * math.exp(-cut):.1f}%')
    return ([r[1] for r in kept], [r[2] for r in kept], src, tgt)


def _scores(model, tok, sets, dec_len=160, beams=5):
    return {f'{s}->{t}': score(translate(model, tok, sets[s], s, t, max_len=dec_len, beams=beams), sets[t])
            for s, t in [('scn', 'en'), ('en', 'scn')]}


def run(name, out, data, topk=100000, r=32, alpha=64, targets=ATTN,
        stage2_epochs=1, stage3_epochs=2, model_id='facebook/nllb-200-1.3B',
        batch_size=16, grad_accum=1, max_len=128, dec_len=160, beams=5):
    """One variant of the staged training; returns (and saves) its results."""
    res_path = f'{out}/results_{name}.json'
    if os.path.exists(res_path):
        print(f'{name}: already done'); return json.load(open(res_path))
    bt, ft_dir = f'{out}/eryk', f'{out}/finetune'
    test = {'scn': [normalize(x, 'std') for x in read(f'{data}/test.scn')], 'en': read(f'{data}/test.en')}
    valid = {'scn': [normalize(x, 'std') for x in read(f'{data}/valid.scn')], 'en': read(f'{data}/valid.en')}
    ban = (set(test['scn']) | set(read(f'{data}/test.scn')) | set(test['en'])
           | set(valid['scn']) | set(read(f'{data}/valid.scn')) | set(valid['en']))
    s2_dir, s3_dir = f'{out}/nllb-lora-{name}-stage2', f'{out}/nllb-lora-{name}'

    # stage 2: back-translations + natural it-en (skipped if its adapter is already saved)
    if not os.path.exists(f'{s2_dir}/adapter_config.json'):
        stage2 = [load_scored(f'{bt}/{f}', s, t, topk, ban) for f, s, t in FILES]
        wm = [(e, i) for e, i in zip(read(f'{bt}/wm-million_tkn2ascii.en-it.txt.en'),
                                      read(f'{bt}/wm-million_tkn2ascii.en-it.txt.it'))
              if e not in ban][:topk]
        stage2 += [([e for e, _ in wm], [i for _, i in wm], 'en', 'it'),
                   ([i for _, i in wm], [e for e, _ in wm], 'it', 'en')]
        m, tok = load_base(model_id); ad = attach_lora(m, r=r, alpha=alpha, targets=targets)
        finetune(ad, tok, build_dataset(tok, stage2, max_len=max_len), out_dir=s2_dir, epochs=stage2_epochs,
                 batch_size=batch_size, grad_accum=grad_accum, save='steps')

    # stage 3: curated set, one checkpoint per epoch
    m, tok = load_base(model_id); ad = attach_lora(m, adapter=s2_dir)
    tr = {'scn': read(f'{ft_dir}/train.scn-en.scn'), 'en': read(f'{ft_dir}/train.scn-en.en')}
    ti = {'scn': read(f'{ft_dir}/train.scn-it.scn'), 'it': read(f'{ft_dir}/train.scn-it.it')}
    se = [(s, e) for s, e in zip(tr['scn'], tr['en']) if s not in ban and e not in ban]
    si = [(s, i) for s, i in zip(ti['scn'], ti['it']) if s not in ban]
    ds3 = build_dataset(tok, [([s for s, _ in se], [e for _, e in se], 'scn', 'en'),
                              ([e for _, e in se], [s for s, _ in se], 'en', 'scn'),
                              ([s for s, _ in si], [i for _, i in si], 'scn', 'it'),
                              ([i for _, i in si], [s for s, _ in si], 'it', 'scn')],
                        max_len=max_len)
    finetune(ad, tok, ds3, out_dir=s3_dir, epochs=stage3_epochs,
             batch_size=batch_size, grad_accum=grad_accum, save='epoch')

    # choose the epoch on validation (sum of the two BLEU), then score it on the test set
    ckpts = sorted(glob(f'{s3_dir}-trainer/checkpoint-*'), key=lambda p: int(p.rsplit('-', 1)[1]))
    curve, best = [], None
    for ep, ck in enumerate(ckpts, 1):
        m, tok = load_base(model_id, adapter=ck)
        v = _scores(m, tok, valid, dec_len, beams)
        curve.append({'epoch': ep, 'valid': v})
        print(f'{name} epoch {ep}: valid {v}')
        if best is None or sum(b for b, _ in v.values()) > sum(b for b, _ in best[1].values()):
            best = (ck, v, ep)
    m, tok = load_base(model_id, adapter=best[0])
    res = {'config': dict(topk=topk, r=r, alpha=alpha, targets=list(targets),
                          stage2_epochs=stage2_epochs, stage3_epochs=stage3_epochs, model=model_id,
                          max_len=max_len, dec_len=dec_len, beams=beams),
           'valid_curve': curve, 'epoch': best[2], 'test': _scores(m, tok, test, dec_len, beams)}
    print(f'{name}: epoch {best[2]} chosen on validation, test {res["test"]}')
    json.dump(res, open(res_path, 'w'), indent=2)
    return res
