"""Export of the final model to CTranslate2 with int8 weights, and a CPU benchmark of it, to size
the cost of serving it without a GPU.

export() merges the LoRA adapter into NLLB-200 1.3B (fp32, on the CPU) and converts the merged
model with int8 weights, keeping the tokenizer next to it, so that the output folder is all a
server needs. bench() then measures, on the CPU and with 1, 2 and 4 threads, the time to load the
model, its resident memory and the latency of one sentence at a time (beam 5, as in the paper)
on the first 100 English sources of our test set, translated into Sicilian; finally it
translates the 8 test files of cell 7.9 with int8 weights and scores them next to the fp16 GPU
hypotheses of that cell, raw and in the tokenized space of Wdowiak, so that the loss due to
the quantization is measured on the same lines. Each thread count runs in its own process, so
that the memory is that of the translator alone.

    from ct2_export import export, bench
    export(ck, f'{OUT}/nllb-final-ct2-int8')
    bench(f'{OUT}/nllb-final-ct2-int8', f'{OUT}/translations', f'{OUT}/results_ct2.json')
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
LANG = {"scn": "scn_Latn", "en": "eng_Latn", "it": "ita_Latn"}


def export(adapter, out, base='facebook/nllb-200-1.3B', quantization='int8', tmp='/tmp/nllb-merged'):
    """Merge `adapter` into `base` and write the CTranslate2 model to `out` (skipped if present)."""
    if os.path.exists(f'{out}/model.bin'):
        return out
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import ctranslate2
    model = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(tmp)
    tok = AutoTokenizer.from_pretrained(base)
    tok.save_pretrained(tmp)
    del model
    ctranslate2.converters.TransformersConverter(tmp).convert(out, quantization=quantization, force=True)
    tok.save_pretrained(out)
    shutil.rmtree(tmp)
    return out


def _translate(tr, tok, lines, src, tgt, batch=16, beams=5, dec_len=256):
    tok.src_lang = LANG[src]
    toks = [tok.convert_ids_to_tokens(tok.encode(x)) for x in lines]
    res = tr.translate_batch(toks, target_prefix=[[LANG[tgt]]] * len(toks), max_batch_size=batch,
                             beam_size=beams, max_decoding_length=dec_len)
    return [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True) for r in res]


def _read_tsv(p):
    rows = [l.split('\t') for l in open(p, encoding='utf-8').read().splitlines()[1:]]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def _rss_mb():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS'):
            return round(int(l.split()[1]) / 1024)


def _worker(model_dir, tr_dir, threads, quality):
    """One measurement in a fresh process; prints a JSON line."""
    import ctranslate2
    from transformers import AutoTokenizer
    t0 = time.time()
    tr = ctranslate2.Translator(model_dir, device='cpu', compute_type='int8',
                                intra_threads=threads, inter_threads=1)
    tok = AutoTokenizer.from_pretrained(model_dir)
    r = {'threads': threads, 'load_s': round(time.time() - t0, 1), 'rss_mb_loaded': _rss_mb()}
    src, _, _ = _read_tsv(f'{tr_dir}/our-test_en2scn.tsv')
    lat = []
    for x in src[:100]:
        t = time.time()
        _translate(tr, tok, [x], 'en', 'scn', batch=1)
        lat.append(time.time() - t)
    lat.sort()
    r.update({'sentences': len(lat), 'mean_words': round(sum(len(x.split()) for x in src[:100]) / len(lat), 1),
              'latency_s_median': round(lat[len(lat) // 2], 2), 'latency_s_p90': round(lat[int(0.9 * len(lat))], 2),
              'latency_s_mean': round(sum(lat) / len(lat), 2), 'rss_mb_peak': _rss_mb()})
    if quality:
        sys.path += [os.path.join(_HERE, '..', 'nllb'), os.path.join(_HERE, '..', 'dataset')]
        from staged import tokenized_bleu
        from sacrebleu.metrics import BLEU
        q = {}
        for f in sorted(os.listdir(tr_dir)):
            if not f.endswith('.tsv'):
                continue
            s, t = f[:-4].split('_')[1].split('2')
            src, ref, fp16 = _read_tsv(f'{tr_dir}/{f}')
            t1 = time.time()
            int8 = _translate(tr, tok, src, s, t)
            bleu = lambda h: round(BLEU().corpus_score(h, [ref]).score, 2)
            q[f[:-4]] = {'fp16 raw': bleu(fp16), 'int8 raw': bleu(int8),
                         'fp16 tokenized': tokenized_bleu(fp16, ref, t), 'int8 tokenized': tokenized_bleu(int8, ref, t),
                         'identical': sum(a == b for a, b in zip(fp16, int8)), 'lines': len(src),
                         'batched_s_per_sentence': round((time.time() - t1) / len(src), 3)}
        r['quality'] = q
    print(json.dumps(r))


def bench(model_dir, tr_dir, out, threads=(1, 2, 4)):
    """Run the measurements (skipped if `out` exists) and return them."""
    if os.path.exists(out):
        return json.load(open(out))
    size = sum(os.path.getsize(f'{model_dir}/{f}') for f in os.listdir(model_dir))
    res = {'cpu': open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].split(':', 1)[1].strip(),
           'cpu_count': os.cpu_count(), 'model_mb': round(size / 2**20), 'runs': []}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for n in threads:
        p = subprocess.run([sys.executable, __file__, model_dir, tr_dir, str(n), str(int(n == max(threads)))],
                           capture_output=True, text=True, env=env)
        if p.returncode:
            raise RuntimeError(p.stderr[-3000:])
        res['runs'].append(json.loads(p.stdout.strip().splitlines()[-1]))
        print(res['runs'][-1])
    json.dump(res, open(out, 'w'), indent=2)
    return res


if __name__ == '__main__':
    _worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == '1')
