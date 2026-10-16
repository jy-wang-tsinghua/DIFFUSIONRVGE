"""Build the prompt panels used by the training scripts from Flow-GRPO's public prompt files.

    python scripts/prepare_data.py                    # downloads the six source files (pinned commit)
    python scripts/prepare_data.py --source DIR       # or reads them from DIR/{pickscore,ocr,geneval}/...

Panels (JSON with train_prompts, eval_prompts and, for GenEval, per-prompt metadata):
  dataset/pickscore/panel.json      all Pick-a-Pic training prompts; validation = the first 64 test prompts
  dataset/ocr/panel.json            OCR training prompts, de-duplicated in order; evaluation = all test prompts
  dataset/geneval/panel_test.json   GenEval training prompts, de-duplicated in order; evaluation = the 553 test prompts
  dataset/geneval/panel_train.json  GenEval training rows (duplicates kept) minus 440 held-out validation prompts
                                    (dataset/geneval_val_heldout_indices.json indexes the de-duplicated training prompts)
"""
import argparse
import json
import os
import urllib.request

URL = 'https://raw.githubusercontent.com/yifan123/flow_grpo/9745241cf1101407e9b518dd4f10498092150522/dataset/'
FILES = ['pickscore/train.txt', 'pickscore/test.txt', 'ocr/train.txt', 'ocr/test.txt',
         'geneval/train_metadata.jsonl', 'geneval/test_metadata.jsonl']
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')


def read(source, name):
    if source:
        with open(os.path.join(source, name), encoding='utf-8') as f:
            return f.read().splitlines()
    with urllib.request.urlopen(URL + name, timeout=120) as r:
        return r.read().decode('utf-8').splitlines()


def dedup(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def write(name, panel):
    path = os.path.join(ROOT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(panel, f, ensure_ascii=False)
    print(f'{name}: {len(panel["train_prompts"])} training / {len(panel["eval_prompts"])} evaluation prompts')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', help='directory with the six source files instead of downloading them')
    args = p.parse_args()
    src = {name: read(args.source, name) for name in FILES}
    lines = lambda name: [l.strip() for l in src[name] if l.strip()]

    write('pickscore/panel.json', dict(train_prompts=lines('pickscore/train.txt'),
                                       eval_prompts=lines('pickscore/test.txt')[:64],
                                       scope='Pick-a-Pic training prompts; validation = test prompts [0:64]'))
    write('ocr/panel.json', dict(train_prompts=dedup(lines('ocr/train.txt')), eval_prompts=lines('ocr/test.txt'),
                                 scope='OCR training prompts de-duplicated in order; evaluation = all test prompts'))

    rows = [json.loads(l) for l in src['geneval/train_metadata.jsonl'] if l.strip()]
    test = [json.loads(l) for l in src['geneval/test_metadata.jsonl'] if l.strip()]
    train_prompts = dedup([r['prompt'] for r in rows])
    metadata = {}
    for r in rows + test:
        metadata.setdefault(r['prompt'], r)
    write('geneval/panel_test.json', dict(train_prompts=train_prompts, eval_prompts=dedup([r['prompt'] for r in test]),
                                          metadata=metadata, scope='GenEval training prompts; evaluation = 553 test prompts'))
    with open(os.path.join(ROOT, 'geneval_val_heldout_indices.json')) as f:
        held = [train_prompts[i] for i in json.load(f)]
    held_set = set(held)
    write('geneval/panel_train.json', dict(train_prompts=[r['prompt'] for r in rows if r['prompt'] not in held_set],
                                           eval_prompts=held, metadata=metadata,
                                           scope='GenEval training rows minus 440 held-out validation prompts'))


if __name__ == '__main__':
    main()
