"""GenEval reward for the stepwise trainer, served out of process on the rank's GPU.

THE REWARD is `reward_server/gen_eval.py` from the GenEval reward server of Flow-GRPO, called
unchanged: Mask2Former Swin-S (mmdet, sha256 743b7d99..., the checkpoint DiffusionNFT's
gen_eval.py names) finds objects, CLIP ViT-L/14 classifies colours, and
`compute_geneval` returns per image

    score          partial credit over count / colour / position clauses - the TRAINING
                   reward DiffusionNFT and Flow-GRPO use (multi_score keeps `scores`)
    correct        the GenEval benchmark judgement (count >=, colour, position)
    strict         exact counts, colours and positions

The scoring logic of that file and DiffusionNFT's in-process copy differ only in formatting.
Images go in as uint8 arrays, as DiffusionNFT passes them, not as the
JPEG bytes Flow-GRPO's HTTP client sends.

ONE CHANGE, SPEED ONLY: `color_classification` builds a DataLoader with num_workers=4 for
every image, i.e. forks four processes per colour clause, which dominates the reward's run time;
num_workers=0 gives identical scores. It is applied by
patching torch.utils.data.DataLoader inside the server process.

The detector upsamples 512 images to its 1024 test scale, so memory is large; the server runs
batch 4 and reserves it before the trainer loads its model.
Run as `python -m rvge.stepwise_geneval` with the reward_server conda env.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.connection import Client, Listener

import numpy as np

AUTHKEY = b'stepwise-geneval'
# Checkout of the GenEval reward server used by Flow-GRPO (Mask2Former + CLIP); see README.
REWARD_SERVER_DIR = os.environ.get('RVGE_GENEVAL_SERVER', '')
REWARD_SERVER_PYTHON = os.environ.get('RVGE_GENEVAL_PYTHON', sys.executable)
TAGS = ('single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr')


def reward_summary(rows):
    """GenEval benchmark aggregates of evaluation rows; empty for rows without GenEval fields.

`geneval_overall` is the official GenEval repo's number: the mean over task tags of each tag's accuracy, so tags
with more prompts do not dominate. `geneval_overall_flat` is DiffusionNFT's: `scripts/evaluation.py`
takes a plain mean over every row. The eight-column readout follows NFT and uses the flat one;
`geneval_overall` stays the panel's key.
"""
    tagged = [r for r in rows if 'tag' in r]
    if not tagged:
        return {}
    out = {}
    for key, field in (('geneval_overall', 'correct'), ('geneval_strict_overall', 'strict')):
        by_tag = {t: float(np.mean([r[field] for r in tagged if r['tag'] == t]))
                  for t in TAGS if any(r['tag'] == t for r in tagged)}
        out[key] = float(np.mean(list(by_tag.values())))
        out[key + '_by_tag'] = by_tag
    out['geneval_overall_flat'] = float(np.mean([r['correct'] for r in tagged]))       # DiffusionNFT's aggregation
    out['geneval_strict_overall_flat'] = float(np.mean([r['strict'] for r in tagged]))
    out['geneval_score_mean'] = float(np.mean([r['score'] for r in tagged]))
    return out


TRAIN_REWARDS = ('score', 'strict', 'mix', 'soft', 'soft_mix', 'soft1')
SOFT_REWARDS = ('soft', 'soft_mix', 'soft1')


def train_values(value, mode='score'):
    """The training reward of one server result. Evaluation never uses this: its records come from
result_details, i.e. the discrete GenEval judgement.

    score     partial credit, DiffusionNFT / Flow-GRPO's training reward (default)
    strict    1 only if every count, colour and position clause is exactly right
    mix       (score + strict) / 2: keeps the partial-credit gradient but takes the other half away
              from any image with a wrong clause (under `score` a wrong colour costs only half a
              clause). strict is computed on the training path anyway
              (reward_server gen_eval.py evaluate_reward), so neither option costs scorer time.
    soft      soft_image_score: the same detector and CLIP colour classifier, with every threshold and
              argmax replaced by a sigmoid or a probability, so adjacent endpoints almost never tie
    soft_mix  (score + soft) / 2
"""
    if mode == 'score':
        return list(value['scores'])
    if mode == 'strict':
        return list(value['strict'])
    if mode == 'mix':
        return [.5 * (s + t) for s, t in zip(value['scores'], value['strict'])]
    if mode in ('soft', 'soft1'):  # the server computes the one-sided variant for soft1 requests
        return list(value['soft'])
    if mode == 'soft_mix':
        return [.5 * (s + t) for s, t in zip(value['scores'], value['soft'])]
    raise ValueError(f'unknown GenEval training reward {mode!r}')


SOFT_DETECTION_TAU = .05   # detector confidence scale: 0.1 above threshold -> presence 0.88
SOFT_DETECTION_FLOOR = .2  # candidates more than this below threshold are ignored (presence < 0.018)
SOFT_POSITION_TAU = .1     # normalised-offset scale around the benchmark's 0.5 cut


def _sigmoid(x):
    return 1. / (1. + np.exp(-np.asarray(x, dtype=np.float64)))


def _presence(scores, threshold, tau=SOFT_DETECTION_TAU, one_sided=False):
    """Soft detection indicator. one_sided: exactly 1 at or above threshold (as the discrete filter), 2*sigmoid below,
so it is continuous at the threshold and pushing an already-detected object's confidence higher earns nothing."""
    z = (np.asarray(scores, dtype=np.float64) - threshold) / tau
    return np.where(z > 0, 1., 2. * _sigmoid(z)) if one_sided else _sigmoid(z)


def soft_count_clause(scores, threshold, count, tau=SOFT_DETECTION_TAU, one_sided=False):
    """Count clause of evaluate_reward (1 - |count - found| / count) with found = sum of soft presences."""
    found = float(np.sum(_presence(scores, threshold, tau, one_sided))) if len(scores) else 0.
    return max(0., 1. - abs(count - found) / count), found


def soft_relation(box_a, box_b, relation, position_threshold=.1, tau=SOFT_POSITION_TAU, one_sided=False):
    """Probability-like version of reward_server relative_position: same revised offset, sigmoid at the 0.5 cut.
one_sided: 1 once the relation holds, 2*sigmoid(margin / tau) below the cut."""
    boxes = np.array([box_a[:4], box_b[:4]], dtype=np.float64).reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    norm = np.linalg.norm(offset)
    dx, dy = (revised / norm) if norm > 0 else (0., 0.)
    margin = {'left of': -dx, 'right of': dx, 'above': -dy, 'below': dy}[relation] - .5
    if one_sided:
        return 1. if margin > 0 else float(2. * _sigmoid(margin / tau))
    return float(_sigmoid(margin / tau))


def soft_image_score(candidates, metadata, colour_prob, threshold, counting_threshold, position_threshold=.1,
                     one_sided=False):
    """Soft GenEval training reward of one image, clause for clause like evaluate_reward.

candidates: {classname: [(score, box, mask), ...]} sorted by detector score, highest first.
colour_prob(classname, objects, colour) -> per-object colour value from the CLIP colour classifier
(two-sided: P(colour); one-sided: 1 if colour is the argmax else P(colour) / P(argmax)).
Include clauses only (evaluate_reward ignores exclude clauses too). Colour and position clauses are
always present and weighted by detector presence, where the discrete reward zeroes them after a wrong count.

one_sided (training reward soft1): every clause that the discrete judgement accepts scores
exactly 1, and only failing clauses get continuous partial credit. The two-sided version keeps paying for
satisfied clauses (confidence of found objects), which rewards raising detection confidence over fixing the
failing clauses.
"""
    thr = counting_threshold if metadata['tag'] == 'counting' else threshold
    clauses, groups = [], []
    includes = metadata.get('include', [])
    for req in includes:
        count = req['count']
        cand = candidates.get(req['class'], [])
        clauses.append(soft_count_clause([c[0] for c in cand], thr, count, one_sided=one_sided)[0])
        top = cand[:count]
        presence = _presence([c[0] for c in top], thr, one_sided=one_sided)
        if 'color' in req:
            p = np.asarray(colour_prob(req['class'], top, req['color']), dtype=np.float64) if top else np.zeros(0)
            clauses.append(float(np.sum(presence * p)) / count)
        if 'position' in req:
            relation, target = req['position']
            target_top, target_presence = groups[target]
            pairs = [pa * pb * soft_relation(a[1], b[1], relation, position_threshold, one_sided=one_sided)
                     for a, pa in zip(top, presence) for b, pb in zip(target_top, target_presence)]
            clauses.append(float(np.sum(pairs)) / (count * includes[target]['count']))
        groups.append((top, presence))
    return float(np.mean(clauses)) if clauses else 0.


def physical_gpu(gpu_id, visible=None):
    """The physical device a rank-local index names, honouring CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES') if visible is None else visible
    if visible:
        return visible.split(',')[gpu_id].strip()
    return str(gpu_id)


def _closure(fn):
    while hasattr(fn, '__wrapped__'):  # compute_geneval is wrapped by @torch.no_grad()
        fn = fn.__wrapped__
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))


def build_soft_compute(compute):
    """compute_geneval plus the soft training reward, sharing one detector pass (server process only).

reward_server builds its models inside load_geneval's closure, so the detector, the CLIP colour classifier and
the judging functions are taken from there: the discrete fields still come from the unchanged
evaluate_reward / evaluate. Used only for soft training requests; evaluation keeps calling compute_geneval.
"""
    import torch
    from PIL import ImageOps
    import reward_server.gen_eval as G
    ev = _closure(_closure(compute)['evaluate_image'])
    er = _closure(ev['evaluate_reward'])
    cc = _closure(er['color_classification'])
    detector, classnames = ev['object_detector'], ev['classnames']
    threshold, counting_threshold = ev['THRESHOLD'], ev['COUNTING_THRESHOLD']
    max_objects, nms = ev['MAX_OBJECTS'], ev['NMS_THRESHOLD']
    colors, classifiers = cc['COLORS'], cc['COLOR_CLASSIFIERS']
    clip_model, tokenizer, crops = cc['clip_model'], cc['tokenizer'], cc['ImageCrops']

    def colour_prob(image, classname, objects, colour, one_sided=False):
        if classname not in classifiers:  # same templates as color_classification, shared cache
            classifiers[classname] = G.zsc.zero_shot_classifier(
                clip_model, tokenizer, colors,
                [f"a photo of a {{c}} {classname}", f"a photo of a {{c}}-colored {classname}", "a photo of a {c} object"],
                G.DEVICE)
        loader = torch.utils.data.DataLoader(crops(image, [(box, mask) for _, box, mask in objects]),
                                             batch_size=16, num_workers=0)
        logits, _ = G.zsc.run_classification(clip_model, classifiers[classname], loader, G.DEVICE)
        probs = torch.softmax(logits.float(), 1)
        target = probs[:, colors.index(colour)]
        if one_sided:  # 1 when the discrete classifier (argmax) already says the target colour
            return (target / probs.max(1).values).numpy()
        return target.numpy()

    def soft_compute(images, metadatas, only_strict, one_sided=False):
        results = G.inference_detector(detector, [np.array(im) for im in images])
        scores, correct, strict, soft = [], [], [], []
        for result, image_pil, metadata in zip(results, images, metadatas):
            bbox = result[0] if isinstance(result, tuple) else result
            segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            image = ImageOps.exif_transpose(image_pil)
            thr = counting_threshold if metadata['tag'] == 'counting' else threshold
            detected, candidates = {}, {}
            for index, classname in enumerate(classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                near = ordering[bbox[index][ordering, 4] > thr - SOFT_DETECTION_FLOOR][:max_objects].tolist()
                if near:
                    candidates[classname] = [(float(bbox[index][i, 4]), bbox[index][i],
                                              None if segm is None else segm[index][i]) for i in near]
                # below: evaluate_image's own detection filter, verbatim
                ordering = ordering[bbox[index][ordering, 4] > thr]
                ordering = ordering[:max_objects].tolist()
                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                    ordering = [obj for obj in ordering
                                if nms == 1 or ev['compute_iou'](bbox[index][max_obj], bbox[index][obj]) < nms]
                if not detected[classname]:
                    del detected[classname]
            is_strict, score, _ = ev['evaluate_reward'](image, detected, metadata)
            is_correct = False if only_strict else ev['evaluate'](image, detected, metadata)[0]
            scores.append(score)
            strict.append(1. if is_strict else 0.)
            correct.append(1. if is_correct else 0.)
            soft.append(soft_image_score(candidates, metadata,
                                         lambda c, objs, colour: colour_prob(image, c, objs, colour, one_sided),
                                         threshold, counting_threshold, one_sided=one_sided))
        return scores, correct, strict, soft

    return soft_compute


def serve(address, batch):
    sys.path[:0] = [REWARD_SERVER_DIR, os.path.join(REWARD_SERVER_DIR, 'mmdetection')]
    import torch
    loader = torch.utils.data.DataLoader

    class SerialLoader(loader):
        def __init__(self, *args, **kwargs):
            kwargs['num_workers'] = 0
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = SerialLoader
    from PIL import Image
    from reward_server.gen_eval import load_geneval
    compute = load_geneval()
    soft_compute = None  # built on the first soft training request (build_soft_compute)
    warm = {'tag': 'single_object', 'include': [{'class': 'cat', 'count': 1}], 'prompt': 'a photo of a cat'}
    compute([Image.new('RGB', (512, 512))] * batch, [warm] * batch, False)

    listener = Listener(address, family='AF_UNIX', authkey=AUTHKEY)
    conn = listener.accept()
    conn.send(('ready', batch))
    inbox = queue.Queue()

    def reader():
        while True:
            try:
                inbox.put(conn.recv())
            except EOFError:
                inbox.put(None)
                return

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            message = inbox.get()
            if message is None:
                break
            rid, images, metadatas, only_strict, *rest = message
            want_soft = rest[0] if rest else False  # False, or the training mode name (True from older clients)
            one_sided = want_soft == 'soft1'
            try:
                result = dict(scores=[], correct=[], strict=[], tags=[])
                if want_soft:
                    result['soft'] = []
                    if soft_compute is None:
                        soft_compute = build_soft_compute(compute)
                for lo in range(0, len(images), batch):
                    pils = [Image.fromarray(im) for im in images[lo:lo + batch]]
                    metas = list(metadatas[lo:lo + batch])
                    if want_soft:
                        scores, correct, strict, soft = soft_compute(pils, metas, only_strict, one_sided=one_sided)
                        result['soft'] += [float(v) for v in soft]
                    else:
                        scores, correct, strict, _, _ = compute(pils, metas, only_strict)
                    result['scores'] += [float(v) for v in scores]
                    result['correct'] += [float(v) for v in correct]
                    result['strict'] += [float(v) for v in strict]
                    result['tags'] += [m['tag'] for m in metas]
                conn.send((rid, result))
            except Exception as e:
                import traceback
                conn.send((rid, RuntimeError(traceback.format_exc())))
    finally:
        conn.close()
        listener.close()


class GenevalPool:
    """Client of one GenEval server. Same surface as stepwise_ocr.OcrPool."""

    backend = 'gpu'

    def __init__(self, metadata, gpu_id=0, batch=4, python=REWARD_SERVER_PYTHON, startup_timeout=900,
                 train_reward='score'):
        if train_reward not in TRAIN_REWARDS:
            raise ValueError(f'unknown GenEval training reward {train_reward!r}')
        self.train_reward = train_reward
        self.metadata = metadata
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.address = f'/tmp/stepwise_geneval_{os.getpid()}_{gpu_id}_{uuid.uuid4().hex[:8]}.sock'
        env = dict(os.environ, PYTHONPATH=repo, PYTHONUNBUFFERED='1',
                   CUDA_VISIBLE_DEVICES=physical_gpu(gpu_id), OMP_NUM_THREADS='4')
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
            env.setdefault(key, '1')
        self.process = subprocess.Popen(
            [python, '-m', 'rvge.stepwise_geneval', '--address', self.address, '--batch', str(batch)],
            cwd=REWARD_SERVER_DIR, env=env)
        deadline = time.monotonic() + startup_timeout
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f'GenEval server exited with {self.process.returncode} during startup')
            try:
                self.conn = Client(self.address, family='AF_UNIX', authkey=AUTHKEY)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise RuntimeError('GenEval server did not start in time')
                time.sleep(.5)
        status = self.conn.recv()
        if status[0] != 'ready':
            raise RuntimeError(f'GenEval server failed to start: {status}')
        self.failures = 0  # gen_eval.py does not swallow exceptions; a failure raises instead
        self._next = 0
        self._done = {}

    def submit(self, images, prompts, evaluation=False):
        """Training asks only for the strict path, as DiffusionNFT's loop does: the reward
`score` is identical and the benchmark `correct` judgement is skipped."""
        if len(images) != len(prompts):
            raise ValueError('one prompt per image is required')
        missing = [p for p in prompts if p not in self.metadata]
        if missing:
            raise KeyError(f'no GenEval metadata for prompt {missing[0]!r}')
        rid = self._next
        self._next += 1
        # The fifth field asks for the soft training reward; evaluation requests never do, so evaluation
        # records always come from the discrete compute_geneval path.
        self.conn.send((rid, np.ascontiguousarray(images), [self.metadata[p] for p in prompts],
                        not evaluation,
                        self.train_reward if (not evaluation and self.train_reward in SOFT_REWARDS) else False))
        return rid

    def submit_metadata(self, images, metadatas, evaluation=False):
        """As `submit`, with the metadata given per image - DiffusionNFT's reward interface."""
        if len(images) != len(metadatas):
            raise ValueError('one metadata record per image is required')
        rid = self._next
        self._next += 1
        self.conn.send((rid, np.ascontiguousarray(images), list(metadatas), not evaluation))
        return rid

    def _take(self, rid):
        while rid not in self._done:
            got, value = self.conn.recv()
            self._done[got] = value
        value = self._done.pop(rid)
        if isinstance(value, Exception):
            raise RuntimeError('GenEval server failed on a request') from value
        return value

    def result(self, rid):
        """Training values (see train_values); evaluation reads result_details, which keeps `score`."""
        return train_values(self._take(rid), self.train_reward)

    def result_details(self, rid):
        value = self._take(rid)
        rows = [dict(score=s, correct=c, strict=t, tag=g)
                for s, c, t, g in zip(value['scores'], value['correct'], value['strict'], value['tags'])]
        return value['scores'], rows

    def score(self, images, prompts):
        return self.result(self.submit(images, prompts))

    def close(self):
        try:
            self.conn.send(None)
            self.conn.close()
        except Exception:
            pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if os.path.exists(self.address):
            os.unlink(self.address)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--address', required=True)
    p.add_argument('--batch', type=int, default=8)
    a = p.parse_args()
    serve(a.address, a.batch)


if __name__ == '__main__':
    main()
