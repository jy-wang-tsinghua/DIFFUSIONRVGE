"""PaddleOCR reward for the stepwise trainer, served out of process.

THE REWARD (`ocr_reward` below) is the OCR reward Flow-GRPO and DiffusionNFT train on and report:
1 - edit_distance(recognized, target) / len(target), target = the quoted span of the
prompt, both lower-cased with spaces removed, distance capped at len(target).
Images are uint8 NHWC arrays, as in their OCR reward.

WHY A SERVER. The trainer runs under torchrun, so a spawned multiprocessing child
re-imports the training script - torch, diffusers, the PickScore module - before it
can run a single OCR call. This module imports numpy and nothing else at top level,
and runs as `python -m rvge.stepwise_ocr`, so its workers are light.

WHY CPU THREADS ARE PINNED TO 1. Every PaddleOCR process otherwise starts one
OpenCV thread per logical core, which with many workers oversubscribes the machine.
Even then the DB text detector at 512x512 is memory bound on CPU, so throughput
stops scaling after a few workers, which is why `backend='gpu'` exists: the same
model on the rank's own GPU.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.connection import Client, Listener

import numpy as np

_OCR = None
AUTHKEY = b'stepwise-ocr'


def target_text(prompt):
    """The span OcrScorer compares against; a prompt without one is a data error."""
    parts = prompt.split('"')
    if len(parts) < 3:
        raise ValueError(f'OCR prompt has no quoted target text: {prompt!r}')
    return parts[1]


def ocr_reward(ocr, img, prompt):
    """OCR reward of one image: 1 - d / n, where n is the length of the quoted target text and d its edit distance
to the recognized text (both without spaces, lower case), with d = 0 when the recognized text contains the target
and d capped at n. Words with non-positive confidence are ignored; an OCR failure scores 0. This is the OCR reward
definition of Flow-GRPO and DiffusionNFT."""
    from Levenshtein import distance
    target = prompt.split('"')[1].replace(' ', '').lower()
    try:
        lines = ocr.ocr(img, cls=False)[0] or []
        text = ''.join(line[1][0] for line in lines if line[1][1] > 0).replace(' ', '').lower()
    except Exception as e:
        print(f'OCR processing failed: {e}')
        return 0.0
    d = 0 if target in text else min(distance(text, target), len(target))
    return 1 - d / len(target)


def _init_worker(backend, gpu_id, threads, gpu_mem):
    global _OCR
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(threads)
    import cv2
    cv2.setNumThreads(1)
    from paddleocr import PaddleOCR
    kwargs = dict(use_angle_cls=False, lang='en', show_log=False, cpu_threads=threads)
    if backend == 'gpu':
        kwargs.update(use_gpu=True, gpu_id=gpu_id, gpu_mem=gpu_mem)
    else:
        kwargs.update(use_gpu=False)
    _OCR = PaddleOCR(**kwargs)


class _Recorded:
    """Pass-through to PaddleOCR that remembers the exception OcrScorer is about to swallow."""

    def __init__(self, inner):
        self.inner = inner
        self.error = None

    def ocr(self, img, cls=False):
        try:
            return self.inner.ocr(img, cls=cls)
        except Exception as e:
            self.error = e
            raise


def _is_memory_error(error):
    text = f'{type(error).__name__} {error}'.lower()
    return 'resourceexhausted' in text or 'out of memory' in text


def _score_chunk(chunk):
    """Upstream semantics for OCR failures (reward 0), but never for running out of memory.

When PyTorch's cache leaves a GPU worker too little memory, the detector raises
ResourceExhaustedError, and OcrScorer's blanket except would turn it into a silent
reward of 0. That is an infrastructure fault, not an unreadable image, so it is
retried after freeing Paddle's cache and raised if it persists.
"""
    images, prompts = chunk
    values, failures = [], 0
    for img, prompt in zip(images, prompts):
        for attempt in range(4):
            recorded = _Recorded(_OCR)
            value = ocr_reward(recorded, img, prompt)
            if recorded.error is None or not _is_memory_error(recorded.error):
                break
            try:
                import paddle
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(1 + attempt)
        else:
            raise RuntimeError(f'PaddleOCR out of memory on 4 attempts: {recorded.error}')
        failures += recorded.error is not None
        values.append(float(value))
    return values, failures


def _ready(_):
    return _OCR is not None


def serve(address, workers, backend, gpu_id, threads, gpu_mem, chunk):
    """Accept one client. A reader submits every request at once; a writer answers in order.

Submitting on arrival is what lets the client hand over the next batch while the
previous one is still being read, so evaluation keeps generating during OCR.
"""
    import multiprocessing as mp
    import queue
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                               initializer=_init_worker, initargs=(backend, gpu_id, threads, gpu_mem))
    assert all(pool.map(_ready, range(workers * 2)))
    listener = Listener(address, family='AF_UNIX', authkey=AUTHKEY)
    conn = listener.accept()
    conn.send(('ready', workers, backend))
    pending = queue.Queue()

    def writer():
        while True:
            item = pending.get()
            if item is None:
                return
            rid, futures = item
            try:
                parts = [f.result() for f in futures]
                conn.send((rid, ([v for values, _ in parts for v in values], sum(n for _, n in parts))))
            except Exception as e:  # report instead of hanging the trainer
                conn.send((rid, e))

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        while True:
            try:
                message = conn.recv()
            except EOFError:
                break
            if message is None:
                break
            rid, images, prompts = message
            futures = [pool.submit(_score_chunk, (images[lo:lo + chunk], prompts[lo:lo + chunk]))
                       for lo in range(0, len(images), chunk)]
            pending.put((rid, futures))
    finally:
        pending.put(None)
        thread.join(timeout=60)
        pool.shutdown(wait=False, cancel_futures=True)
        conn.close()
        listener.close()


def to_uint8_nhwc(images):
    """[0, 1] float NCHW tensor -> uint8 NHWC numpy, the rewards.py:ocr_score conversion."""
    import torch
    return ((images.float() * 255).round().clamp(0, 255).to(torch.uint8)
            .permute(0, 2, 3, 1).cpu().numpy())


# Python interpreter with paddlepaddle-gpu + paddleocr for --ocr-backend gpu (the CPU backend runs in-process).
GPU_PYTHON = os.environ.get('RVGE_OCR_PYTHON', sys.executable)


def _gpu_environment(python):
    """cuDNN 8 from the paddle venv, the other CUDA 12 libraries from the base env's wheels."""
    venv = os.path.dirname(os.path.dirname(python))
    base = sys.base_prefix if hasattr(sys, 'base_prefix') else sys.prefix
    dirs = []
    for root in (os.path.join(venv, 'lib'), os.path.join(base, 'lib')):
        if not os.path.isdir(root):
            continue
        for py in sorted(os.listdir(root)):
            nvidia = os.path.join(root, py, 'site-packages', 'nvidia')
            if os.path.isdir(nvidia):
                dirs += [os.path.join(nvidia, d, 'lib') for d in sorted(os.listdir(nvidia))
                         if os.path.isdir(os.path.join(nvidia, d, 'lib'))]
    # Paddle dlopens the UNVERSIONED names (libcudnn.so, libcublas.so, ...), which the
    # nvidia wheels do not ship; <venv>/lib_links holds those symlinks.
    links = os.path.join(venv, 'lib_links')
    return os.pathsep.join(([links] if os.path.isdir(links) else []) + dirs
                           + [os.environ.get('LD_LIBRARY_PATH', '')])


class OcrPool:
    """Client of one OCR server process. `submit` returns a handle, `result` blocks on it."""

    def __init__(self, workers=16, backend='cpu', gpu_id=0, threads=1, gpu_mem=3000, chunk=None,
                 python=None, startup_timeout=600):
        if backend not in ('cpu', 'gpu'):
            raise ValueError(f'unknown OCR backend {backend!r}')
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        python = python or (GPU_PYTHON if backend == 'gpu' else sys.executable)
        self.address = f'/tmp/stepwise_ocr_{os.getpid()}_{gpu_id}_{uuid.uuid4().hex[:8]}.sock'
        env = dict(os.environ, PYTHONPATH=repo + os.pathsep + os.environ.get('PYTHONPATH', ''),
                   OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS=str(threads), PYTHONUNBUFFERED='1')
        if backend == 'gpu':
            # CUDA_VISIBLE_DEVICES is inherited, so gpu_id indexes the caller's visible set.
            env['LD_LIBRARY_PATH'] = _gpu_environment(python)
            # naive_best_fit makes gpu_mem an up-front reservation. The pool starts before
            # the trainer loads its model, so Paddle holds its memory before PyTorch's
            # caching allocator can take the card.
            env['FLAGS_allocator_strategy'] = 'naive_best_fit'
        chunk = chunk or (8 if backend == 'gpu' else 2)
        self.process = subprocess.Popen(
            [python, '-m', 'rvge.stepwise_ocr', '--address', self.address, '--workers', str(workers),
             '--backend', backend, '--gpu-id', str(gpu_id), '--threads', str(threads),
             '--gpu-mem', str(gpu_mem), '--chunk', str(chunk)], cwd=repo, env=env)
        deadline = time.monotonic() + startup_timeout
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f'OCR server exited with {self.process.returncode} during startup')
            try:
                self.conn = Client(self.address, family='AF_UNIX', authkey=AUTHKEY)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise RuntimeError('OCR server did not start in time')
                time.sleep(.5)
        status = self.conn.recv()
        if status[0] != 'ready':
            raise RuntimeError(f'OCR server failed to start: {status}')
        self.workers, self.backend = status[1], status[2]
        self.failures = 0  # upstream-style OCR exceptions scored as 0, excluding memory faults
        self._next = 0
        self._done = {}

    def submit(self, images, prompts, evaluation=False):
        if len(images) != len(prompts):
            raise ValueError('one prompt per image is required')
        for p in prompts:
            target_text(p)
        rid = self._next
        self._next += 1
        self.conn.send((rid, np.ascontiguousarray(images), list(prompts)))
        return rid

    def result(self, rid):
        while rid not in self._done:
            got, value = self.conn.recv()
            self._done[got] = value
        value = self._done.pop(rid)
        if isinstance(value, Exception):
            raise RuntimeError('OCR server failed on a request') from value
        values, failures = value
        self.failures += failures
        return values

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
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--gpu-mem', type=int, default=500)
    p.add_argument('--chunk', type=int, default=2)
    a = p.parse_args()
    serve(a.address, a.workers, a.backend, a.gpu_id, a.threads, a.gpu_mem, a.chunk)


if __name__ == '__main__':
    main()
