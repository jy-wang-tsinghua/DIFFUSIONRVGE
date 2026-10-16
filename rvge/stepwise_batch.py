"""Main-trajectory batching for one fresh, synchronized optimizer update."""
from dataclasses import dataclass

import json
import numpy as np


@dataclass(frozen=True)
class UpdateBatch:
    group_size: int
    groups_per_update: int
    world_size: int
    main_steps: int = 10
    batching: str = 'prompt'

    def __post_init__(self):
        if min(self.group_size, self.groups_per_update, self.world_size, self.main_steps) <= 0:
            raise ValueError('batch sizes, world size, and main steps must be positive')
        if self.batching not in ('prompt', 'rows'):
            raise ValueError('batching must be prompt or rows')
        # Prompt-major batches hold one prompt, so every rank must be able to take an
        # equal slice out of each group. Row-major batches mix prompts, so only the
        # flat trajectory list has to divide, which is the weaker requirement.
        if self.batching == 'prompt':
            if self.group_size % self.world_size:
                raise ValueError('group size must divide equally across ranks')
        elif self.trajectories_per_update % self.world_size:
            raise ValueError('trajectories per update must divide equally across ranks')

    @property
    def local_group_size(self):
        return self.group_size // self.world_size

    @property
    def trajectories_per_update(self):
        return self.group_size * self.groups_per_update

    @property
    def local_trajectories_per_update(self):
        return self.trajectories_per_update // self.world_size

    @property
    def microbatch_limit(self):
        """Largest microbatch one backward may hold: a group slice, or the shard."""
        return self.local_group_size if self.batching == 'prompt' else self.local_trajectories_per_update

    @property
    def local_rows_per_update(self):
        return self.local_trajectories_per_update * self.main_steps

    @property
    def main_rows_per_update(self):
        return self.trajectories_per_update * self.main_steps

    def loss_weight(self, microbatch_rows):
        """Local mean; the subsequent rank average gives the global row mean."""
        if not 0 < microbatch_rows <= self.microbatch_limit:
            raise ValueError('microbatch rows must fit within this rank\'s share of one update')
        return microbatch_rows / self.local_rows_per_update


def select_prompt_ids(seed, step, num_prompts, groups_per_update, unique=False):
    if num_prompts <= 0 or groups_per_update <= 0:
        raise ValueError('prompt counts must be positive')
    rng = np.random.default_rng(seed * 100000 + step)
    if unique:
        if groups_per_update > num_prompts:
            raise ValueError('unique prompt groups exceed the available prompt panel')
        return rng.choice(num_prompts, groups_per_update, replace=False).tolist()
    # The default draw (with replacement); unchanged so a given seed keeps its prompt sequence.
    return rng.integers(0, num_prompts, groups_per_update).tolist()


def checkpoint_training_totals(checkpoint):
    """Keep historical counts when a resumed run changes its update batch."""
    if 'training_totals' in checkpoint:
        return dict(checkpoint['training_totals'])
    # Legacy checkpoints were saved with one fixed batch size from step zero.
    config = checkpoint['config']
    return dict(
        main_trajectories=checkpoint['step'] * config['trajectories_per_update'],
        main_training_rows=checkpoint['step'] * config['main_rows_per_update'],
    )


def select_prompt_ids_weighted(seed, step, weights, groups_per_update):
    """Draw prompt rows with probability proportional to `weights` (residual-error curriculum).

Same rng seeding as `select_prompt_ids` so every rank draws the same ids; without replacement, so an update never
spends two groups on one row. `weights` is a 1-D numpy array over the prompt rows (any non-negative scale)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) <= 0 or groups_per_update <= 0 or groups_per_update > len(w):
        raise ValueError('bad prompt weights or group count')
    w = np.clip(w, 0., None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    rng = np.random.default_rng(seed * 100000 + step)
    return rng.choice(len(w), groups_per_update, replace=False, p=w / w.sum()).tolist()


class PromptResidualTable:
    """Per-prompt-row running success estimate that turns the draw toward the prompts still failing.

Every update's groups report their mean reward per row; the table keeps an EMA per row (decay `decay`) starting from
`prior` for rows never seen. The sampling weight of a row is `floor + (1 - success)`: solved rows keep the floor so they
are still revisited (and their table entry can decay back if they regress), failing rows get up to 1 + floor. Rows that
share a prompt text share the entry, so duplicated rows (the dataset's task mixture) are weighted as one prompt but
drawn as many rows. Saved to / loaded from JSON so a resumed run keeps its curriculum.

`mode='mixed'`: the quantity tracked is whether a group carried reward contrast (0 < success < 1, the only groups that yield a
direction), and an unseen prompt falls back to the running mixed rate of its `tag` (per-tag EMA, decay `tag_decay`)
instead of `prior`. A seen prompt starts its own EMA from the tag rate. weight = floor + P(mixed)."""

    def __init__(self, prompts, prior=0.5, decay=0.7, floor=0.1, success_cap=0.99, tags=None, mode='success', tag_decay=0.95):
        self.prompts = list(prompts)
        self.index = {}
        for i, p in enumerate(self.prompts):
            self.index.setdefault(p, []).append(i)
        self.prior, self.decay, self.floor, self.cap = float(prior), float(decay), float(floor), float(success_cap)
        self.mode, self.tag_decay = mode, float(tag_decay)
        if mode not in ('success', 'mixed'):
            raise ValueError(f'unknown residual mode {mode}')
        if mode == 'mixed' and tags is None:
            raise ValueError("mode 'mixed' needs one tag per prompt row")
        self.tags = list(tags) if tags is not None else None
        self.tag_of = {p: t for p, t in zip(self.prompts, self.tags)} if self.tags is not None else {}
        self.success = {}            # prompt text -> EMA of the group success rate (share of endpoints >= cap)
        self.mixed = {}              # prompt text -> EMA of the mixed indicator (mode 'mixed')
        self.tag_mixed = {}          # tag -> EMA of the mixed indicator over that tag's groups
        self.seen = {}

    def _tag_rate(self, p):
        return self.tag_mixed.get(self.tag_of.get(p), self.prior)

    def update(self, row_ids, group_success_rates):
        for rid, rate in zip(row_ids, group_success_rates):
            p = self.prompts[rid]
            old = self.success.get(p, self.prior)
            self.success[p] = self.decay * old + (1. - self.decay) * float(rate)
            if self.mode == 'mixed':
                m = 1. if 0. < float(rate) < 1. else 0.
                t = self.tag_of[p]
                self.tag_mixed[t] = self.tag_decay * self.tag_mixed.get(t, self.prior) + (1. - self.tag_decay) * m
                self.mixed[p] = self.decay * self.mixed.get(p, self._tag_rate(p)) + (1. - self.decay) * m
            self.seen[p] = self.seen.get(p, 0) + 1

    def weights(self):
        w = np.empty(len(self.prompts), dtype=np.float64)
        for i, p in enumerate(self.prompts):
            if self.mode == 'mixed':
                w[i] = self.floor + self.mixed.get(p, self._tag_rate(p))
            else:
                w[i] = self.floor + (1. - self.success.get(p, self.prior))
        return w

    def summary(self):
        vals = np.array(list(self.success.values())) if self.success else np.array([])
        out = dict(prompts_seen=len(self.success), mean_success_seen=(float(vals.mean()) if len(vals) else None),
                   share_solved_seen=(float((vals >= self.cap).mean()) if len(vals) else None))
        if self.mode == 'mixed':
            out['tag_mixed'] = {t: round(v, 3) for t, v in sorted(self.tag_mixed.items())}
            w = self.weights()
            if self.tags is not None:
                tot = w.sum()
                out['tag_draw_share'] = {t: round(float(w[[i for i, tt in enumerate(self.tags) if tt == t]].sum() / tot), 3)
                                         for t in sorted(set(self.tags))}
        return out

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(dict(prior=self.prior, decay=self.decay, floor=self.floor, cap=self.cap, mode=self.mode,
                           tag_decay=self.tag_decay, success=self.success, mixed=self.mixed, tag_mixed=self.tag_mixed,
                           seen=self.seen), f)

    def load(self, path):
        d = json.load(open(path))
        self.success = {k: float(v) for k, v in d.get('success', {}).items()}
        self.mixed = {k: float(v) for k, v in d.get('mixed', {}).items()}
        self.tag_mixed = {k: float(v) for k, v in d.get('tag_mixed', {}).items()}
        self.seen = {k: int(v) for k, v in d.get('seen', {}).items()}
        return self


def source_allocation(shares, groups_per_update):
    """Integer group counts per source for one update: largest-remainder rounding of share * groups (sum exact)."""
    items = [(k, float(v)) for k, v in shares.items() if float(v) > 0]
    if not items:
        raise ValueError('no source with a positive share')
    total = sum(v for _, v in items)
    raw = [(k, groups_per_update * v / total) for k, v in items]
    counts = {k: int(np.floor(x)) for k, x in raw}
    for k, _ in sorted(raw, key=lambda kx: kx[1] - np.floor(kx[1]), reverse=True)[:groups_per_update - sum(counts.values())]:
        counts[k] += 1
    return counts


def select_prompt_ids_by_source(seed, step, sources, shares, groups_per_update, row_weights=None):
    """Source-stratified prompt draw for the multi reward.

`sources` is one source name per prompt row; `shares` maps source -> share of the update's groups (normalised
here); the count per source is fixed by `source_allocation`, so the mix does not fluctuate from update to update.
Within a source the rows are drawn without replacement with probability proportional to `row_weights` (uniform
when None; the GenEval tag shares live there). The drawn groups are then shuffled, so the block shard hands every
rank a mix of sources rather than 6 GenEval groups to one rank (the detector is 6x slower than the other
scorers). Same rng seeding as `select_prompt_ids`: every rank draws the same ids."""
    sources = np.asarray(sources)
    counts = source_allocation(shares, groups_per_update)
    w = np.ones(len(sources), dtype=np.float64) if row_weights is None else np.clip(np.asarray(row_weights, dtype=np.float64), 0., None)
    rng = np.random.default_rng(seed * 100000 + step)
    picked = []
    for name in sorted(counts):
        n = counts[name]
        if n == 0:
            continue
        rows = np.flatnonzero(sources == name)
        if len(rows) < n:
            raise ValueError(f'source {name!r} has {len(rows)} rows for {n} groups')
        pw = w[rows]
        if pw.sum() <= 0:
            pw = np.ones_like(pw)
        picked.extend(rng.choice(rows, n, replace=False, p=pw / pw.sum()).tolist())
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]
