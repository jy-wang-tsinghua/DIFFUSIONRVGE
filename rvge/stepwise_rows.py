"""Row-major sampling: batches may mix prompts, so batch size is a free choice.

Nothing in the endpoint estimator or the actor loss couples trajectories to each
other. The per-prompt grouping exists only to share an initial noise root, so it
constrains what a batch may contain in the prompt-major implementation, and
through that it ties the trajectory count, the prompt count, and the batch size
to one another. Sharding a flat list of rows instead unties all three.

A row is one trajectory: (group index, prompt id, index within its group). Its
noise depends on those three fields alone, so which rank holds it, and how rows
are batched, do not change what is sampled.
"""
import time

import torch

from rvge.stepwise_credit import (flow_coefficients, ddim_coefficients, sde_coefficients,
                                       sde_schedule_of, uniform_value_nodes)


def build_rows(pids, group_size):
    return [(g, pid, j) for g, pid in enumerate(pids) for j in range(group_size)]


def shard_rows(rows, world, rank, mode='interleave'):
    """Interleave, so every rank holds a mixture of prompts rather than a block.

`mode='block'` gives each rank a contiguous run of rows instead, so a whole prompt group
sits on one rank when the per-rank count is a multiple of the group size - what a
group-pooled fit needs. The samples are unchanged: a row's noise depends only on its
(group, prompt, index) fields, not on the rank that runs it.
"""
    if len(rows) % world:
        raise ValueError('total trajectories per update must divide across ranks')
    if mode == 'block':
        per = len(rows) // world
        return rows[rank * per:(rank + 1) * per]
    if mode != 'interleave':
        raise ValueError(f'unknown shard mode {mode!r}')
    return rows[rank::world]


def gathered_order(rows, world, mode='interleave'):
    """The row order an all_gather produces: every rank's shard, rank by rank.

A prompt group is spread across all ranks here, so a statistic taken over the
whole group has to index the gathered tensor, not the local shard.
"""
    return [r for w in range(world) for r in shard_rows(rows, world, w, mode)]


def group_positions(rows, group):
    return [i for i, (g, _, _) in enumerate(rows) if g == group]


def row_key(seed, step, group):
    return seed * 10**12 + step * 10**7 + group * 10**4


def root_seed(seed, step, group, index, independent=False):
    """Initial-noise seed of one trajectory.

Historically a group is 12 trajectories off ONE initial noise: `row_key` carries
no trajectory index, so every member of a group starts from the same x0 and they
differ only through the injected noise. That sharing is what makes the group a
matched-baseline set, and the shipped per-trajectory estimator never uses it.

`independent=True` gives each trajectory its own root. The index occupies the
group slot's low digits, which is 10**4 wide, so a group of any realistic size
cannot reach the next group's seeds. index 0 coincides with the shared seed by
construction, which is harmless: a run is one mode or the other.
"""
    key = 12_000_000_000 + row_key(seed, step, group)
    return key + (index if independent else 0)


@torch.no_grad()
def sample_rows(rt, rows, prompts, args, step, batch):
    """Sample this rank's rows, batching across prompts at width `batch`.

`batch` must equal the training microbatch: identical shapes make the replayed
velocity bitwise identical, which is what the fresh-on-policy guard checks.
"""
    texts = [prompts[pid] for _, pid, _ in rows]
    schedule, noise_exponent = sde_schedule_of(args)
    n = len(rows)
    f0, s0, n0 = rt.forward_samples, rt.score_samples, rt.nfe
    independent = getattr(args, 'independent_roots', False)
    x = torch.cat([rt.noise(root_seed(args.seed, step, g, j, independent)) for g, _, j in rows])
    xs, vs, children, firsts, mus, bs, taus = [], [], [], [], [], [], []
    realized = []
    for q, (t, s) in enumerate(zip(rt.times, rt.times[1:])):
        v = torch.cat([rt.velocity_rows(x[lo:lo + batch], t, texts[lo:lo + batch])
                       for lo in range(0, n, batch)])
        xs.append(x.cpu()); vs.append(v.cpu())
        firsts.append(x + (rt.times[q + 1] - t) * v)
        if args.sde_noise_mode == 'state_rms':
            rms = (x + (s - t) * v).double().flatten(1).square().mean(1).sqrt().float()
            unit = flow_coefficients(t, s, 1., rt.times[1])[2]
            level = (args.noise_radius * rms / unit).view(-1, 1, 1, 1)
        else:
            level = torch.full((n, 1, 1, 1), args.noise_radius, device=rt.device, dtype=torch.float32)
        a, b, tau = sde_coefficients(t, s, level, rt.times[1], schedule, noise_exponent)
        eps = torch.cat([rt.noise(21_000_000_000 + row_key(args.seed, step, g) + q * 97 + j * 1009)
                         for g, _, j in rows])
        mu = a * x + b * v
        child = mu + tau * eps
        realized.extend((tau * eps).double().flatten(1).square().mean(1).sqrt().cpu().tolist())
        children.append(child.cpu()); mus.append(mu.cpu())
        bs.append(torch.as_tensor(b, device=rt.device, dtype=torch.float32).expand_as(x[:, :1, :1, :1]).cpu())
        taus.append(torch.as_tensor(tau, device=rt.device, dtype=torch.float32).expand_as(x[:, :1, :1, :1]).cpu())
        x = child
    live = _suffixes(rt, texts, firsts, batch, args.rollout_chunk)
    ends = torch.cat(live + [x])
    end_texts = texts * (len(rt.times) - 1) + texts
    torch.cuda.synchronize(); tic = time.monotonic()
    scores = rt.score_rows(ends, end_texts, args.score_chunk).cpu()
    torch.cuda.synchronize(); score_seconds = time.monotonic() - tic
    ys = [y.cpu() for y in live] + [x.cpu()]
    rewards = torch.stack(list(scores.split(n)), 1)
    assert rt.forward_samples - f0 == 55 * n, 'forward sample count changed'
    assert rt.score_samples - s0 == 11 * n, 'score sample count changed'
    bank = dict(x=torch.stack(xs, 1), old=torch.stack(vs, 1), child=torch.stack(children, 1),
                mean=torch.stack(mus, 1), b=torch.stack(bs, 1), tau=torch.stack(taus, 1),
                y=torch.stack(ys, 1), r=rewards)
    mix = getattr(rt, 'last_mix', None)
    if mix is not None and 'source' in mix and len(mix['task']) == len(ends):
        # Multi reward under the SDE sampler (for --quality-mode task_floor). Unlike the ODE sampler, which
        # scores ONE endpoint per row and whose columns are [n, 1], this one scores all 11 states per row, so every
        # component is [n, states] exactly like bank['r'] - the estimator reads adjacent differences of bank['r'], so a
        # recompose that only knew the first state would silently flatten the last 10 gains to whatever that state said.
        # rt.last_mix is ordered like `ends` = the 10 suffixes then the SDE endpoint, i.e. scores.split(n) stacked on 1.
        src = mix['source'].long().cpu()
        assert (src.view(-1, n) == src[:n]).all(), 'multi source ids must repeat per scored state'
        bank['src'] = src[:n]
        for key, col in (('task', 'r_task'), ('pickscore', 'q_pick'), ('clipscore', 'q_clip'),
                         ('hpsv2', 'q_hps'), ('q', 'q_all')):
            bank[col] = torch.stack(list(mix[key].float().cpu().split(n)), 1)
    return bank, dict(forward_calls=rt.nfe - n0, forward_samples=rt.forward_samples - f0,
                      score_samples=rt.score_samples - s0, score_seconds=score_seconds,
                      innovation_rms_mean=float(sum(realized) / len(realized)))


@torch.no_grad()
def _suffixes(rt, texts, firsts, batch, chunk):
    """Advance every stopped ODE suffix; suffixes sharing a node go together."""
    live = list(firsts)
    for k in range(1, len(rt.times) - 1):
        nodes = torch.cat(live[:k])
        node_texts = texts * k
        width = chunk or batch
        advance = torch.cat([rt.velocity_rows(nodes[lo:lo + width], rt.times[k], node_texts[lo:lo + width])
                             for lo in range(0, len(nodes), width)])
        pieces = advance.split(len(firsts[0]))
        delta = rt.times[k + 1] - rt.times[k]
        for i in range(k):
            live[i] = live[i] + delta * pieces[i]
    return live


@torch.no_grad()
def sample_rows_paper(rt, rows, prompts, args, step, batch):
    """Row-major sampling for `--method paper`: the T' substeps batch across prompts.

`uniform_value_nodes(t_q, T')` depends only on q, so at every (q, substep) EVERY
trajectory of the update sits on the same time node and they go through the
transformer together. Prompt-major walks one group at a time instead, which on
eight ranks leaves group_size/world rows per launch, and narrow launches rather
than compute then set the rollout time. Here the main steps go at width `batch`
and the substeps at `rollout_chunk`.

THE SAMPLED SET IS UNCHANGED. Both the initial noise and the per-step innovation
are functions of (seed, step, group, index-within-group) only - `root_seed` and
`row_key` here are the same ones `sample_group` uses - so the union over ranks is
the same set of trajectories. What moves is which rank holds which trajectory,
and the batch composition of every transformer call, and with it the BF16
reduction order. That is the same class of difference as changing the world size.

The MAIN-step velocity is taken at width `batch`, not `rollout_chunk`: the
training loop replays it at `--train-microbatch`, and the fresh-on-policy guard
compares the replay bitwise. The substeps are never replayed, so they take the
wider `--rollout-chunk`.
"""
    if args.method != 'paper':
        raise ValueError('sample_rows_paper is for --method paper')
    texts = [prompts[pid] for _, pid, _ in rows]
    n = len(rows)
    f0, s0, n0 = rt.forward_samples, rt.score_samples, rt.nfe
    independent = getattr(args, 'independent_roots', False)
    x = torch.cat([rt.noise(root_seed(args.seed, step, g, j, independent)) for g, _, j in rows])
    xs, vs, children, ys, rs, mus, bs, taus = [], [], [], [], [], [], [], []
    realized = []
    width = args.rollout_chunk or batch
    score_seconds = 0.

    def field(state, node, w):
        return torch.cat([rt.velocity_rows(state[lo:lo + w], node, texts[lo:lo + w])
                          for lo in range(0, n, w)])

    def score(state):
        nonlocal score_seconds
        torch.cuda.synchronize(); tic = time.monotonic()
        out = rt.score_rows(state, texts, args.score_chunk).cpu()
        torch.cuda.synchronize(); score_seconds += time.monotonic() - tic
        return out

    for q, (t, s) in enumerate(zip(rt.times, rt.times[1:])):
        v = field(x, t, batch)
        xs.append(x.cpu()); vs.append(v.cpu())
        # Eq. (7) of the paper: x_0(t) by forward Euler over T' uniform substeps.
        nodes = uniform_value_nodes(t, args.value_substeps)
        y = x + (nodes[1] - nodes[0]) * v
        for u, w in zip(nodes[1:-1], nodes[2:]):
            y = y + (w - u) * field(y, u, width)
        ys.append(y.cpu()); rs.append(score(y))
        a, b, tau = (flow_coefficients(t, s, args.paper_noise, rt.times[1]) if args.paper_sampler == 'flow'
                     else ddim_coefficients(t, s, args.ddim_eta))
        eps = torch.cat([rt.noise(21_000_000_000 + row_key(args.seed, step, g) + q * 97 + j * 1009)
                         for g, _, j in rows])
        mu = a * x + b * v
        child = mu + tau * eps
        realized.extend((tau * eps).double().flatten(1).square().mean(1).sqrt().cpu().tolist())
        children.append(child.cpu()); mus.append(mu.cpu())
        one = x[:, :1, :1, :1]
        bs.append(torch.full_like(one, float(b)).cpu())
        taus.append(torch.full_like(one, float(tau)).cpu())
        x = child
    ys.append(x.cpu()); rs.append(score(x))

    expected = (len(rt.times) - 1) * args.value_substeps
    assert rt.forward_samples - f0 == expected * n, 'forward sample count changed'
    assert rt.score_samples - s0 == len(rt.times) * n, 'score sample count changed'
    bank = dict(x=torch.stack(xs, 1), old=torch.stack(vs, 1), child=torch.stack(children, 1),
                mean=torch.stack(mus, 1), b=torch.stack(bs, 1), tau=torch.stack(taus, 1),
                y=torch.stack(ys, 1), r=torch.stack(rs, 1))
    return bank, dict(forward_calls=rt.nfe - n0, forward_samples=rt.forward_samples - f0,
                      score_samples=rt.score_samples - s0, score_seconds=score_seconds,
                      innovation_rms_mean=float(sum(realized) / len(realized)))


@torch.no_grad()
def sample_rows_ode(rt, rows, prompts, args, step, batch):
    """Independent ODE trajectories, one endpoint and one reward each.

Every row is a plain 10-step ODE sample from its own initial noise (`root_seed` with independent roots): T main
states and velocities are recorded for the actor rows, the endpoint is scored once. No SDE innovation, no stopped
branches: 10 forward passes and 1 score per trajectory instead of 55 and 11. The bank has the shape of
`sample_rows`'s with a single endpoint column, y [n, 1, ...] and r [n, 1], so the reward accounting that reads
r[:, 0] and r[:, -1] is unchanged; adjacent gains do not exist here.
"""
    texts = [prompts[pid] for _, pid, _ in rows]
    n = len(rows)
    f0, s0, n0 = rt.forward_samples, rt.score_samples, rt.nfe
    cloud = getattr(args, 'cloud_size', 0) or 0
    if cloud:
        # Row j of a group is main j // (1 + C) when j % (1 + C) == 0, else the (j % (1 + C))-th cloud member of that
        # main: x_T' = sqrt(1 - s^2) x_T + s eps, variance preserving, seeded per (group, main, member).
        s_ = float(args.cloud_strength)
        roots = []
        for g, _, j in rows:
            m, c = divmod(j, 1 + cloud)
            main = rt.noise(root_seed(args.seed, step, g, m, True))
            roots.append(main if c == 0 else
                         (1. - s_ * s_) ** .5 * main + s_ * rt.noise(root_seed(args.seed, step, g, 100 + m * 50 + c, True)))
        x = torch.cat(roots)
    else:
        x = torch.cat([rt.noise(root_seed(args.seed, step, g, j, True)) for g, _, j in rows])
    xs, vs = [], []
    for q, (t, s) in enumerate(zip(rt.times, rt.times[1:])):
        v = torch.cat([rt.velocity_rows(x[lo:lo + batch], t, texts[lo:lo + batch]) for lo in range(0, n, batch)])
        xs.append(x.cpu()); vs.append(v.cpu())
        x = x + (s - t) * v
    torch.cuda.synchronize(); tic = time.monotonic()
    scores = rt.score_rows(x, texts, args.score_chunk).cpu()
    torch.cuda.synchronize(); score_seconds = time.monotonic() - tic
    assert rt.forward_samples - f0 == (len(rt.times) - 1) * n, 'forward sample count changed'
    assert rt.score_samples - s0 == n, 'score sample count changed'
    one = torch.zeros(n, 1, 1, 1)
    bank = dict(x=torch.stack(xs, 1), old=torch.stack(vs, 1), child=torch.stack(xs[1:] + [x.cpu()], 1),
                mean=torch.stack(xs[1:] + [x.cpu()], 1), b=one.expand(n, len(rt.times) - 1, 1, 1).clone(),
                tau=one.expand(n, len(rt.times) - 1, 1, 1).clone(), y=x.cpu()[:, None], r=scores[:, None])
    mix = getattr(rt, 'last_mix', None)
    if mix is not None and 'ocr' in mix and len(mix['ocr']) == n:
        # Mixed OCR + PickScore reward: the raw OCR component decides failure eligibility and the success statistics.
        bank['r_ocr'] = mix['ocr'].float().cpu()[:, None]
    if mix is not None and 'source' in mix and len(mix['task']) == n:
        # Multi reward: source id per row (stepwise_runtime.SOURCES), task component (failure eligibility,
        # group classes) and the three quality scores + q for the per-source training statistics.
        bank['src'] = mix['source'].long().cpu()
        bank['r_task'] = mix['task'].float().cpu()[:, None]
        bank['q_pick'] = mix['pickscore'].float().cpu()[:, None]
        bank['q_clip'] = mix['clipscore'].float().cpu()[:, None]
        bank['q_hps'] = mix['hpsv2'].float().cpu()[:, None]
        bank['q_all'] = mix['q'].float().cpu()[:, None]
    return bank, dict(forward_calls=rt.nfe - n0, forward_samples=rt.forward_samples - f0,
                      score_samples=rt.score_samples - s0, score_seconds=score_seconds, innovation_rms_mean=0.)
