"""Fit each direction against every head endpoint of its prompt group, not only its own trajectory.

The head-8 estimator fits the direction at (trajectory i, step q) on the 7 other head
endpoints of trajectory i alone. On a discrete reward that row set is empty of signal
whenever the 8 branches of one trajectory all land on the same reward value, which on
GenEval is the common case. The other trajectories of the same prompt are samples of the same
reward landscape and have already been paid for; a trajectory whose branches all failed
still sees, through them, which endpoint images of this prompt the reward accepted.

With independent roots the sibling endpoints are far from y[i, q], so their rows are long
chords: the per-row normalisation e_j / |e_j|, b_j / |e_j| in `endpoint_gradient` already
weights them down against the short own-branch chords, and a sibling row is the same
"toward the better sample" contrast that a group advantage supplies to GRPO / NFT, expressed
in endpoint space instead of through the likelihood.

The default shard interleaves a group across ranks;
`--shard-mode block` keeps a group on one rank, which is what makes this fit local.
"""
import torch

from rvge.endpoint_gradient import endpoint_gradient, local_linear_gradient


@torch.no_grad()
def pooled_group_directions(y, r, head_rows=8, ridge=.3, nearest=None, own_only_steps=()):
    """[N, T, ...] directions for one prompt group from its [N, T+1, ...] endpoints and [N, T+1] rewards.

At (i, q) the row set is every endpoint of the group with time index below `head_rows`,
except the base point y[i, q] itself, in one ridge with an empty future term - the head
scope of `trajectory_window_directions` with the group's other trajectories folded in.
`nearest` keeps only the K rows closest to the base point in endpoint space (rank for
chord length, as `stepwise_pooled_fit.pool_near`); `own_only_steps` lists steps q that
fall back to the own-trajectory head fit.
"""
    if y.ndim < 3 or r.shape != y.shape[:2]:
        raise ValueError('expected y [N, T+1, ...] and r [N, T+1]')
    n, steps = r.shape[0], r.shape[1] - 1
    head = min(head_rows, steps + 1)
    pool_y = y[:, :head].flatten(0, 1)
    pool_r = r[:, :head].flatten(0, 1)
    out = []
    for i in range(n):
        per = []
        for q in range(steps):
            base_y, base_r = y[i, q], r[i, q]
            if q in own_only_steps:
                keep = torch.zeros(len(pool_y), dtype=torch.bool, device=y.device)
                keep[i * head:(i + 1) * head] = True
            else:
                keep = torch.ones(len(pool_y), dtype=torch.bool, device=y.device)
            if q < head:
                keep[i * head + q] = False
            rows_y, rows_r = pool_y[keep], pool_r[keep]
            if nearest and len(rows_y) > nearest:
                d = (rows_y.flatten(1).double() - base_y.flatten().double()).norm(dim=1)
                idx = d.topk(nearest, largest=False).indices
                rows_y, rows_r = rows_y[idx], rows_r[idx]
            ends = torch.cat([rows_y, base_y[None]])
            rew = torch.cat([rows_r, base_r[None]])
            per.append(endpoint_gradient(ends, rew, len(rows_y), 0, ridge))
        out.append(torch.stack(per))
    return torch.stack(out).float()


def group_slices(rows, group_size):
    """Contiguous (group, slice) runs of a block-sharded row list; fails unless every group is whole.

`group_size=None` (--mixed-extra-k) accepts groups of unequal size: a run is still one whole group."""
    runs, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i][0] != rows[start][0]:
            runs.append((rows[start][0], slice(start, i)))
            start = i
    seen = [g for g, _ in runs]
    if len(seen) != len(set(seen)) or (group_size is not None and any(sl.stop - sl.start != group_size for _, sl in runs)):
        raise ValueError('a prompt group is split across ranks; use --shard-mode block')
    return runs


@torch.no_grad()
def sibling_directions(y, r, trained, ridge=.3, steps=10, centre='self'):
    """[M, T, ...] directions for the trained trajectories of an ODE group from the group's endpoints.

`centre='mean'` puts the ridge base point at the leave-one-out mean (ȳ, r̄) of the other endpoints
instead of (y_i, r_i): the fit becomes the regression of r on centred y - the shared group-covariance direction with
the self term removed. The self term (toward ȳ for rows below the group mean, toward y_i for rows above it) is a
coherent blur/sharpen force that random row selection merely balances; this isolates the shared part.

`y` is [K, ...] (one endpoint per trajectory) and `r` is [K]; `trained` is a count (the first M rows) or a list
of row indices. The direction of trajectory i is one ridge over the other K-1 endpoints centred at y[i] - the
same `endpoint_gradient` past term, with independent initial noises supplying the displacements instead of
injected SDE noise. It is fitted once in endpoint space and
applied at every main state of trajectory i, so it is repeated along T.
"""
    if y.ndim < 2 or r.shape != y.shape[:1]:
        raise ValueError('expected y [K, ...] and r [K]')
    k = y.shape[0]
    if k < 2:
        raise ValueError('a sibling fit needs at least two trajectories per group')
    out = []
    rows_ = range(trained) if isinstance(trained, int) else list(trained)
    if len(rows_) == 0:
        return torch.zeros((0, steps) + tuple(y.shape[1:]), dtype=torch.float32, device=y.device)
    for i in rows_:
        others = torch.tensor([j for j in range(k) if j != i], device=y.device)
        if centre == 'local':
            # Kernel-weighted local-linear ridge, free intercept, raw displacements, chord floor; the base point is the
            # kernel centre only (endpoint_gradient.local_linear_gradient). Uniform chords -> the group covariance.
            d = local_linear_gradient(y[others], r[others], y[i], ridge_ratio=ridge)
            out.append(d[None].expand(steps, *d.shape).clone()); continue
        if centre == 'mean':
            base_y, base_r = y[others].mean(0, keepdim=True), r[others].mean().reshape(1)
        elif centre == 'self':
            base_y, base_r = y[i:i + 1], r[i:i + 1]
        else:
            raise ValueError(f'unknown sibling centre {centre}')
        ends = torch.cat([y[others], base_y])
        rew = torch.cat([r[others], base_r])
        d = endpoint_gradient(ends, rew, k - 1, 0, ridge)
        out.append(d[None].expand(steps, *d.shape).clone())
    return torch.stack(out).float()


@torch.no_grad()
def cloud_directions(y, r, mains, cloud, ridge=.3, steps=10, pool_mains=False):
    """[M, T, ...] directions for the M main trajectories of an ODE-cloud group.

The group holds M x (1 + C) endpoints ordered main_0, its C cloud members, main_1, its C members, ... (see
`stepwise_rows.sample_rows_ode` with a cloud size): each cloud member started from the main's initial noise
perturbed by a variance-preserving Gaussian of strength s. The direction of main m is one ridge over its own C
cloud endpoints centred at its endpoint (a local finite-difference design); `pool_mains` adds the other mains and their clouds as further rows.
"""
    if y.ndim < 2 or r.shape != y.shape[:1] or y.shape[0] != mains * (1 + cloud):
        raise ValueError('expected M x (1 + C) endpoints')
    width = 1 + cloud
    out = []
    for m in range(mains):
        base = m * width
        rows = list(range(base + 1, base + width))
        if pool_mains:
            rows += [j for j in range(len(y)) if j // width != m]
        idx = torch.tensor(rows, device=y.device)
        ends = torch.cat([y[idx], y[base:base + 1]])
        rew = torch.cat([r[idx], r[base:base + 1]])
        d = endpoint_gradient(ends, rew, len(rows), 0, ridge)
        out.append(d[None].expand(steps, *d.shape).clone())
    return torch.stack(out).float()


@torch.no_grad()
def group_advantages(r, global_std, clip=5., eps=1e-4):
    """[K] DiffusionNFT advantages of one group: (r - group mean) / (update-wide std + eps), clipped at +-clip, / clip."""
    a = (r.double() - r.double().mean()) / (float(global_std) + eps)
    return (a.clamp(-clip, clip) / clip).float()


@torch.no_grad()
def anchor_directions(x, old, times, y, r, trained, global_std, clip=5.):
    """[M, T, ...] DiffusionNFT's one-sample value-gradient direction on the ODE's own states.

In the path-space view (Xu et al. 2026, Prop. 3) every diffusion-RL method estimates the value gradient
grad_{x_t} V_t = E[A(x_0) | x_t]; DiffusionNFT's estimator is A_i (x_0^i - x0hat(x_t)) / t up to weights,
where x0hat = x_t - t v_old(x_t) is the model's own posterior mean. The sibling ridge is that estimator's t = 1
limit (prior covariance, Cov(A, x_0)) repeated at every state; this one is local: at state q of trajectory i
the direction is A_i times the residual between its own endpoint and the model's one-step prediction there,
e_{i,q} = y_i - (x_{i,q} - t_q old_{i,q}), rms-normalised per row (NFT divides by the per-sample mean |e|), so a
success pulls the field toward its own endpoint and a failure pushes it away, with |A| set by how far the
sample sits from its group mean. `x`/`old` are [K, T, ...] states and velocities, `times` the T sampling
times, `y` [K, ...] endpoints, `r` [K] rewards; `trained` as in `sibling_directions`.
"""
    if x.ndim < 3 or old.shape != x.shape or y.shape != x.shape[:1] + x.shape[2:] or r.shape != x.shape[:1]:
        raise ValueError('expected x/old [K, T, ...], y [K, ...] and r [K]')
    steps = x.shape[1]
    a = group_advantages(r, global_std, clip)
    t = torch.tensor([float(v) for v in list(times)[:steps]], dtype=torch.float64, device=x.device)
    t = t.view(steps, *([1] * (x.ndim - 2)))
    out = []
    for i in (range(trained) if isinstance(trained, int) else trained):
        e = y[i][None].double() - (x[i].double() - t * old[i].double())
        rms = e.flatten(1).square().mean(1).sqrt().clamp_min(1e-30).view(steps, *([1] * (e.ndim - 1)))
        out.append((float(a[i]) * e / rms).float())
    return torch.stack(out)


def select_trained(r, per, mode, generator, max_reward=None, eligibility=None):
    """Row indices of one group that carry actor rows: the first `per`, or the `per` farthest from the group mean.

`max_reward`: only rows with r < max_reward are eligible, so a group contributes min(per, #failures) rows
(possibly none).
`eligibility` (mixed OCR + PickScore reward): the per-row values the max_reward rule reads instead of `r`
(the OCR component), so a zero-OCR image with a high PickScore still counts as a failure; ordering still uses `r`."""
    if mode == 'first':
        return list(range(per))
    jitter = torch.rand(len(r), generator=generator).double().to(r.device) * 1e-6  # random tie-break
    if mode == 'abs_adv':
        key = (r.double() - r.double().mean()).abs()
    elif mode == 'low':      # does the shared direction only help the low-reward rows of a group?
        key = -r.double()
    elif mode == 'high':     # negative control: the rows already at the reward cap
        key = r.double()
    else:
        raise ValueError(f'unknown --train-select {mode}')
    order = (key + jitter).argsort(descending=True).tolist()
    if max_reward is not None:
        elig = r if eligibility is None else eligibility
        order = [i for i in order if float(elig[i]) < max_reward]
    return sorted(order[:per])


@torch.no_grad()
def group_nearest_success_directions(y, r, cap=.99, scale=3e-5):
    """[K, T, ...] directions for every (trajectory, state) of one prompt group from the group's completions.

`y` is [K, T+1, ...] (the ODE completion of every state plus the final endpoint of each SDE trajectory) and `r` [K, T+1].
State q of trajectory i gets the unit vector from its own completion toward the NEAREST completion in the whole group
(any trajectory, any state, excluding itself) whose reward is >= `cap` - one sharp success image, no averaging. Same
trajectory completions share the layout and are usually the nearest; siblings supply a target when the trajectory
has none. Zero when the state is
itself a success or no success exists. Scaled to the recipe's direction rms.
"""
    if y.ndim < 3 or r.shape != y.shape[:2]:
        raise ValueError('expected y [K, T+1, ...] and r [K, T+1]')
    k, cols = r.shape
    steps = cols - 1
    flat = y.flatten(2).double()                                  # [K, T+1, D]
    succ = [(j, c) for j in range(k) for c in range(cols) if float(r[j, c]) >= cap]
    out = torch.zeros(k, steps, *y.shape[2:], dtype=torch.float32, device=y.device)
    if not succ:
        return out
    sy = torch.stack([flat[j, c] for j, c in succ])               # [S, D]
    for i in range(k):
        for q in range(steps):
            if float(r[i, q]) >= cap:
                continue
            d2 = (sy - flat[i, q][None]).square().sum(1)
            for n, (j, c) in enumerate(succ):
                if j == i and c == q:
                    d2[n] = float('inf')
            n = int(d2.argmin())
            if not torch.isfinite(d2[n]):
                continue
            dy = sy[n] - flat[i, q]
            out[i, q] = (scale * dy / dy.square().mean().sqrt().clamp_min(1e-30)).reshape(y.shape[2:]).float()
    return out
