"""Does widening the fit at q pay for rank, or only for chord length?

Folding the whole trajectory into the fit at q adds rows (rank), but an endpoint
after q sits further from y_q than one before it, so it also lengthens the average
chord - and raising the SDE radius lengthens the same chord. If a gain is the chord
moving rather than the rank, the noise radius already buys it and it does not add.

The window separates the two. Rows within +-k of q hold the chord near the base
point while the row count still grows with k, so gain against k at fixed noise
is a rank axis with the chord pinned, and gain against noise at fixed k is a
chord axis with the rank pinned. If the two collapse onto one curve in chord
length, the rank reading is wrong.

Every variant here is the same `endpoint_gradient` with a different row set, so
nothing but the rows changes between them.
"""
import torch

from rvge.endpoint_gradient import endpoint_gradient, local_linear_gradient


def window_rows(steps, q, width=None):
    """Indices of the other endpoints a fit at q may use, within +-width.

`steps` is T, so endpoint indices run 0..T inclusive. `width=None` takes them
all, which is the traj_all row set.
"""
    if width is None:
        lo, hi = 0, steps
    else:
        if width < 1:
            raise ValueError('width must be at least 1')
        lo, hi = max(0, q - width), min(steps, q + width)
    return [j for j in range(lo, hi + 1) if j != q]


@torch.no_grad()
def fit_rows(y, r, rows, q, ridge=.3):
    """One joint ridge over `rows`, centred at y[q], with an empty future term.

The base point goes last and is queried there, so the past term sees exactly
{y[j] - y[q]} against {r[j] - r[q]} and `stop == q` leaves the future empty.
This is the same wiring `stepwise_pooled_fit` uses, so traj_all agrees bitwise.
"""
    if not rows:
        raise ValueError('a fit needs at least one row')
    idx = torch.tensor(rows, device=y.device, dtype=torch.long)
    ends = torch.cat([y[idx], y[q:q + 1]])
    rew = torch.cat([r[idx], r[q:q + 1]])
    return endpoint_gradient(ends, rew, len(rows), 0, ridge)


def row_sets(steps, q, widths):
    """The row set behind each fitted variant, for chord and rank accounting."""
    sets = {'own_past': list(range(q)), 'traj_all': window_rows(steps, q)}
    for w in widths:
        sets[f'traj_w{w}'] = window_rows(steps, q, w)
    return sets


@torch.no_grad()
def window_directions(y, r, q, widths=(1, 2, 3, 5), future_cap=4, ridge=.3):
    """Directions at step q of one trajectory: production fit and width family.

`y` is [T+1, ...] and `r` is [T+1]: the stopped endpoints of one trajectory.
`own` is the production estimator and `own_past` is it without the future
secants, which is the fit the width family widens.
"""
    steps = len(y) - 1
    out = dict(own=endpoint_gradient(y, r, q, future_cap, ridge),
               own_past=endpoint_gradient(y, r, q, 0, ridge),
               traj_all=fit_rows(y, r, window_rows(steps, q), q, ridge))
    for w in widths:
        out[f'traj_w{w}'] = fit_rows(y, r, window_rows(steps, q, w), q, ridge)
    return out


@torch.no_grad()
def chord_lengths(y, rows, q):
    """|y[j] - y[q]| for a fit's rows, relative to |y[q]|.

The chord is what the radius moves, so reporting it in the same units for every
variant is what lets the width axis and the noise axis be read on one scale.
"""
    if not rows:
        return torch.zeros(0, dtype=torch.float64, device=y.device)
    idx = torch.tensor(rows, device=y.device, dtype=torch.long)
    base = y[q].flatten().double()
    return (y[idx].flatten(1).double() - base).norm(dim=1) / base.norm().clamp_min(1e-30)


@torch.no_grad()
def trajectory_window_directions(endpoints, rewards, scope='own', width=3, future_cap=4,
                                 ridge=.3, normalize=True, include_final=False, head_rows=8, target_min=None):
    """`trajectory_directions` with the fit's row set chosen by `scope`.

Contract, shapes and normalization are those of the production helper, and
`scope='own'` returns its result bitwise, so this can stand in for it
unconditionally. The other scopes replace the past ridge and its capped future
secants with one joint ridge over the row set named:

  own_past      the trajectory's endpoints before q
  traj_window   its endpoints within +-width of q
  traj_all      every one of its other endpoints
  head          its endpoints with index < head_rows, and no future term

`head` exists because the last endpoints are not neutral filler in the fit: their
displacements are the smallest of the trajectory, so their secants Dr/|Dy|
amplify the same reward noise over a much shorter chord.
"""
    steps = len(endpoints) - 1
    stop = steps + 1 if include_final else steps

    def at(q):
        if scope == 'own':
            return endpoint_gradient(endpoints, rewards, q, future_cap, ridge)
        if scope == 'own_past':
            return endpoint_gradient(endpoints, rewards, q, 0, ridge)
        if scope == 'head':
            rows = [j for j in range(min(head_rows, steps + 1)) if j != q]
            if not rows:
                raise ValueError('head scope left no rows; raise --fit-head-rows')
            return fit_rows(endpoints, rewards, rows, q, ridge)
        if scope == 'local':
            # Local-linear ridge over ALL other completions of the trajectory (kernel + chord floor make the short late
            # chords safe, so nothing is dropped); the base is the kernel centre only. See endpoint_gradient.
            rows = [j for j in range(steps + 1) if j != q]
            return local_linear_gradient(endpoints[rows], rewards[rows], endpoints[q], ridge_ratio=ridge)
        if scope == 'nearest_success':
            # Like nearest_higher but the target must be a SUCCESS (r >= 0.99). With GenEval's
            # partial credit an empty image (0) outscores a cluttered one (negative), so "toward the trajectory's best
            # completion" can push failing trajectories toward emptiness; a success is a real target.
            y = endpoints.detach().flatten(1).double()
            r = rewards.detach().double().to(y.device)
            higher = [j for j in range(steps + 1) if j != q and r[j] >= .99 and r[j] > r[q] + 1e-6]
            if not higher:
                return torch.zeros_like(endpoints[q])
            d = torch.stack([(y[j] - y[q]).norm() for j in higher])
            j = higher[int(d.argmin())]
            dy = y[j] - y[q]
            # unit rms x 3e-5: one flip = one vote at the rms scale of the recipe's ridge directions
            return (3e-5 * dy / dy.square().mean().sqrt().clamp_min(1e-30)).reshape(endpoints.shape[1:]).float()
        if scope == 'nearest_higher':
            # One secant toward the NEAREST completion of the same
            # trajectory whose reward exceeds this state's, no averaging - an averaged target (a ridge over several
            # completions or siblings) acts as a blur force when only low rows are trained. Zero when no completion scores higher.
            y = endpoints.detach().flatten(1).double()
            r = rewards.detach().double().to(y.device)
            # `target_min`: a target must score above it, so an empty
            # image (GenEval 0) never becomes the "best" of an all-cluttered counting trajectory (negative scores).
            floor = -float('inf') if target_min is None else float(target_min)
            higher = [j for j in range(steps + 1) if j != q and r[j] > r[q] + 1e-6 and r[j] > floor]
            if not higher:
                return torch.zeros_like(endpoints[q])
            d = torch.stack([(y[j] - y[q]).norm() for j in higher])
            j = higher[int(d.argmin())]
            dy = y[j] - y[q]
            # Unit rms x 3e-5 per row (one flip = one vote at the recipe's direction scale): a secant over a short chord
            # would otherwise dominate the update, and the FDFO actor loss forbids the trainer-level normalisation.
            return (3e-5 * dy / dy.square().mean().sqrt().clamp_min(1e-30)).reshape(endpoints.shape[1:]).float()
        rows = window_rows(steps, q, None if scope == 'traj_all' else width)
        return fit_rows(endpoints, rewards, rows, q, ridge)

    if scope not in ('own', 'own_past', 'traj_window', 'traj_all', 'head', 'nearest_higher', 'nearest_success', 'local'):
        raise ValueError(f'unknown fit scope {scope!r}')
    directions = torch.stack([at(q) for q in range(stop)])
    if normalize:
        scale = directions.flatten(1).square().mean(1).sqrt()
        directions /= scale.clamp_min(1e-30).reshape(-1, *([1] * (directions.ndim - 1)))
    return directions.float().detach()
