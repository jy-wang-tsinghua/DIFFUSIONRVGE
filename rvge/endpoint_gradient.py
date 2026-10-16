"""Stopped endpoint secants and local velocity actor surrogate.

q is the forward sampling index (physical diffusion time decreases). These
endpoint vectors are transferred to input coordinates; they are not unbiased
input-value gradients. No reward-model derivatives enter the estimator.
"""
import torch


@torch.no_grad()
def endpoint_gradient(endpoints, rewards, q, future_cap=4, ridge=0.1):
    """Past j<q ridge + mean of normalized edges q..min(q+cap,T)-1."""
    if endpoints.ndim < 2 or len(endpoints) != len(rewards):
        raise ValueError("expected [T+1,...] endpoints and [T+1] rewards")
    steps = len(endpoints) - 1
    if not 0 <= q <= steps or future_cap < 0 or ridge <= 0:
        raise ValueError("invalid query, cap or ridge")
    if not torch.isfinite(endpoints).all() or not torch.isfinite(rewards).all():
        raise ValueError("nonfinite estimator inputs")
    shape = endpoints.shape[1:]
    y = endpoints.detach().flatten(1)
    r = rewards.detach().to(device=y.device, dtype=torch.float64)
    past = torch.zeros_like(y[0], dtype=torch.float64)
    if q:
        # Match the audited cached FP32 endpoint subtraction before ridge FP64.
        e = (y[:q] - y[q]).double()
        norms = e.norm(dim=1)
        keep = norms > 1e-14
        if keep.any():
            a = e[keep] / norms[keep, None]
            c = (r[:q] - r[q])[keep] / norms[keep]
            gram = a @ a.T
            alpha = torch.linalg.solve(gram + ridge * torch.eye(len(a), device=y.device, dtype=a.dtype), c)
            past = a.T @ alpha
    stop = min(q + future_cap, steps)
    future = torch.zeros_like(past)
    if stop > q:
        dy = y[q+1:stop+1].double() - y[q:stop].double()
        dr = r[q+1:stop+1] - r[q:stop]
        n2 = dy.square().sum(1)
        # A zero displacement contributes zero, with the original edge count.
        secants = torch.where((n2 > 1e-28)[:, None],
                             dr[:, None] * dy / n2.clamp_min(1e-28)[:, None], 0.)
        future = secants.mean(0)
    return (past + future).reshape(shape)


@torch.no_grad()
def trajectory_directions(endpoints, rewards, future_cap=4, ridge=0.1, normalize=True, include_final=False):
    """T directions by default; include_final adds the clean-end query q=T.

The q=T entry carries only the past ridge term and is required when a training
row at q is paired with the value gradient of the state its velocity produces.
"""
    stop = len(endpoints) if include_final else len(endpoints)-1
    directions = torch.stack([endpoint_gradient(endpoints, rewards, q, future_cap, ridge)
                              for q in range(stop)])
    if normalize:
        scale = directions.flatten(1).square().mean(1).sqrt()
        directions /= scale.clamp_min(1e-30).reshape(-1, *([1]*(directions.ndim-1)))
    return directions.float().detach()


def velocity_ascent_loss(velocity, old_velocity, direction, weight=1.):
    """Positive dot loss: descending v moves the next state along +g (dt<0).

weight may precondition time contributions; it is not the old KDE beta factor.
"""
    return weight * ((velocity.float() - old_velocity.detach().float()) * direction.detach()).mean()


@torch.no_grad()
def local_linear_gradient(endpoints, rewards, base, kernel_scale=None, ridge_ratio=.3, base_reward=None, chord_floor=True):
    """Kernel-weighted local-linear ridge with a free intercept on RAW displacements.

    min_{g,b} sum_j w_j (r_j - b - <g, y_j - base>)^2 + lam |g|^2,   w_j = exp(-|y_j - base|^2 / (2 h^2)),
    lam = ridge_ratio * weighted median |y_j - base|^2 (chord-length floor: one flip contributes at most dr / 2 sqrt(lam)).

    Three properties the unit-chord secant ridge lacks on a discrete reward: nearer samples weigh more (kernel), a
    short chord cannot blow up (raw displacements + lam, no dr/|dy|), and the reward offset goes into the intercept
    instead of forcing g along -base (no "toward / away from the mean" blur-sharpen force). Dual solve, K x K.
    `kernel_scale` h defaults to the median chord length (uniform chords -> uniform weights -> the group covariance).
    `base_reward` adds the base point itself as a sample at zero displacement - NOT recommended: with a free intercept
    the regression then explains the base's own reward offset by the one direction that separates it from the others,
    its own noise, which is the "toward the mean" blur force again (synthetic check: cos(g, -eps_own) 0.60 with the base,
    0.00 without). `chord_floor` stretches chords shorter than the kernel scale to that scale (see below).
    """
    y = endpoints.detach().flatten(1).double()
    r = rewards.detach().double().to(y.device)
    b = base.detach().flatten().double()
    if base_reward is not None:
        y = torch.cat([y, b[None]]); r = torch.cat([r, torch.tensor([float(base_reward)], dtype=r.dtype, device=r.device)])
    e = y - b
    n2 = e.square().sum(1)
    med = n2[n2 > 0].median() if (n2 > 0).any() else torch.tensor(1., dtype=n2.dtype, device=n2.device)
    h2 = (float(kernel_scale) ** 2) if kernel_scale else float(med)
    if chord_floor:
        # A flip closer than the smoothing scale is a flip AT the smoothing scale: stretch short chords to length
        # sqrt(h2) so one near flip is a bounded, non-negligible vote instead of a divergent secant (secant ridge) or
        # a zero-leverage sample (plain least squares).
        n = n2.sqrt()
        stretch = torch.where(n > 0, (h2 ** .5 / n.clamp_min(1e-30)).clamp_min(1.), torch.ones_like(n))
        e = e * stretch[:, None]
        n2 = e.square().sum(1)
    w = torch.exp(-n2 / (2 * h2))
    if float(w.sum()) <= 0:
        return torch.zeros_like(endpoints[0])
    w = w / w.sum()
    e_c = e - (w[:, None] * e).sum(0, keepdim=True)          # weighted centring = free intercept
    r_c = r - (w * r).sum()
    lam = ridge_ratio * float(med)
    sw = w.sqrt()
    k = (sw[:, None] * e_c) @ (sw[:, None] * e_c).T
    alpha = torch.linalg.solve(k + lam * torch.eye(len(k), dtype=k.dtype, device=k.device), sw * r_c)
    return ((sw * alpha)[:, None] * e_c).sum(0).reshape(endpoints.shape[1:]).float()
