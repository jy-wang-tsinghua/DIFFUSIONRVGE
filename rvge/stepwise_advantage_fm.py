"""Advantage-weighted flow matching on the stopped endpoints (DiffusionNFT's update, one step per epoch).

DiffusionNFT (config beta=1, gradient_step_per_epoch=1) trains, for every sampled image x0 with
normalised advantage A clipped to [-c, c] and r = A/(2c) + 1/2:

    L = r * |x0_pos - x0|^2 / w_pos + (1 - r) * |x0_neg - x0|^2 / w_neg,
    x0_pos = x_t - t * v_theta,   x0_neg = x_t - t * (2 v_old - v_theta),   x_t = (1 - t) x0 + t eps,

with w the detached per-sample mean |x0_pred - x0|. With one optimizer step per epoch v_old is
v_theta at the point of differentiation, where the two weight factors coincide and the negative
term's gradient is minus the positive one's, so the whole update is

    dL/dv = (2r - 1) * d/dv |x_t - t v - x0|^2 / w = (A / c) * d/dv |x0_pred - x0|^2 / w :

a regression of the model's x0 prediction toward each sample's own x0, weighted by its centred
advantage. That is a selection signal - imitate the samples the reward accepted, unlearn the ones
it rejected - and it needs no reward differences along a trajectory, which is exactly the input
the endpoint estimator runs out of on a discrete reward.

This module applies that term to the endpoints the stepwise rollout already scored: the ODE
completions of every main state and the SDE endpoint, 11 per trajectory, 66 per prompt group of
6, 3168 per 288-trajectory update against NFT's 1152 images. The advantage is centred per prompt
group over all of its endpoints (NFT: per prompt over 24 images), scaled by the global std of the
update's endpoint rewards (NFT's sample.global_std), clipped at c = 5 and divided by c.
"""
import torch


def endpoint_advantages(rewards, slices, global_std, clip=5.):
    """[N, T+1] weights (A / clip) for one rank's endpoints; groups are the contiguous row slices."""
    adv = torch.zeros_like(rewards, dtype=torch.float64)
    for _, sl in slices:
        r = rewards[sl].double()
        adv[sl] = (r - r.mean()) / (global_std + 1e-4)
    return (adv.clamp(-clip, clip) / clip).float()


def select_endpoints(weights, slices, per_group, generator, min_weight=0.):
    """Up to `per_group` (row, endpoint) pairs per group with |weight| > min_weight, drawn at random."""
    picks = []
    for _, sl in slices:
        w = weights[sl]
        cand = [(sl.start + i, k) for i in range(w.shape[0]) for k in range(w.shape[1]) if abs(float(w[i, k])) > min_weight]
        if not cand:
            continue
        order = torch.randperm(len(cand), generator=generator).tolist()[:per_group]
        picks.extend(cand[j] for j in order)
    return picks


def advantage_fm_loss(pred, x_t, t, x0, weight):
    """Mean over the batch of weight * |x0_pred - x0|^2 / mean|x0_pred - x0| (NFT's adaptive weighting)."""
    x0_pred = x_t - t * pred.float()
    with torch.no_grad():
        factor = (x0_pred.double() - x0.double()).abs().mean(dim=tuple(range(1, x0.ndim)), keepdim=True).clamp_min(1e-5).float()
    per = ((x0_pred - x0) ** 2 / factor).mean(dim=tuple(range(1, x0.ndim)))
    return (weight * per).mean()
