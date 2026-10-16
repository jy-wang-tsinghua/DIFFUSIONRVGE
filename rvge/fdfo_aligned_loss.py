"""FDFO's old-policy loss-to-gradient map with a stopped external estimator.

This aligns only how a supplied reward gradient enters the actor loss. It does
not replace endpoint ridge/secant estimation or claim identical FDFO estimates.
"""


def fdfo_first_order_velocity_loss(velocity, old_velocity, reward_gradient, weight=1.):
    """Batch mean of 2 * coordinate-summed dot products; dL/dv = 2*w*g/B.

The caller averages main times, prompt groups, and ranks by their sample count.
Raw reward-gradient magnitudes must be preserved before calling this function.
Positive sign accounts for the decreasing-time flow sampler (dt < 0).
"""
    import torch
    if velocity.shape != old_velocity.shape or velocity.shape != reward_gradient.shape:
        raise ValueError('velocity, old velocity, and gradient shapes must agree')
    if velocity.ndim < 2 or velocity.shape[0] == 0:
        raise ValueError('an explicit nonempty batch dimension is required')
    delta = velocity.float() - old_velocity.detach().float()
    direction = reward_gradient.detach().to(device=velocity.device, dtype=torch.float32)
    return 2. * weight * (delta * direction).flatten(1).sum(1).mean()
