"""Gain-level baselines and time weights for the endpoint estimator.

The endpoint estimator consumes only reward *differences*, so subtracting any
per-trajectory constant - a global offset or a prompt-group mean - cancels
exactly and cannot act as a baseline. Centering has to happen one level down, on
the per-step gains, which is equivalent to removing a linear ramp from the
reward sequence and does survive the differencing.
"""
import torch

# Values of --reward-normalization. Every gain mode except gain_group_std divides by the same scalar
# scale (fixed, running, or the first group's own std), fed by the same pooled group std, so between
# them only the centring differs; gain_step_whiten additionally shapes that scale per step.
#
#   gain_group_std     pooled group x all-step mean, divided by that group's pooled std
#   gain_running_std   pooled group x all-step mean, running std (the head8 production mode)
#   gain_step_mean     group x per-step mean mu_k: removes a drift that differs between steps
#   gain_step_whiten   gain_step_mean, and the scale times a running per-step std profile (rms 1)
#   gain_step_loo      leave-one-out group x per-step mean; exactly gain_step_mean times N/(N-1)
#   gain_step_shared   pooled group mean plus the update-wide per-step shape (rows batching only)
#   gain_curvature     subtract kappa_g * |y_{k+1} - y_k|^2, kappa_g least squares over the group
#   gain_uncentered    no centring; same direction as 'none', divided by the running std
#   reward_zscore      (r - mean) / std of the group's final rewards; the uncentred direction
#                      rescaled per group, since reward-level centring is a no-op
#
# Why gain_curvature (from the regression reading of the estimator): the fit regresses
# r(y_m) - r(y_q) on <g, y_m - y_q>, so the only thing a baseline may legitimately remove is what that
# linear model cannot explain. The systematic part is the second-order Taylor term 1/2 dy^T H dy of the
# injected noise: even in the noise (the signal <g, dy> is odd) and proportional to |dy|^2, which falls
# steeply from q0 to q8. A flat mean ramp subtracts the same amount at every step, most harmfully at the
# late steps whose short chords 1/|dy| amplifies; one curvature coefficient per group follows |dy|^2.
REWARD_NORMALIZATIONS = ('none', 'gain_group_std', 'gain_running_std', 'gain_step_mean', 'gain_step_whiten',
                         'gain_step_loo', 'gain_step_shared', 'gain_curvature', 'gain_uncentered', 'reward_zscore')


def parse_reward_scale(text):
    """--reward-scale: None (off), a positive constant, or 'batch_std'.

With --reward-normalization none (one trajectory per prompt) no baseline or group statistic is used at all:
the regression consumes reward differences, so the only thing left to choose is a unit-setting scalar - a constant such
as 26 for PickScore, or the std of the update's gathered final rewards.
"""
    if text is None or str(text).strip() in ('', '1', '1.0', 'none'):
        return None
    if str(text).strip() == 'batch_std':
        return 'batch_std'
    value = float(text)
    if not value > 0:
        raise ValueError(f'--reward-scale must be positive or batch_std, got {text!r}')
    return value


def batch_reward_scale(final_rewards, eps=1e-8):
    """Uncorrected std of one update's final rewards (every rank's rows gathered)."""
    r = final_rewards.double().flatten()
    return max(float((r - r.mean()).square().mean().sqrt()), eps)


def endpoint_step_squares(endpoints):
    """|y_{k+1} - y_k|^2 per trajectory and step, [N, T], from stopped endpoints [N, T+1, ...]."""
    return torch.stack([(endpoints[:, k + 1] - endpoints[:, k]).double().flatten(1).square().sum(1)
                        for k in range(endpoints.shape[1] - 1)], 1)


def group_gain_statistics(rewards):
    """Mean and uncorrected std of the per-step gains of one prompt group."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError('expected [N, T+1] endpoint rewards')
    gains = rewards.double()[:, 1:] - rewards.double()[:, :-1]
    mean = gains.mean()
    return mean, (gains - mean).square().mean().sqrt()


def step_gain_statistics(rewards):
    """Per-step mean [T] and uncorrected per-step std [T] of one group's gains."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError('expected [N, T+1] endpoint rewards')
    gains = rewards.double()[:, 1:] - rewards.double()[:, :-1]
    mean = gains.mean(0)
    return mean, (gains - mean).square().mean(0).sqrt()


def normalize_group_rewards(rewards, scale, mean=None, center=True, eps=1e-8):
    """Center the gains on the group mean, divide by `scale`, reintegrate.

`scale` and `mean` are supplied by the caller so that statistics gathered over
the whole prompt group, or a running global standard deviation, can be used
while this rank holds only its own shard. The returned sequence starts at zero:
the estimator is invariant to that offset, and it makes the reconstruction
independent of the untouched absolute reward level.
"""
    if mean is None:
        mean, _ = group_gain_statistics(rewards)
    gains = rewards.double()[:, 1:] - rewards.double()[:, :-1]
    if center:
        gains = gains - mean
    gains = gains / max(float(scale), eps)
    zero = torch.zeros_like(gains[:, :1])
    return torch.cat([zero, gains.cumsum(1)], 1).to(rewards.dtype)


def update_running_gain_std(running, group_std, decay):
    """Exponential running gain scale, fed by informative groups only.

A group whose gains are all equal carries no signal, and its centred gains are
exactly zero whatever they are divided by. Folding its zero std in would only
shrink the scale the next informative group is divided by - and if it is the first
group, leave the scale at 0 and the division at eps. A discrete reward (OCR)
produces such groups; a continuous one never does, so for PickScore this is the
original recursion bit for bit.
"""
    group_std = float(group_std)
    if group_std <= 0:
        return running
    return group_std if running is None else decay * running + (1 - decay) * group_std


class GainNormalizer:
    """Every --reward-normalization mode behind one per-group call.

Statistics come from the whole gathered prompt group `whole` [N, T+1] and are
applied to this rank's rows of it, `local` [n, T+1] (n may be 0: every rank walks
every group so the running state advances identically everywhere). The scale is
read from the state before the group and the state is updated after it, which is
the historical order; gain_running_std and gain_group_std reproduce the old inline
trainer code bit for bit.
"""

    # A per-step profile entry below this (relative to rms 1) would turn a step that
    # has shown no spread so far into a division by ~eps. PickScore never gets near it.
    profile_floor = .1

    def __init__(self, mode, decay=.9, fixed=None, fixed_start=0, eps=1e-8):
        if mode not in REWARD_NORMALIZATIONS:
            raise ValueError(f'unknown reward normalization {mode!r}')
        self.mode, self.decay, self.fixed, self.fixed_start, self.eps = mode, decay, fixed, fixed_start, eps
        self.running = None
        self.running_steps = None
        self.update_profile = None
        self.last_kappa = None

    @property
    def needs_update_statistics(self):
        return self.mode == 'gain_step_shared'

    @property
    def needs_step_squares(self):
        """The caller must pass endpoint_step_squares for `local` and `whole`."""
        return self.mode == 'gain_curvature'

    def begin_update(self, rewards):
        """Record the update-wide per-step gain shape from every gathered row, [M, T+1]."""
        mean, _ = step_gain_statistics(rewards)
        self.update_profile = mean - mean.mean()

    @staticmethod
    def curvature(whole, whole_squares):
        """Least-squares kappa of gains ~ kappa * |dy|^2 over one group, through the origin."""
        gains = whole.double()[:, 1:] - whole.double()[:, :-1]
        s = whole_squares.double()
        return float((gains * s).sum() / s.square().sum().clamp_min(1e-300))

    def scale(self, group_std, step):
        if self.mode == 'gain_group_std':
            return float(group_std)
        if self.fixed is not None and step >= self.fixed_start:
            return self.fixed
        return self.running if self.running is not None else float(group_std)

    def step_profile(self, step_std):
        profile = self.running_steps if self.running_steps is not None else step_std
        rms = float(profile.square().mean().sqrt())
        if rms <= 0:
            return torch.ones_like(profile)
        return (profile / rms).clamp_min(self.profile_floor)

    def __call__(self, local, whole, step, local_squares=None, whole_squares=None):
        """-> (normalized local rewards, scalar scale, whether the group carries signal).

`local_squares` / `whole_squares` are endpoint_step_squares of the same rows, needed by gain_curvature only.
"""
        mean, std = group_gain_statistics(whole)
        informative = float(std) > 0
        if self.mode == 'none':
            return local.clone(), 1., informative
        if self.mode == 'reward_zscore':
            final = whole.double()[:, -1]
            center = final.mean()
            scale = float((final - center).square().mean().sqrt())
            out = ((local.double() - center) / max(scale, self.eps)).to(local.dtype)
        else:
            scale = self.scale(std, step)
            if self.mode in ('gain_group_std', 'gain_running_std'):
                out = normalize_group_rewards(local, scale, mean=mean)
            elif self.mode == 'gain_uncentered':
                out = normalize_group_rewards(local, scale, mean=mean, center=False)
            else:
                step_mean, step_std = step_gain_statistics(whole)
                gains = local.double()[:, 1:] - local.double()[:, :-1]
                if self.mode == 'gain_step_loo':
                    n = len(whole)
                    if n < 2:
                        raise ValueError('leave-one-out centring needs at least two trajectories per group')
                    center = (n * step_mean - gains) / (n - 1)
                elif self.mode == 'gain_step_shared':
                    if self.update_profile is None:
                        raise RuntimeError('gain_step_shared needs begin_update() with the whole update first')
                    center = mean + self.update_profile
                elif self.mode == 'gain_curvature':
                    if local_squares is None or whole_squares is None:
                        raise ValueError('gain_curvature needs the endpoint step squares of local and whole rows')
                    self.last_kappa = self.curvature(whole, whole_squares)
                    center = self.last_kappa * local_squares.double()
                else:
                    center = step_mean
                divisor = torch.full_like(step_mean, max(float(scale), self.eps))
                if self.mode == 'gain_step_whiten':
                    divisor = (divisor * self.step_profile(step_std)).clamp_min(self.eps)
                    if informative:
                        self.running_steps = (step_std.clone() if self.running_steps is None else
                                              self.decay * self.running_steps + (1 - self.decay) * step_std)
                residual = (gains - center) / divisor
                zero = torch.zeros_like(residual[:, :1])
                out = torch.cat([zero, residual.cumsum(1)], 1).to(local.dtype)
        self.running = update_running_gain_std(self.running, std, self.decay)
        return out, scale, informative


def inverse_rms_time_weights(sum_squares, counts, floor=1e-30):
    """Weights proportional to 1/rms per main time index, averaging to one.

Equalizes the gradient energy each main time contributes. `sum_squares` and
`counts` are accumulated across ranks so the statistic covers the whole update.
"""
    rms = (sum_squares / counts.clamp_min(1)).sqrt()
    weights = 1. / rms.clamp_min(floor)
    return weights / weights.mean()
