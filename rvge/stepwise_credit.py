"""Stepwise-Flow-GRPO, local PDF equations (2), (5), (6), and optional (8)."""
import math
import torch


def flow_coefficients(t, s, level, first_nonboundary_time):
    if not (0 <= s < t <= 1):
        raise ValueError("require decreasing physical time")
    denom = 1 - (first_nonboundary_time if t == 1 else t)
    diffusion2 = t / denom * level**2
    dt = s-t
    return 1+diffusion2/(2*t)*dt, (1+diffusion2*(1-t)/(2*t))*dt, (diffusion2*(-dt))**.5


def ddim_coefficients(t, s, eta=.9):
    # Main-text Eq. (8)/(9); sigma=eta*s*sqrt(1-t). Zero-noise edges
    # at t=1 or s=0 are deterministic and do not admit Gaussian log densities.
    tau = eta*s*math.sqrt(1-t)
    noise_coef = math.sqrt(max(s*s-tau*tau,0.))
    return 1-s+noise_coef, -(1-s)*t+noise_coef*(1-t), tau


SDE_SCHEDULES = ('flow', 'flow_annealed', 'ddim', 'state_rms')


def sde_coefficients(t, s, level, first_nonboundary_time, schedule='flow', noise_exponent=0.):
    """Main-trajectory SDE step: mu = a*x + b*v, child = mu + tau*eps.

`level` is eta, a float or a per-row tensor. The schedules differ only in how
tau is spread over physical time, i.e. in where along generation the exploration
-- and with it the departure from the ODE manifold -- is placed.

  flow           tau = eta*sqrt(t/(1-t))*sqrt(t-s)   Flow-GRPO Eq. (1), the shipped one
  flow_annealed  the same, times t**noise_exponent   monotone, tail driven to zero
  ddim           tau = eta*s*sqrt(1-t)               Stepwise-CA Eq. (8)/(9)

`state_rms` is not a tau shape but a normalization of one; callers implement it
by scaling `level` and pass 'flow' here. Every schedule reduces to the Euler ODE
step a=1, b=s-t at eta=0, which is what makes them comparable at all.
"""
    if schedule in ('flow', 'state_rms'):
        return flow_coefficients(t, s, level, first_nonboundary_time)
    if schedule == 'flow_annealed':
        if noise_exponent < 0:
            raise ValueError("annealing exponent must be non-negative")
        return flow_coefficients(t, s, level * float(t)**noise_exponent, first_nonboundary_time)
    if schedule != 'ddim':
        raise ValueError(f"unknown sde schedule {schedule!r}")
    if not (0 <= s < t <= 1):
        raise ValueError("require decreasing physical time")
    # Eq. (8) puts sqrt(s**2-tau**2) on the predicted noise, so eta*sqrt(1-t) > 1
    # leaves no real coefficient: the step is being asked to inject more noise
    # than the marginal at s carries. Fail instead of clamping, which would
    # silently run a different schedule than the one requested.
    tau = level * (s * math.sqrt(1 - t))
    slack = s * s - tau * tau
    if isinstance(slack, torch.Tensor):
        if bool((slack < 0).any()):
            raise ValueError("ddim schedule level exceeds 1/sqrt(1-t); no real noise coefficient")
        noise_coef = slack.clamp_min(0.).sqrt()
    else:
        if slack < 0:
            raise ValueError("ddim schedule level exceeds 1/sqrt(1-t); no real noise coefficient")
        noise_coef = math.sqrt(slack)
    return 1 - s + noise_coef, -(1 - s) * t + noise_coef * (1 - t), tau


def sde_schedule_of(args):
    """Schedule settings of an args namespace, defaulting to the shipped flow SDE."""
    schedule = getattr(args, 'sde_schedule', 'flow')
    if schedule not in SDE_SCHEDULES:
        raise ValueError(f"unknown sde schedule {schedule!r}")
    if schedule == 'state_rms':
        raise ValueError("state_rms is selected through --sde-noise-mode, not --sde-schedule")
    if getattr(args, 'sde_noise_mode', 'constant') == 'state_rms' and schedule != 'flow':
        # state_rms divides by the schedule's own unit tau to hit radius*rms
        # exactly, which cancels the shape and leaves only the much smaller
        # O(eta**2) drift terms differing. The resulting arm would not be the
        # schedule it is labelled with.
        raise ValueError('state_rms noise mode cancels the tau shape; it is only defined with --sde-schedule flow')
    return schedule, float(getattr(args, 'sde_noise_exponent', 0.))


def uniform_value_nodes(t, substeps=5):
    if t==0:
        return [0.]
    if substeps<1:
        raise ValueError("positive substeps required")
    return [float(t)*(substeps-i)/substeps for i in range(substeps+1)]


def stochastic_steps(times, sampler='flow', level=.7, eta=.9):
    """Transition indices that admit a Gaussian policy, i.e. those with tau>0.

    The DDIM schedule sigma=eta*s*sqrt(1-t) vanishes at both ends of the SD3.5
    grid: at t=1 the sqrt(1-t) factor is zero and at the last step s is zero.
    Such a transition is deterministic, so no propensity ratio and no policy
    gradient term exist for it -- Eq. (6) simply has no summand there. Callers
    skip these steps rather than flooring tau, which would fabricate a policy
    the paper does not define. The Flow-GRPO schedule keeps tau>0 throughout.
    """
    keep = []
    for q, (t, s) in enumerate(zip(times, times[1:])):
        tau = (flow_coefficients(t, s, level, times[1])[2] if sampler == 'flow'
               else ddim_coefficients(t, s, eta)[2])
        if float(tau) > 0:
            keep.append(q)
    return keep


def step_gradient_coefficients(times, sampler='flow', level=.7, eta=.9):
    """Per-transition (b, tau) of the sampler, i.e. what the policy gradient rides on.

    d/dv of log pi carries b/tau, so (b_q/tau_q)**2 is the share of the update's
    score-function variance that transition q contributes, while the reward's
    response to that step's exploration scales with tau_q. The ratio is therefore
    the conditioning number of the step, and it is a property of the sampler AND
    the time grid together -- not of either alone.
    """
    out = []
    for t, s in zip(times, times[1:]):
        _, b, tau = (flow_coefficients(t, s, level, times[1]) if sampler == 'flow'
                     else ddim_coefficients(t, s, eta))
        out.append((float(b), float(tau)))
    return out


def trainable_transitions(times, sampler='flow', level=.7, eta=.9,
                          timestep_fraction=1., max_btau=float('inf')):
    """Transitions that carry a policy-gradient term, under three separate rules.

    1. tau > 0, i.e. `stochastic_steps`: a deterministic transition has no Gaussian
       density, so Eq. (6) has no summand there. The DDIM schedule has such edges at
       t=1 and s=0.
    2. q < int(T * timestep_fraction), which is Flow-GRPO's own truncation
       (scripts/train_sd3.py:345,869); config/grpo.py::pickscore_sd3 ships 0.99, so
       at T=10 upstream never trains the last transition.
    3. |b|/tau <= max_btau. A step far above the rest of its own schedule is
       numerically the same object as rule 1's deterministic edge: it contributes
       (b/tau)**2 of gradient variance for a reward response that scales with tau.
       On the SD3.5 grid the DDIM schedule puts q8 at 39.75 against at most 1.77
       everywhere else, in either schedule. Default inf leaves the rule off.
    """
    if not 0 < timestep_fraction <= 1:
        raise ValueError('timestep fraction must lie in (0, 1]')
    if max_btau <= 0:
        raise ValueError('max |b|/tau must be positive')
    keep = int((len(times) - 1) * timestep_fraction)
    out = []
    for q, (b, tau) in enumerate(step_gradient_coefficients(times, sampler, level, eta)):
        if tau <= 0 or q >= keep:
            continue
        if abs(b) / tau > max_btau:
            continue
        out.append(q)
    return out


def joint_gain_advantages(rewards, eps=1e-8):
    """One prompt group [N,T+1], jointly normalized over N and T (Eq. 5)."""
    gains = rewards.double()[:,1:]-rewards.double()[:,:-1]
    mean = gains.mean()
    std = (gains-mean).square().mean().sqrt()
    return ((gains-mean)/std.clamp_min(eps)).float(), gains, mean, std


def gaussian_logprob(sample, mean, tau):
    if not torch.all(tau>0):
        raise ValueError("deterministic transitions have no Gaussian density")
    # Retain Flow-GRPO's mean over latent coordinates, not its CPS MSE proxy.
    z=(sample.detach().float()-mean.float())/tau
    return (-.5*z.square()-tau.log()-.5*math.log(2*math.pi)).flatten(1).mean(1)


def grpo_loss(logprob, old_logprob, advantage, clip_range=1e-4):
    ratio=(logprob-old_logprob.detach()).exp()
    a=advantage.detach().clamp(-5,5)
    return torch.maximum(-a*ratio,-a*ratio.clamp(1-clip_range,1+clip_range)).mean(),ratio


def gaussian_kl(mean, reference_mean, tau):
    return (.5*((mean-reference_mean.detach())/tau).square()).flatten(1).mean(1).mean()
