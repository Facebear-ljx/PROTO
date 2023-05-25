from typing import Tuple

import jax.numpy as jnp
import jax
from functools import partial

from common import Batch, InfoDict, Model, Params, PRNGKey
from ensemble import subsample_ensemble


def gumbel_rescale_loss(diff, alpha, args=None):
    """ Gumbel loss J: E[e^x - x - 1]. For stability to outliers, we scale the gradients with the max value over a batch
    and optionally clip the exponent. This has the effect of training with an adaptive lr.
    """
    z = diff/alpha
    if args.max_clip is not None:
        z = jnp.minimum(z, args.max_clip) # clip max value
    max_z = jnp.max(z, axis=0)
    max_z = jnp.where(max_z < -1.0, -1.0, max_z)
    max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    loss = jnp.exp(z - max_z) - z*jnp.exp(-max_z) - jnp.exp(-max_z)  # scale by e^max_z
    return loss

def gumbel_log_loss(diff, alpha=1.0):
    """ Gumbel loss J: E[e^x - x - 1]. We can calculate the log of Gumbel loss for stability, i.e. Log(J + 1)
    log_gumbel_loss: log((e^x - x - 1).mean() + 1)
    """
    diff = diff
    x = diff/alpha
    grad = grad_gumbel(x, alpha)
    # use analytic gradients to improve stability
    loss = jax.lax.stop_gradient(grad) * x
    return loss

def grad_gumbel(x, alpha, clip_max=7):
    """Calculate grads of log gumbel_loss: (e^x - 1)/[(e^x - x - 1).mean() + 1]
    We add e^-a to both numerator and denominator to get: (e^(x-a) - e^(-a))/[(e^(x-a) - xe^(-a)).mean()]
    """
    # clip inputs to grad in [-10, 10] to improve stability (gradient clipping)
    x = jnp.minimum(x, clip_max)  # jnp.clip(x, a_min=-10, a_max=10)

    # calculate an offset `a` to prevent overflow issues
    x_max = jnp.max(x, axis=0)
    # choose `a` as max(x_max, -1) as its possible for x_max to be very small and we want the offset to be reasonable
    x_max = jnp.where(x_max < -1, -1, x_max)

    # keep track of original x
    x_orig = x
    # offsetted x
    x1 = x - x_max

    grad = (jnp.exp(x1) - jnp.exp(-x_max)) / \
        (jnp.mean(jnp.exp(x1) - x_orig * jnp.exp(-x_max), axis=0, keepdims=True))
    return grad

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def update_v(critic: Model, value: Model, batch: Batch,
             expectile: float, loss_temp: float, double: bool, vanilla: bool, key: PRNGKey, args) -> Tuple[Model, InfoDict]:
    actions = batch.actions

    rng1, rng2 = jax.random.split(key)
    obs = batch.observations
    acts = batch.actions

    qs = critic(obs, acts)
    if double:
        q = qs.min(axis=0)
    else:
        q = qs[0, :]

    # zero_acts = jnp.zeros_like(batch.actions)
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # detach the action part so the value network is a V function instead of a Q function
        v = value.apply({'params': value_params}, obs)

        if vanilla:
            value_loss = expectile_loss(q - v, expectile).mean()
        else:
            if args.log_loss:
                value_loss = gumbel_log_loss(q - v, alpha=loss_temp, args=args).mean()
            else:
                value_loss = gumbel_rescale_loss(q - v, alpha=loss_temp, args=args).mean()
            if args.bc_pretrain:
                value_loss = ((q - v) ** 2).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_v_online(critic: Model, value: Model, target_online_actor: Model, batch: Batch,
             expectile: float, loss_temp: float, double: bool, vanilla: bool, upon_bc: bool, key: PRNGKey, args) -> Tuple[Model, InfoDict]:

    obs = batch.observations

    # online sample pi_k
    if upon_bc:
        acts_pi= batch.actions
    else:
        dist_pi_k = target_online_actor(obs)
        acts_pi = dist_pi_k.sample(seed=key)

    q1, q2 = critic(obs, acts_pi)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)

        if vanilla:
            value_loss = expectile_loss(q - v, expectile).mean()
        else:
            if args.log_loss:
                value_loss = gumbel_log_loss(q - v, alpha=loss_temp, args=args).mean()
            else:
                value_loss = gumbel_rescale_loss(q - v, alpha=loss_temp, args=args).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'temparature': loss_temp
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, offline_actor: Model, behavior: Model, batch: Batch,
             discount: float, double: bool, key: PRNGKey, loss_temp: float, offline: bool, args) -> Tuple[Model, InfoDict]:
    # zero_acts = jnp.zeros_like(batch.actions)
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        acts = batch.actions
        qs = critic.apply({'params': critic_params}, batch.observations, acts)

        v = target_value(batch.observations)

        def mse_loss(q, q_target, *args):
            loss_dict = {}

            x = q-q_target
            loss = x ** 2
            loss_dict['critic_loss'] = loss.mean()

            return loss.mean(), loss_dict

        critic_loss = mse_loss

        if double:
            critic_loss, loss_dict = critic_loss(qs, target_q, v, loss_temp)
        else:
            critic_loss, loss_dict = critic_loss(qs[0, :], target_q,  v, loss_temp)

        loss_dict.update({
            'q1': qs[0, :].mean(),
            'q2': qs[1, :].mean()
        })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_q_online(critic: Model, target_critic: Model, target_online_actor: Model, online_actor: Model, behavior: Model, offline_actor: Model, batch: Batch,
             discount: float, double: bool, key: PRNGKey, temp: float, temp_online: float, log_alpha: float, args) -> Tuple[Model, InfoDict]:
    
    rng1, rng2, rng3, rng4, rng5, rng6, rng = jax.random.split(key, num=7)
    dist = online_actor(batch.next_observations, rngs={"dropout": rng1})
    dist_k = target_online_actor(batch.next_observations, rngs={"dropout": rng2})

    next_acts = dist.sample(seed=rng3, temperature=0.)
    noise = (jax.random.normal(key=rng6, shape=(next_acts.shape[0], next_acts.shape[1])) * 0.2).clip(-0.5, 0.5)
    next_acts += noise
    next_acts = next_acts

    next_acts_k = dist_k.sample(seed=rng6, temperature=0.)  # no random here.

    value_reg = temp_online * ((next_acts - next_acts_k) ** 2).mean() if args.entropy_backup else 0

    if double:
        target_qs = target_critic(batch.next_observations, next_acts_k, rngs={"dropout": rng4})
    else:
        # random sample a target q value to update (used for REDQ)
        target_params = subsample_ensemble(rng5, target_critic.params, num_sample=1, num_qs=2)
        target_qs = target_critic.apply({"params": target_params}, batch.next_observations, next_acts, rngs={"dropout": rng4})
    target_q = batch.rewards + discount * batch.masks * (target_qs.min(axis=0) - value_reg)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        acts = batch.actions
        qs = critic.apply({'params': critic_params}, batch.observations, acts)

        def mse_loss(q, q_target, *args):
            loss_dict = {}

            x = q-q_target
            loss = x ** 2
            loss_dict['critic_loss'] = loss.mean()

            return loss.mean(), loss_dict

        critic_loss = mse_loss

        critic_loss, loss_dict = critic_loss(qs, target_q)

        loss_dict.update({
            'q1': qs[0, :].mean(),
            'q2': qs[1, :].mean()
        })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, rng, info


    
def grad_norm(model, params, obs, action, lambda_=10):

    @partial(jax.vmap, in_axes=(0, 0))
    @partial(jax.jacrev, argnums=1)
    def input_grad_fn(obs, action):
        return model.apply({'params': params}, obs, action)

    def grad_pen_fn(grad):
        # We use gradient penalties inspired from WGAN-LP loss which penalizes grad_norm > 1
        penalty = jnp.maximum(jnp.linalg.norm(grad1, axis=-1) - 1, 0)**2
        return penalty

    grad1, grad2 = input_grad_fn(obs, action)

    return grad_pen_fn(grad1), grad_pen_fn(grad2)


def huber_loss(x, delta: float = 1.):
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear
