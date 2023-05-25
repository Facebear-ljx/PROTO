from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, offline_actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float, double: bool, args) -> Tuple[Model, InfoDict]:
    # zero_acts = jnp.zeros_like(batch.actions)
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    exp_a = jnp.exp((q - v) / temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = offline_actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        if args.bc_pretrain:
            actor_loss = - log_probs.mean()
        else:
            actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_offline_actor, info = offline_actor.apply_gradient(actor_loss_fn)

    return new_offline_actor, info


def update_online(key: PRNGKey, online_actor: Model, offline_actor: Model, critic: Model, target_actor: Model, behavior: Model,
           batch: Batch, temp_offline: float, temp_online: float, ratio: float, double: bool, log_alpha: float, args) -> Tuple[Model, InfoDict]:
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(key, num=5)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist_pi = online_actor.apply({'params': actor_params},
                                        batch.observations,
                                        training=True,
                                        rngs={'dropout': rng1})
        # online samples
        acts_pi = dist_pi.sample(seed=rng2)
        
        # q1, q2 = critic(batch.observations, acts_pi)
        qs = critic.apply({'params': critic.params}, batch.observations, acts_pi)
        q = qs.mean(axis=0)
            
        log_pi_probs = dist_pi.log_prob(acts_pi)
        
        dist_mu = behavior(batch.observations, rngs={'dropout': rng3})
        log_mu_probs = dist_mu.log_prob(acts_pi)
        
        dist_pi_0 = offline_actor(batch.observations, rngs={'dropout': rng4})
        log_pi_0_probs = dist_pi_0.log_prob(acts_pi)
        
        dist_pi_k = target_actor(batch.observations, rngs={'dropout': rng5})
        log_pi_k_probs = dist_pi_k.log_prob(acts_pi)
        
        a_0 = dist_pi_0.sample(seed=rng5)
        
        if args.sac:
            actor_reg = log_alpha() * log_pi_probs + temp_online * (log_pi_probs - log_pi_k_probs)
        else:
            actor_reg = temp_online * (log_pi_probs - log_pi_k_probs)
        actor_loss = (-q + actor_reg).mean()
        # online actor loss

        return actor_loss, {'actor_loss': actor_loss,
                            'bc_loss': -log_mu_probs.mean(),
                            'trust_region_pi0': -log_pi_0_probs.mean(),
                            'entropy': -log_pi_probs.mean(),
                            'trust_region_pik': -log_pi_k_probs.mean(),
                            'mse_loss': ((acts_pi - a_0)**2).mean(),
                            'mse_loss_mu': ((acts_pi - batch.actions)**2).mean()}

    new_actor, info = online_actor.apply_gradient(actor_loss_fn)

    return new_actor, info




def update_alpha(key: PRNGKey, entropy: float, log_alpha: float, target_entropy: float):
    
    def temperature_loss_fn(temp_params):
        temperature = log_alpha.apply({'params': temp_params})
        alpha_loss = temperature * (entropy - target_entropy).mean()
        return alpha_loss, {'alpha_loss': alpha_loss, 'alpha': temperature, 'target_entropy': target_entropy}
    
    new_log_alpha, info = log_alpha.apply_gradient(temperature_loss_fn)
    
    return new_log_alpha, info


def update_mu(key: PRNGKey, behavior: Model, batch: Batch) -> Tuple[Model, InfoDict]:

    def bc_loss_fn(behavior_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = behavior.apply({'params': behavior_params},
                                batch.observations,
                                training=True,
                                rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        bc_loss = (-log_probs).mean()

        return bc_loss, {'bc_loss': bc_loss}

    new_behavior, info = behavior.apply_gradient(bc_loss_fn)

    return new_behavior, info