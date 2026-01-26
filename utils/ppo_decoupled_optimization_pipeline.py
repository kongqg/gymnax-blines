from functools import partial
import optax
import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple
from collections import defaultdict
import flax
from flax.training.train_state import TrainState
import numpy as np
import tqdm
import gymnax
import wandb


class BatchManager:
    def __init__(
            self,
            discount: float,
            gae_lambda: float,
            n_steps: int,
            num_envs: int,
            action_size,
            state_space,
            tau: float,
            algo,
    ):
        self.num_envs = num_envs
        self.action_size = action_size
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.tau = tau
        self.algo = algo

        try:
            temp = state_space.shape[0]
            self.state_shape = state_space.shape
        except Exception:
            self.state_shape = [state_space]
        self.reset()

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "states": jnp.empty(
                (self.n_steps, self.num_envs, *self.state_shape),
                dtype=jnp.float32,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, *self.action_size),
            ),
            "rewards": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, state, action, reward, done, log_pi, value):
        return {
            "states": buffer["states"].at[buffer["_p"]].set(state),
            "actions": buffer["actions"].at[buffer["_p"]].set(action),
            "rewards": buffer["rewards"].at[buffer["_p"]].set(reward.squeeze()),
            "dones": buffer["dones"].at[buffer["_p"]].set(done.squeeze()),
            "log_pis_old": buffer["log_pis_old"].at[buffer["_p"]].set(log_pi),
            "values_old": buffer["values_old"].at[buffer["_p"]].set(value),
            "_p": (buffer["_p"] + 1) % self.n_steps,
        }

    @partial(jax.jit, static_argnums=0)
    def get(self, buffer):
        if self.algo == "ppo":
            gae, target = self.calculate_gae(
                value=buffer["values_old"],
                reward=buffer["rewards"],
                done=buffer["dones"],
            )
        elif self.algo == "dhvl":
            gae, target = self.calculate_expectile_gae(
                value=buffer["values_old"],
                reward=buffer["rewards"],
                done=buffer["dones"],
            )
        batch = (
            buffer["states"][:-1],
            buffer["actions"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
        )
        return batch

    @partial(jax.jit, static_argnums=0)
    def calculate_expectile_gae(
            self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        v_all = value.astype(jnp.float32)  # [T+1,B]
        T = v_all.shape[0] - 1

        v_old = v_all[:T]  # V_t     [T,B]
        v_tp1 = v_all[1:T + 1]  # V_{t+1} [T,B]
        last_value = v_all[-1]  # V_T     [B]

        r = reward.astype(jnp.float32)[:T]  # r_t [T,B]

        d = (1.0 - done.astype(jnp.float32))[:T]  # [T,B]

        def step(G_next, xs):
            r_t, d_t, v_tp1_t = xs  # each [B]
            u = G_next - v_tp1_t
            pos = jnp.maximum(u, 0.0)  # (u)_+
            neg = jnp.maximum(-u, 0.0)  # (u)_-
            lam = self.gae_lambda / self.tau
            soft = v_tp1_t + lam * (self.tau * pos - (1.0 - self.tau) * neg)
            G_t = r_t + self.discount * d_t * soft
            return G_t, G_t

        _, G_rev = jax.lax.scan(
            step,
            last_value,
            (r[::-1], d[::-1], v_tp1[::-1]),
        )

        returns = G_rev[::-1]  # [T,B]
        adv = jax.lax.stop_gradient(returns - v_old)
        return adv, returns

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
            self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]


class RolloutManager(object):
    def __init__(self, model, env_name, env_kwargs, env_params):
        # Setup functionalities for vectorized batch rollout
        self.env_name = env_name
        self.env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_size = self.env.action_space(self.env_params).shape
        self.apply_fn = model.apply
        self.select_action = self.select_action_ppo

    @partial(jax.jit, static_argnums=0)
    def select_action_ppo(
            self,
            train_state: TrainState,
            obs: jnp.ndarray,
            rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        value, pi = policy(train_state.apply_fn, train_state.params, obs, rng)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value[:, 0], rng

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def batch_evaluate(self, rng_input, train_state, num_envs):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs))

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action, _, _, rng = self.select_action(train_state, obs, rng_net)
            next_o, next_s, reward, done, _ = self.batch_step(
                jax.random.split(rng_step, num_envs),
                state,
                action.squeeze(),
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry, y = [
                next_o,
                next_s,
                train_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ], [new_valid_mask]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                train_state,
                rng_episode,
                jnp.array(num_envs * [0.0]),
                jnp.array(num_envs * [1.0]),
            ],
            (),
            self.env_params.max_steps_in_episode,
        )

        cum_return = carry_out[-2].squeeze()
        return jnp.mean(cum_return)


@partial(jax.jit, static_argnums=0)
def policy(
        apply_fn: Callable[..., Any],
        params: flax.core.frozen_dict.FrozenDict,
        obs: jnp.ndarray,
        rng,
):
    value, pi = apply_fn(params, obs, rng)
    return value, pi


def train_ppo(rng, config, model, params, mle_log):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    num_total_epochs = int(config.num_train_steps // config.num_train_envs + 1)
    num_steps_warm_up = int(config.num_train_steps * config.lr_warmup)
    schedule_fn = optax.linear_schedule(
        init_value=-float(config.lr_begin),
        end_value=-float(config.lr_end),
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutManager(
        model, config.env_name, config.env_kwargs, config.env_params
    )

    batch_manager = BatchManager(
        discount=config.gamma,
        gae_lambda=config.gae_lambda,
        n_steps=config.n_steps + 1,
        num_envs=config.num_train_envs,
        action_size=rollout_manager.action_size,
        state_space=rollout_manager.observation_space,
        tau=config.tau,
        algo=config.algo,
    )

    @partial(jax.jit, static_argnums=5)
    def get_transition(
            train_state: TrainState,
            obs: jnp.ndarray,
            state: dict,
            batch,
            rng: jax.random.PRNGKey,
            num_train_envs: int,
    ):
        action, log_pi, value, new_key = rollout_manager.select_action(
            train_state, obs, rng
        )
        # print(action.shape)
        new_key, key_step = jax.random.split(new_key)
        b_rng = jax.random.split(key_step, num_train_envs)
        # Automatic env resetting in gymnax step!
        next_obs, next_state, reward, done, _ = rollout_manager.batch_step(
            b_rng, state, action
        )
        batch = batch_manager.append(
            batch, obs, action, reward, done, log_pi, value
        )
        return train_state, next_obs, next_state, batch, new_key

    batch = batch_manager.reset()

    rng, rng_step, rng_reset, rng_eval, rng_update = jax.random.split(rng, 5)
    obs, state = rollout_manager.batch_reset(
        jax.random.split(rng_reset, config.num_train_envs)
    )

    total_steps = 0
    log_steps, log_return = [], []

    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="PPO", leave=True)
    for step in t:
        train_state, obs, state, batch, rng_step = get_transition(
            train_state,
            obs,
            state,
            batch,
            rng_step,
            config.num_train_envs,
        )
        total_steps += config.num_train_envs
        if step % (config.n_steps + 1) == 0:
            metric_dict, train_state, rng_update = update(
                train_state=train_state,
                batch=batch_manager.get(batch),
                num_envs=config.num_train_envs,
                n_steps=config.n_steps,
                n_minibatch=config.n_minibatch,
                actor_update_iter=config.actor_update_iter,
                value_update_iter=config.value_update_iter,
                clip_eps=config.clip_eps,
                entropy_coeff=config.entropy_coeff,
                critic_coeff=config.critic_coeff,
                rng=rng_update,
                tau=config.tau,
                algo=config.algo,
            )

            t_dict = {f"train/{k}": v for k, v in metric_dict.items()}
            t_dict["step"] = total_steps
            wandb.log(t_dict, step=total_steps)
            batch = batch_manager.reset()

        if (step + 1) % config.evaluate_every_epochs == 0:
            rng, rng_eval = jax.random.split(rng)
            rewards = rollout_manager.batch_evaluate(
                rng_eval,
                train_state,
                config.num_test_rollouts,
            )
            log_steps.append(total_steps)
            log_return.append(rewards)
            t.set_description(f"R: {str(rewards)}")
            t.refresh()

            if mle_log is not None:
                mle_log.update(
                    {"num_steps": total_steps},
                    {"return": rewards},
                    model=train_state.params,
                    save=True,
                )
            m_dict = {"eval/episode_return": float(rewards)}
            wandb.log(m_dict, step=total_steps)

    return (
        log_steps,
        log_return,
        train_state.params,
    )


@jax.jit
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def expectile_actor_loss_helper(tau, adv):
    weight = jnp.where(adv > 0, 2.0 * tau, 2.0 * (1.0 - tau))
    return weight


def expectile_value_loss_helper(target, value, value_clipped, tau):
    delta1 = target - value
    delta2 = target - value_clipped

    w1 = jnp.where(delta1 > 0, 2.0 * tau, 2.0 * (1.0 - tau))
    w2 = jnp.where(delta2 > 0, 2.0 * tau, 2.0 * (1.0 - tau))

    return w1, w2


def loss_actor(
        params_model: flax.core.frozen_dict.FrozenDict,
        apply_fn: Callable[..., Any],
        obs: jnp.ndarray,
        log_pi_old: jnp.ndarray,
        gae: jnp.ndarray,
        action: jnp.ndarray,
        clip_eps: float,
        entropy_coeff: float,
        tau: float,
        algo: str
) -> jnp.ndarray:
    """Actor-only loss function."""
    _, pi = apply_fn(params_model, obs, rng=None)
    log_prob = pi.log_prob(action[..., -1])

    ratio = jnp.exp(log_prob - log_pi_old)
    gae_mean = gae.mean()
    gae_norm = (gae - gae_mean) / (gae.std() + 1e-8)

    loss_actor1 = ratio * gae_norm
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm

    if algo == "dhvl":
        actor_weight = expectile_actor_loss_helper(tau, gae)
        loss_actor = -jnp.minimum(actor_weight * loss_actor1, actor_weight * loss_actor2)
    else:
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)

    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    total_loss = loss_actor - entropy_coeff * entropy

    return total_loss, (loss_actor, entropy, gae_mean)


def loss_critic(
        params_model: flax.core.frozen_dict.FrozenDict,
        apply_fn: Callable[..., Any],
        obs: jnp.ndarray,
        target: jnp.ndarray,
        value_old: jnp.ndarray,
        clip_eps: float,
        critic_coeff: float,
        tau: float,
        algo: str
) -> jnp.ndarray:
    """Critic-only loss function."""
    value_pred, _ = apply_fn(params_model, obs, rng=None)
    value_pred = value_pred[:, 0]

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)

    if algo == "dhvl":
        w1, w2 = expectile_value_loss_helper(target, value_pred, value_pred_clipped, tau)
        value_loss = 0.5 * jnp.maximum(w1 * value_losses, w2 * value_losses_clipped).mean()
    else:
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    total_loss = critic_coeff * value_loss
    return total_loss, (value_loss, value_pred.mean(), target.mean())


def update(
        train_state: TrainState,
        batch: Tuple,
        num_envs: int,
        n_steps: int,
        n_minibatch: int,
        actor_update_iter: int,
        value_update_iter: int,
        clip_eps: float,
        entropy_coeff: float,
        critic_coeff: float,
        rng: jax.random.PRNGKey,
        tau: float,
        algo,
):
    """Perform decoupled updates for Actor and Critic."""
    obs, action, log_pi_old, value, target, gae = batch
    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch
    idxes = jnp.arange(size_batch)
    avg_metrics_dict = defaultdict(float)


    # --- Actor Updates ---
    for _ in range(actor_update_iter):
        rng, rng_perm = jax.random.split(rng)
        idxes = jax.random.permutation(rng_perm, idxes)
        idxes_list = [
            idxes[start: start + size_minibatch]
            for start in jnp.arange(0, size_batch, size_minibatch)
        ]

        train_state, actor_metrics = update_epoch(
            train_state,
            idxes_list,
            flatten_dims(obs),
            flatten_dims(action),
            flatten_dims(log_pi_old),
            flatten_dims(value),
            jnp.array(flatten_dims(target)),
            jnp.array(flatten_dims(gae)),
            clip_eps,
            entropy_coeff,
            critic_coeff,
            tau,
            algo,
            "actor"
        )

        loss_actor_val, entropy_val, gae_val = actor_metrics
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor_val)
        avg_metrics_dict["entropy"] += np.asarray(entropy_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)

    # --- Critic Updates ---
    for _ in range(value_update_iter):
        rng, rng_perm = jax.random.split(rng)
        idxes = jax.random.permutation(rng_perm, idxes)
        idxes_list = [
            idxes[start: start + size_minibatch]
            for start in jnp.arange(0, size_batch, size_minibatch)
        ]

        train_state, critic_metrics = update_epoch(
            train_state,
            idxes_list,
            flatten_dims(obs),
            flatten_dims(action),
            flatten_dims(log_pi_old),
            flatten_dims(value),
            jnp.array(flatten_dims(target)),
            jnp.array(flatten_dims(gae)),
            clip_eps,
            entropy_coeff,
            critic_coeff,
            tau,
            algo,
            "critic"
        )


        value_loss_val, value_pred_val, target_val = critic_metrics
        avg_metrics_dict["value_loss"] += np.asarray(value_loss_val)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred_val)
        avg_metrics_dict["target"] += np.asarray(target_val)

    # Final averaging for logs
    for k in avg_metrics_dict:
        count = actor_update_iter if k in ["actor_loss", "entropy", "gae"] else value_update_iter
        avg_metrics_dict[k] /= count

    return avg_metrics_dict, train_state, rng


@partial(jax.jit, static_argnames=("algo", "mode"))
def update_epoch(
        train_state: TrainState,
        idxes: jnp.ndarray,
        obs,
        action,
        log_pi_old,
        value,
        target,
        gae,
        clip_eps: float,
        entropy_coeff: float,
        critic_coeff: float,
        tau: float,
        algo: str,
        mode: str
):
    """JITed epoch update for either Actor or Critic."""
    last_aux = None
    for idx in idxes:
        batch_obs = obs[idx]
        batch_target = target[idx]
        batch_value = value[idx]
        batch_log_pi = log_pi_old[idx]
        batch_gae = gae[idx]
        batch_action = jnp.expand_dims(action[idx], -1)

        if mode == "actor":
            grad_fn = jax.value_and_grad(loss_actor, has_aux=True)
            _, grads = grad_fn(
                train_state.params,
                train_state.apply_fn,
                obs=batch_obs,
                log_pi_old=batch_log_pi,
                gae=batch_gae,
                action=batch_action,
                clip_eps=clip_eps,
                entropy_coeff=entropy_coeff,
                tau=tau,
                algo=algo
            )
            _, last_aux = loss_actor(
                train_state.params, train_state.apply_fn, batch_obs,
                batch_log_pi, batch_gae, batch_action, clip_eps, entropy_coeff, tau, algo
            )
        else:  # mode == "critic"
            grad_fn = jax.value_and_grad(loss_critic, has_aux=True)
            _, grads = grad_fn(
                train_state.params,
                train_state.apply_fn,
                obs=batch_obs,
                target=batch_target,
                value_old=batch_value,
                clip_eps=clip_eps,
                critic_coeff=critic_coeff,
                tau=tau,
                algo=algo
            )
            _, last_aux = loss_critic(
                train_state.params, train_state.apply_fn, batch_obs,
                batch_target, batch_value, clip_eps, critic_coeff, tau, algo
            )

        train_state = train_state.apply_gradients(grads=grads)

    return train_state, last_aux