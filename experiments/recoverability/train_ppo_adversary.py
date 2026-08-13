#!/usr/bin/env python3
import argparse
import csv
import os
import time

import numpy as np
from integrator_chain_env import IntegratorChainEnv
from metrics import control_effort, recovery_time, time_to_violation

from safe_control_gym.controllers.ppo.ppo import PPO
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.utils.registration import make

REWARD_MODES = [
    'constraint_seeking',
    'horizon_exhaustion',
    'intervention_maximizing',
    'chattering',
    'actuator_saturation',
]

EVAL_COST_FUNCTION_CHOICES = [
    Cost_Function.ONE_STEP_COST.value,
    Cost_Function.CONSTANT_COST.value,
    Cost_Function.REGULARIZED_COST.value,
    Cost_Function.BLENDED_COST.value,
    Cost_Function.LQR_COST.value,
    Cost_Function.PRECOMPUTED_COST.value,
    Cost_Function.LEARNED_COST.value,
]


def build_safety_filter(env_func, args, cost_function, horizon, mpsc_cost_horizon):
    return make(
        'nl_mpsc',
        env_func,
        horizon=horizon,
        q_lin=[args.q_lin] * args.r,
        r_lin=[args.r_lin],
        cost_function=cost_function,
        mpsc_cost_horizon=mpsc_cost_horizon,
        use_acados=True,
        use_terminal_set=args.use_terminal_set,
        terminal_set_scale=args.terminal_set_scale,
        terminal_cost_weight=args.terminal_cost_weight,
        soften_constraints=True,
        slack_cost=args.slack_cost,
        max_w=args.max_w,
        regularization_weight=args.regularization_weight,
        blended_cost_alpha=args.blended_cost_alpha,
    )


def build_reward_fn(mode, x_limit, umax):
    mode = str(mode)
    prev_u = {'value': 0.0}

    def reward_fn(state, u, info):
        x = float(state[0])
        xdot = float(state[1]) if len(state) > 1 else 0.0
        margin = float(info.get('distance_to_constraint', x_limit - abs(x)))
        margin_norm = max(margin / max(x_limit, 1e-6), -2.0)
        sat_ratio = abs(float(u)) / max(umax, 1e-6)

        if mode == 'constraint_seeking':
            return abs(x) + 0.2 * abs(xdot)
        if mode == 'horizon_exhaustion':
            return abs(xdot) - 0.2 * margin_norm
        if mode == 'intervention_maximizing':
            return 0.8 * abs(xdot) + 0.6 * sat_ratio + 0.2 * abs(x)
        if mode == 'chattering':
            du = abs(float(u) - prev_u['value'])
            prev_u['value'] = float(u)
            return 1.2 * du + 0.2 * abs(xdot)
        if mode == 'actuator_saturation':
            return 1.5 * sat_ratio + 0.2 * abs(xdot) - 0.1 * margin_norm

        raise ValueError(f'Unknown reward mode: {mode}')

    return reward_fn


def make_env_func(r, reward_fn, umax=1.0, x_limit=2.0, v_limit=None, dt=0.05, horizon=500, init_scale=0.05):
    def env_fn(seed=None, **_kwargs):
        return IntegratorChainEnv(
            r=r,
            dt=dt,
            umax=umax,
            x_limit=x_limit,
            v_limit=v_limit,
            horizon=horizon,
            seed=seed,
            init_scale=init_scale,
            reward_fn=reward_fn,
        )

    return env_fn


def _append_summary_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(args):
    run_name = args.run_name or f'r{args.r}_{args.reward_mode}_h{args.mpsc_horizon}_u{args.umax}_mw{args.max_w}_seed{args.seed}'
    outdir = os.path.join(args.output_root, run_name)
    os.makedirs(outdir, exist_ok=True)

    reward_fn = build_reward_fn(args.reward_mode, x_limit=args.x_limit, umax=args.umax)
    env_func = make_env_func(
        r=args.r,
        reward_fn=reward_fn,
        umax=args.umax,
        x_limit=args.x_limit,
        v_limit=args.v_limit,
        dt=args.dt,
        horizon=args.env_horizon,
        init_scale=args.init_scale,
    )

    rollout_steps = max(args.rollout_steps, 20 * args.r)
    num_workers = max(1, int(args.num_workers))

    ppo_kwargs = dict(
        hidden_dim=args.hidden_dim,
        activation='tanh',
        norm_obs=False,
        norm_reward=False,
        clip_obs=10,
        clip_reward=10,
        gamma=0.99,
        use_gae=False,
        gae_lambda=0.95,
        use_clipped_value=False,
        clip_param=0.2,
        target_kl=0.01,
        entropy_coef=0.01,
        opt_epochs=args.opt_epochs,
        mini_batch_size=args.mini_batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_grad_norm=0.5,
        max_env_steps=args.max_env_steps,
        num_workers=num_workers,
        rollout_batch_size=num_workers,
        rollout_steps=rollout_steps,
        deque_size=4,
        eval_batch_size=4,
        log_interval=args.log_interval,
        save_interval=0,
        num_checkpoints=0,
        eval_interval=0,
        eval_save_best=False,
        tensorboard=False,
        adv_reward_temperature=50,
    )

    start = time.time()
    checkpoint_path = args.checkpoint_path or os.path.join(outdir, 'model_latest.pt')
    ctrl = PPO(
        env_func,
        training=not args.eval_from_checkpoint,
        checkpoint_path=checkpoint_path,
        output_dir=outdir,
        use_gpu=False,
        seed=args.seed,
        **ppo_kwargs,
    )
    ctrl.reset()

    if args.eval_from_checkpoint:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Missing checkpoint for eval-only run: {checkpoint_path}')
        ctrl.model = ctrl.get_prior(ctrl.env, ctrl.prior_info)
        ctrl.load(checkpoint_path)
        obs0, _ = ctrl.env.reset()
        ctrl.obs = ctrl.obs_normalizer(obs0)

    # Backward-compatible defaults: if train/eval horizons are not explicitly
    # provided, keep using the manifest mpsc_horizon value.
    train_mpsc_horizon = int(args.train_mpsc_horizon) if args.train_mpsc_horizon is not None else int(args.mpsc_horizon)
    eval_mpsc_horizon = int(args.eval_mpsc_horizon) if args.eval_mpsc_horizon is not None else int(args.mpsc_horizon)

    # Backward-compatible default keeps one-step training objective.
    train_cost_function = Cost_Function(args.train_sf_cost_function)
    train_mpsc_cost_horizon = int(args.train_mpsc_cost_horizon)
    if not args.eval_from_checkpoint:
        ctrl.safety_filter = build_safety_filter(
            env_func,
            args,
            train_cost_function,
            train_mpsc_horizon,
            train_mpsc_cost_horizon,
        )
        ctrl.safety_filter.reset()

    print(f'[RUN] {run_name}')
    print(f'[CFG] r={args.r}, reward={args.reward_mode}, train_mpsc_h={train_mpsc_horizon}, eval_mpsc_h={eval_mpsc_horizon}, train_cost_fn={train_cost_function.value}, train_mpsc_cost_h={train_mpsc_cost_horizon}, eval_cost_fn={args.eval_sf_cost_function}, eval_mpsc_cost_h={args.mpsc_cost_horizon}, umax={args.umax}, v_limit={args.v_limit}, max_w={args.max_w}, steps={args.max_env_steps}, seed={args.seed}')
    if not args.eval_from_checkpoint:
        ctrl.learn()

    # Evaluation-time NL-MPSC can use a different cost function/cost horizon.
    eval_cost_function = Cost_Function(args.eval_sf_cost_function)
    ctrl.safety_filter = build_safety_filter(
        env_func,
        args,
        eval_cost_function,
        eval_mpsc_horizon,
        args.mpsc_cost_horizon,
    )
    # Set the underlying controller for cost functions that need it (e.g., PRECOMPUTED_COST).
    ctrl.safety_filter.cost_function.uncertified_controller = ctrl
    ctrl.safety_filter.reset()

    eval_env = IntegratorChainEnv(
        r=args.r,
        dt=args.dt,
        umax=args.umax,
        x_limit=args.x_limit,
        v_limit=args.v_limit,
        horizon=args.env_horizon,
        seed=args.seed,
        init_scale=args.init_scale,
        reward_fn=reward_fn,
    )
    res = ctrl.run(env=eval_env, n_episodes=args.eval_episodes)

    metric_env = IntegratorChainEnv(
        r=args.r,
        dt=args.dt,
        umax=args.umax,
        x_limit=args.x_limit,
        v_limit=args.v_limit,
        horizon=args.env_horizon,
        seed=args.seed,
        init_scale=args.init_scale,
        reward_fn=reward_fn,
    )
    obs, info = ctrl.env_reset(metric_env, True)
    obs = ctrl.obs_normalizer(obs)
    positions = []
    actions = []
    terminations = 0
    truncations = 0
    corrections = []
    violating_episodes = 0
    recovered_episodes = 0
    episode_recovery_times = []
    episodes = 0
    episode_positions = [float(obs[0])]
    while episodes < args.eval_episodes:
        action = ctrl.select_action(obs=obs, info=info)
        uncert = metric_env.denormalize_action(action)
        cert, _ = ctrl.safety_filter.certify_action(np.atleast_1d(np.squeeze(obs))[:metric_env.symbolic.nx], uncert, info)
        corrections.append(float(np.linalg.norm(np.atleast_1d(uncert) - np.atleast_1d(cert))))
        env_action = metric_env.normalize_action(cert)
        step_out = metric_env.step(env_action)
        if len(step_out) == 5:
            next_obs, _, terminated, truncated, info = step_out
        else:
            next_obs, _, done, info = step_out
            terminated = bool(info.get('terminated', done))
            truncated = bool(info.get('truncated', False))
        positions.append(float(next_obs[0]))
        episode_positions.append(float(next_obs[0]))
        actions.append(float(np.asarray(env_action).reshape(-1)[0]))
        terminations += int(terminated)
        truncations += int(truncated)
        done = bool(terminated or truncated)
        if done:
            ep_ttv = time_to_violation(episode_positions, metric_env.x_limit)
            if ep_ttv is not None:
                violating_episodes += 1
                ep_recovery = recovery_time(
                    episode_positions,
                    ep_ttv,
                    nominal_threshold=args.recovery_nominal_threshold,
                    consecutive=args.recovery_consecutive,
                )
                if ep_recovery is not None:
                    recovered_episodes += 1
                    episode_recovery_times.append(float(ep_recovery))
            episodes += 1
            obs, info = ctrl.env_reset(metric_env, True)
            episode_positions = [float(obs[0])]
        else:
            obs = next_obs
        obs = ctrl.obs_normalizer(obs)

    ttv = time_to_violation(positions, metric_env.x_limit)
    intervention_freq = float(np.mean(np.asarray(corrections) > 1e-8)) if corrections else 0.0
    intervention_mag = float(np.mean(corrections)) if corrections else 0.0
    recovery_time_mean = float(np.mean(episode_recovery_times)) if episode_recovery_times else -1.0
    recovery_time_median = float(np.median(episode_recovery_times)) if episode_recovery_times else -1.0
    recovered_after_violation_frac = (
        float(recovered_episodes / violating_episodes) if violating_episodes > 0 else -1.0
    )

    row = {
        'run_name': run_name,
        'reward_mode': args.reward_mode,
        'r': args.r,
        'seed': args.seed,
        'max_env_steps': args.max_env_steps,
        'env_horizon': args.env_horizon,
        'mpsc_horizon': eval_mpsc_horizon,
        'train_mpsc_horizon': train_mpsc_horizon,
        'train_cost_function': train_cost_function.value,
        'train_mpsc_cost_horizon': train_mpsc_cost_horizon,
        'eval_mpsc_horizon': eval_mpsc_horizon,
        'eval_cost_function': eval_cost_function.value,
        'mpsc_cost_horizon': args.mpsc_cost_horizon,
        'max_w': args.max_w,
        'umax': args.umax,
        'x_limit': args.x_limit,
        'v_limit': '' if args.v_limit is None else args.v_limit,
        'ep_return_mean': float(np.mean(res['ep_returns'])),
        'ep_length_mean': float(np.mean(res['ep_lengths'])),
        'time_to_violation': -1 if ttv is None else int(ttv),
        'recovery_time_mean': recovery_time_mean,
        'recovery_time_median': recovery_time_median,
        'violating_episodes': int(violating_episodes),
        'recovered_episodes': int(recovered_episodes),
        'recovered_after_violation_frac': recovered_after_violation_frac,
        'terminations': int(terminations),
        'truncations': int(truncations),
        'intervention_freq': intervention_freq,
        'intervention_mag': intervention_mag,
        'control_effort': float(control_effort(actions)),
        'elapsed_s': float(time.time() - start),
        'eval_from_checkpoint': bool(args.eval_from_checkpoint),
        'checkpoint_path': checkpoint_path if args.eval_from_checkpoint else '',
        'regularization_weight': float(args.regularization_weight),
        'blended_cost_alpha': float(args.blended_cost_alpha),
    }

    print('[RESULT]', row)
    _append_summary_row(args.summary_csv, row)
    ctrl.close()
    return row


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO adversary against native NL-MPSC on integrator chains.')
    parser.add_argument('--r', type=int, required=True)
    parser.add_argument('--reward_mode', type=str, default='horizon_exhaustion', choices=REWARD_MODES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_env_steps', type=int, default=30000)
    parser.add_argument('--env_horizon', type=int, default=500)
    parser.add_argument('--mpsc_horizon', type=int, default=12)
    parser.add_argument('--mpsc_cost_horizon', type=int, default=6)
    parser.add_argument('--train_mpsc_horizon', type=int, default=None)
    parser.add_argument('--eval_mpsc_horizon', type=int, default=None)
    parser.add_argument('--train_mpsc_cost_horizon', type=int, default=1)
    parser.add_argument('--train_sf_cost_function', type=str, default=Cost_Function.ONE_STEP_COST.value,
                        choices=EVAL_COST_FUNCTION_CHOICES)
    parser.add_argument('--eval_sf_cost_function', type=str, default=Cost_Function.ONE_STEP_COST.value,
                        choices=EVAL_COST_FUNCTION_CHOICES)
    parser.add_argument('--eval_from_checkpoint', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--max_w', type=float, default=0.002)
    parser.add_argument('--regularization_weight', type=float, default=1.0)
    parser.add_argument('--blended_cost_alpha', type=float, default=0.5)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--umax', type=float, default=1.0)
    parser.add_argument('--x_limit', type=float, default=2.0)
    parser.add_argument('--v_limit', type=float, default=None)
    parser.add_argument('--init_scale', type=float, default=0.05)
    parser.add_argument('--rollout_steps', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--opt_epochs', type=int, default=10)
    parser.add_argument('--mini_batch_size', type=int, default=8)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--q_lin', type=float, default=1.0)
    parser.add_argument('--r_lin', type=float, default=1.0)
    parser.add_argument('--slack_cost', type=float, default=250.0)
    parser.add_argument('--use_terminal_set', action='store_true')
    parser.add_argument('--terminal_set_scale', type=float, default=0.5)
    parser.add_argument('--terminal_cost_weight', type=float, default=0.0)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--eval_episodes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    parser.add_argument('--recovery_nominal_threshold', type=float, default=0.1)
    parser.add_argument('--recovery_consecutive', type=int, default=5)
    parser.add_argument('--output_root', type=str, default='experiments/recoverability/ppo_runs')
    parser.add_argument('--summary_csv', type=str, default='experiments/recoverability/ppo_runs/summary.csv')
    parser.add_argument('--run_name', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
