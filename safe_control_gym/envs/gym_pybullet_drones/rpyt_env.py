import numpy as np
from gymnasium import spaces

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor


class RPYTRateWrapper:
    """A minimal wrapper that exposes a 2D RPYT-style action: [theta_rate, thrust].

    NOTE: This wrapper implements a pragmatic mapping for QuadType.TWO_D only.
    - Input action (normalized): [rate_cmd, thrust_cmd] in [-1,1].
    - `denormalize_action` returns physical per-motor thrusts [T1, T2].
    - `normalize_action` approximates the inverse mapping.
    This provides a compact way for RL agents to output rate+thrust commands
    while keeping the underlying environment and safety-filter interfaces unchanged.
    """

    def __init__(self, env, kp=1.0, max_rate=5.0):
        self.env = env
        self.kp = float(kp)
        self.max_rate = float(max_rate)

        # Expose a 2-dim normalized action space: [rate, thrust]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Keep observation and other attributes delegating to env
        self.observation_space = env.observation_space
        self.state_dim = env.state_dim
        self.action_dim = 2

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        # action is in wrapper space (normalized). Convert to underlying env normalized action.
        phys = self.denormalize_action(action)
        inner = self.env.normalize_action(phys)
        return self.env.step(inner)

    def _parse_action(self, action):
        """Return a flat 2D action [rate_norm, thrust_norm]."""
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError('RPYT action must have 2 elements, got scalar.')
        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(f'RPYT action must have shape (2,), got {arr.shape}.')
            return arr
        # Handle batched actions (e.g., (1, 2) from vectorized rollouts)
        if arr.ndim >= 2:
            flat_last = arr.reshape(-1, arr.shape[-1])
            if flat_last.shape[1] != 2:
                raise ValueError(f'RPYT action last dimension must be 2, got {arr.shape}.')
            return flat_last[0]
        raise ValueError(f'Unsupported RPYT action shape: {arr.shape}')

    def denormalize_action(self, action):
        # action: [rate_norm, thrust_norm]
        parsed = self._parse_action(action)
        rate_norm = float(parsed[0])
        thrust_norm = float(parsed[1])

        # Compute per-motor base thrust (physical) from normalized thrust
        # For the underlying env (2D) pass a symmetric per-motor thrust vector to denormalize.
        base_phys = self.env.denormalize_action(np.array([thrust_norm, thrust_norm], dtype=np.float32))
        # base_phys is per-motor physical thrusts; take their mean
        base = float(np.mean(base_phys))

        # current angular rate (theta_dot) from env state: [x, x_dot, z, z_dot, theta, theta_dot]
        cur_rate = float(self.env.state[5])
        desired_rate = rate_norm * self.max_rate

        # Simple P-control in rate to torque; torque = Iyy * kp * (desired_rate - cur_rate)
        Iyy = float(getattr(self.env, 'J', np.array([[0, 0], [0, 1.4e-5]]))[1][1]) if hasattr(self.env, 'J') else getattr(self.env, 'J', 1.4e-5)
        length = float(getattr(self.env, 'L', 1.0))
        torque = Iyy * self.kp * (desired_rate - cur_rate)

        # From symbolic dynamics (2D): theta_ddot ~ length*(T2-T1)/Iyy/sqrt(2)
        # Torque ~ length*(T2-T1)/sqrt(2)  => T2-T1 = torque * sqrt(2) / length
        thrust_diff = torque * np.sqrt(2.0) / max(1e-6, length)

        T1 = base - thrust_diff / 2.0
        T2 = base + thrust_diff / 2.0

        # Clip to env physical bounds
        lo, hi = self.env.physical_action_bounds
        T1 = float(np.clip(T1, lo[0], hi[0]))
        T2 = float(np.clip(T2, lo[1], hi[1]))
        return np.array([T1, T2], dtype=np.float32)

    def normalize_action(self, physical_action):
        # physical_action is an array [T1, T2]
        # thrust component: average per-motor normalized thrust
        try:
            thrust_norms = self.env.normalize_action(np.array([physical_action[0], physical_action[1]], dtype=np.float32))
            thrust_norm = float(np.mean(thrust_norms))
        except Exception:
            thrust_norm = 0.0

        # rate component: approximate current rate normalized to max_rate
        cur_rate = float(self.env.state[5]) if hasattr(self.env, 'state') and len(self.env.state) > 5 else 0.0
        rate_norm = float(np.clip(cur_rate / max(1e-6, self.max_rate), -1.0, 1.0))
        return np.array([rate_norm, thrust_norm], dtype=np.float32)

    # Delegate attribute access to underlying env where appropriate
    def __getattr__(self, name):
        return getattr(self.env, name)


def make_rpyt_quad(*args, **kwargs):
    """Factory used by the registration system to create a wrapped quadrotor env.

    Accepts the same keyword args as `Quadrotor` and returns an instance of
    `RPYTRateWrapper(Quadrotor(...))`.
    """
    env = Quadrotor(*args, **kwargs)
    # Only support 2D wrapped behavior for now.
    if env.QUAD_TYPE != env.QUAD_TYPE.TWO_D:
        raise ValueError('RPYTRateWrapper only supported for QUAD_TYPE TWO_D')
    return RPYTRateWrapper(env)
