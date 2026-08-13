'''Blend precomputed-trajectory tracking with rate-of-change regularization.'''

import numpy as np

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.precomputed_cost import PRECOMPUTED_COST


class BLENDED_COST(PRECOMPUTED_COST):
    '''Hybrid cost interpolating between precomputed and regularized MPSC costs.'''

    def __init__(self,
                 env,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 output_dir: str = '.',
                 regularization_weight: float = 1.0,
                 blend_alpha: float = 0.5,
                 ):
        super().__init__(env, mpsc_cost_horizon, decay_factor, output_dir)
        self.regularization_weight = regularization_weight
        self.blend_alpha = min(max(float(blend_alpha), 0.0), 1.0)
        self.uses_prev_u = True
        self.supports_precomputed = (
            env.__class__.__name__ != 'IntegratorChainEnv'
            and getattr(env, 'NAME', None) in {'cartpole', 'quadrotor'}
        )

    def get_cost(self, opti_dict):
        '''Return a weighted combination of precomputed and regularized smoothness terms.'''

        opti = opti_dict['opti']
        next_u = opti_dict['next_u']
        u_L = opti_dict['u_L']
        v_var = opti_dict['v_var']

        v_L = opti.parameter(self.model.nu, self.mpsc_cost_horizon)
        prev_u = opti.parameter(self.model.nu, 1)

        opti_dict['v_L'] = v_L
        opti_dict['prev_u'] = prev_u

        cost = (u_L - next_u).T @ (u_L - next_u)

        precomputed_cost = 0
        regularized_cost = (v_var[:, 0] - prev_u).T @ (v_var[:, 0] - prev_u)
        for h in range(1, self.mpsc_cost_horizon):
            precomputed_cost += (self.decay_factor**h) * (v_L[:, h] - v_var[:, h]).T @ (v_L[:, h] - v_var[:, h])
            regularized_cost += (self.decay_factor**h) * (v_var[:, h] - v_var[:, h - 1]).T @ (v_var[:, h] - v_var[:, h - 1])

        if self.supports_precomputed:
            cost += (1.0 - self.blend_alpha) * precomputed_cost
            cost += self.blend_alpha * self.regularization_weight * regularized_cost
        else:
            cost += self.regularization_weight * regularized_cost
        return cost

    def prepare_cost_variables(self, opti_dict, obs, iteration):
        '''Prepare both the precomputed trajectory guess and previous-action regularizer.'''

        opti = opti_dict['opti']
        prev_u = opti_dict['prev_u']
        if self.supports_precomputed:
            super().prepare_cost_variables(opti_dict, obs, iteration)
        else:
            opti.set_value(opti_dict['v_L'], 0.0)
        opti.set_value(prev_u, opti_dict['prev_u_val'])

    def calculate_unsafe_path(self, obs, uncertified_action, iteration):
        if not self.supports_precomputed:
            action = np.asarray(uncertified_action).reshape(self.model.nu, 1)
            return np.repeat(action, self.mpsc_cost_horizon, axis=1)
        return super().calculate_unsafe_path(obs, uncertified_action, iteration)
