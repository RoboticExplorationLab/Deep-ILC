import numpy as np
from dm_control import suite
import concurrent.futures
from collections import OrderedDict
from dm_env.specs import Array
import numpy as np
import torch
import mujoco

class Space:
    def __init__(self, shape, dtype, low=np.inf, high=np.inf):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high, self.shape)

class DMControlEnvWrapper:
    def __init__(self, domain_name, task_name):
        self.dm_control_env = suite.load(domain_name, task_name)
        self.action_space = self._convert_to_space(self.dm_control_env.action_spec())
        self.observation_space = self._convert_to_space(self.dm_control_env.observation_spec())
        self._max_episode_steps = int(self.dm_control_env._step_limit)
        self.device = 'cpu'

    def _convert_to_space(self, spec):
        if isinstance(spec, Array):
            return Space(
                low=spec.minimum,
                high=spec.maximum,
                shape=spec.shape,
                dtype=spec.dtype
            )
        elif isinstance(spec, OrderedDict):
            return Space(shape=np.array([sum([spec[item].shape[0] for item in spec])]),
                dtype=spec['position'].dtype)
        else:
            raise NotImplementedError("Conversion for spec {} not implemented.".format(type(spec)))

    def _convert_to_observation(self, time_step):
        return np.concatenate([v for v in time_step.observation.values()])

    def reset(self):
        time_step = self.dm_control_env.reset()
        return self._convert_to_observation(time_step)

    def step(self, action):
        time_step = self.dm_control_env.step(action)
        obs = self._convert_to_observation(time_step)
        reward = time_step.reward
        done = time_step.last()
        info = {'done_inf' : False}
        return obs, reward, done, info

def create_env():
    return suite.load(domain_name='cheetah', task_name='run')

def compute_reward(env, state, action):
    env.physics.set_state(state)
    env.physics.set_control(action)
    reward = env.task.get_reward(env.physics)
    return reward

def compute_reward_perturbed(args):
    env, state, action, epsilon, idx, perturbation_type = args
    perturbed_state = state.copy()
    perturbed_action = action.copy()

    if perturbation_type == 'state':
        perturbed_state[idx] += epsilon
    else:
        perturbed_action[idx] += epsilon

    env.physics.set_state(perturbed_state)
    env.physics.set_control(perturbed_action)
    reward = compute_reward(env, perturbed_state, perturbed_action)
    return reward

def compute_jacobian_dmc(states, actions, true_next_states, env, args, done_inf, args_jacobian):
    state = states[0].cpu().numpy()
    action = actions[0].cpu().numpy()
    true_next_state = true_next_states[0].cpu().numpy()
    envs = [env.dm_control_env]
    state = np.concatenate([np.zeros(1), state])
    true_next_state = np.concatenate([np.zeros(1), true_next_state])
    df_dx, df_du, dr_dx, dr_du = mujoco_jacs(state, action, true_next_state, envs)
    df_dx = torch.tensor(df_dx[1:, 1:]).t()
    df_du = torch.tensor(df_du[:,1:]).t()
    dr_dx = torch.tensor(dr_dx[1:])
    dr_du = torch.tensor(dr_du)
    jac_ssa = torch.cat([df_dx, df_du], dim=-1).to(states).unsqueeze(0)
    grad_rsa = torch.cat([dr_dx, dr_du], dim=-1).to(actions).unsqueeze(0)
    return jac_ssa, grad_rsa

def mujoco_jacs(state, action, true_next_state, envs):
    env = envs[0]
    data = env.physics.data._data
    model = env.physics.model._model
    nv = model.nv
    nu = model.nu
    num_states = 2*nv
    num_actions = nu
    qpos = state[:nv].copy()
    qvel = state[nv:].copy()
    data.qpos = qpos
    data.qvel = qvel
    data.ctrl = action.copy()
    df_dx = np.zeros((2*nv, 2*nv))
    df_du = np.zeros((2*nv, nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, df_dx, df_du, None, None)
    
    perturbation_type = 'state'
    future_results = [compute_reward_perturbed((env, true_next_state, action, epsilon, i, perturbation_type)) for i in range(num_states)]
    reward_pos = future_results
    future_results = [compute_reward_perturbed((env, true_next_state, action, -epsilon, i, perturbation_type)) for i in range(num_states)]
    reward_neg = future_results

    dr_dx = (np.array(reward_pos) - np.array(reward_neg)) / (2 * epsilon)

    perturbation_type = 'action'
    future_results = [compute_reward_perturbed((env, true_next_state, action, epsilon, i, perturbation_type)) for i in range(num_actions)]
    reward_pos = future_results
    future_results = [compute_reward_perturbed((env, true_next_state, action, -epsilon, i, perturbation_type)) for i in range(num_actions)]
    reward_neg = future_results

    dr_du = (np.array(reward_pos) - np.array(reward_neg)) / (2 * epsilon)
    dr_du = dr_du + dr_dx.dot(df_du)
    dr_dx = dr_dx.dot(df_dx)
    return df_dx.transpose(), df_du.transpose(), dr_dx, dr_du
