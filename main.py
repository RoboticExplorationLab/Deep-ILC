import argparse
import datetime
import numpy as np
import itertools
from dmc_grad import DMControlEnvWrapper
from envs.cartpole import Cartpole
from envs.airplane import YakPlane
from envs.quadrotor import Quadrotor
from envs.rex_quadrotor import RexQuadrotor
from envs.acrobot import AcrobotEnv
from envs.cheetah import CheetahEnv
from envs.hopper import HopperEnv
from envs.ant import AntEnv

import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from utils import compute_jacobian_online, compute_jacobian_batch
from pretrain import pretrain_sac
import ipdb
import time
import sys
from rl_plotter.logger import Logger

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="rexquadrotor",
                    help='Choose environment (cartpole, acrobot, rexquadrotor, halfcheetah, hopper, DmcCheetah, DmcHopper)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_sizePi', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--hidden_sizeQ', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--eval_interval', type=int, default=40, 
                    help='evaluation interval (default: 40 episodes)')
parser.add_argument('--exp_name', type=str, default='scratch_training_test',
                    help='name of the experiment')
parser.add_argument('--offline', action="store_true",
                    help='Train with offline data (i.e without collecting any experiences) (default: False)')
parser.add_argument('--jacobian', action="store_true",
                    help='Train with approximate model jacobians (default: False)')
parser.add_argument('--pretrain', action="store_true",
                    help='Pretrain the model in the approximate model(default: False)')
parser.add_argument('--pretrain_jacobian', action="store_true",
                    help='Pretrain the model in the approximate model with model jacobians (default: False)')
parser.add_argument('--load', action="store_true",
                    help='Load the model saved with exp-name (default: False)')
parser.add_argument('--test', action="store_true",
                    help='Run without logging or saving (default: False)')
parser.add_argument('--zeroth', action="store_true",
                    help='Train with zeroth order jacobians from the approximate model (default: False)')
parser.add_argument('--jac_s_coeff', type=float, default=1.0,
                    help='Coefficient for the state jacobian loss (default: 1.0)')
args = parser.parse_args()

args.dflex_env = False
args.dmc_env = False
device = torch.device("cuda" if args.cuda else "cpu")
if args.env_name == 'cartpole':
    T = 100
    dt = 0.05
    env = Cartpole(mc=1.8, mp=0.6, b=0.08, deadband=0.05, mu=0.6, device=device, dt=dt, max_steps=T) #Cartpole(mu=0.4, device=device)
    env_nom =  Cartpole(b=0.0, deadband=0.0, u_max=np.inf, mu=0.0, device=device, dt=dt, max_steps=T)
elif args.env_name == 'acrobot':
    T = 100
    dt = 0.05
    env_nom = AcrobotEnv(1, device=device, dt=dt, T=T)
    env = AcrobotEnv(1, device=device, dt=dt, T=T, l1=1.2, m1=1.2)
elif args.env_name == 'quadrotor':
    env = Quadrotor(mass=0.75,
                    J=([[0.0026, 0.0003, 0.0],
                        [0.0003, 0.0026, 0.0],
                        [0.0, 0.0, 0.005]]),
                    motor_dist=0.2,
                    kf=0.95,
                    km=0.026,
                    cross_A_x = 0.3,
                    cross_A_y = 0.3,
                    cross_A_z = 0.65,
                    cd=[0.4, 0.4, 0.4],
                    max_steps=100,
                    dt=0.05)
    env_nom =  Quadrotor(max_steps=100, dt=0.05)
elif args.env_name == 'rexquadrotor':
    env_nom = RexQuadrotor()
    env = RexQuadrotor(mass=1.1*2.0,
                        J=[[0.0165, 0.0, 0.0], [0.0, 0.0165, 0.0],[0.0, 0.0, 0.0234]],
                        kf=1.1*0.0244101,
                        km=1.1*0.00029958,
                        bf=0.9*(-30.48576),
                        bm=0.9*(-0.367697),
                        cd=[0.3, 0.3, 0.3])
elif args.env_name == 'airplane':
    env = YakPlane(m=0.075*1.5, b=0.45*0.85, lin=True, max_steps=50, dt=0.04) 
    dtype = torch.float32
    if args.zeroth:
        dtype = torch.double
    env_nom = YakPlane(m=0.075*1.2, b=0.45*0.95, lin=True, max_steps=50, dt=0.04, dtype=dtype)
elif args.env_name == 'halfcheetah':
    env_params =  { 'density'   : 1000.0,
                    'stiffness' : 0.0,
                    'damping'   : 0.7,
                    'contact_ke': 1.e+5,
                    'contact_kd': 5.e+3,
                    'contact_kf': 1.e+3,
                    'contact_mu': 0.7,
                    'limit_ke'  : 1.e+3,
                    'limit_kd'  : 1.e+1,
                    'armature'  : 0.1}
    env = CheetahEnv(num_envs = 1, \
                device = 'cpu', \
                render = False, \
                seed = args.seed, \
                episode_length=1000, \
                stochastic_init = True, \
                MM_caching_frequency = 16, \
                no_grad=False,
                env_params=env_params)
    env_nom = CheetahEnv(num_envs = 1, \
                device = 'cpu', \
                render = False, \
                seed = args.seed, \
                episode_length=1000, \
                stochastic_init = True, \
                MM_caching_frequency = 16, \
                no_grad=False)
    args.dflex_env = True
elif args.env_name == 'hopper':
    env_params =  { 'density'   : 1000.0,
                    'stiffness' : 0.0,
                    'damping'   : 1.6,
                    'contact_ke': 1.e+5,
                    'contact_kd': 5.e+3,
                    'contact_kf': 1.e+3,
                    'contact_mu': 0.72,
                    'limit_ke'  : 1.e+3,
                    'limit_kd'  : 1.e+1,
                    'armature'  : 1.0}
    env = HopperEnv(num_envs = 1, \
                device = 'cpu', \
                render = False, \
                seed = args.seed, \
                episode_length=1000, \
                stochastic_init = True, \
                MM_caching_frequency = 16, \
                no_grad=False,
                env_params=env_params)
    env_nom = HopperEnv(num_envs = 1, \
                device = 'cpu', \
                render = False, \
                seed = args.seed, \
                episode_length=1000, \
                stochastic_init = True, \
                MM_caching_frequency = 16, \
                no_grad=False)
    args.dflex_env = True
elif args.env_name == 'DmcCheetah':
    from dm_control import suite
    args.dmc_env = True
    env = DMControlEnvWrapper(domain_name='cheetah', task_name='run')
    env_nom = DMControlEnvWrapper(domain_name='cheetah', task_name='run')
    args.jac_s_coeff = 0.0
    # Access the MuJoCo model
    model = env.dm_control_env.physics.model
    # Change damping coefficients for all joints
    model.dof_damping = model.dof_damping*0.7
    # Change contact stiffness coefficients
    # model.actuator_gainprm = model.actuator_gainprm
    # Change friction coefficients
    model.geom_friction = model.geom_friction*0.7
    
if args.offline:
    data = np.load('data')
else:
    data = None

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Saving and Loading
save_path = 'runs/SAC_{}_{}'.format(args.env_name,args.exp_name)
ckpt_save_path = save_path+'/checkpoint'
args.ckpt_save_path = ckpt_save_path
if args.load:
    if args.jacobian:
        save_path+= '/ft_jac_up2/'
        args.exp_name += '_ft_jac_up2/'
    else:
        save_path+= '/ft_rl/'
        args.exp_name += '_ft_rl/'

if args.load:
    agent.load_nets(ckpt_save_path)

# Tensorboard
if not args.test:
    writer = SummaryWriter(save_path+f'/seed{args.seed}/')
    config = {'args' : args}
    logger = Logger(log_dir="./rl_plotter_logs", exp_name=args.exp_name, env_name=args.env_name, seed=args.seed, config=config)
else:
    writer = None

# Memory
if args.offline:
    state, action, next_state = torch.tensor(data['state']).to(device).float(), torch.tensor(data['action']).to(device).float(), torch.tensor(data['next_state']).to(device).float()
    data['jac_ssa'][i], data['grad_rsa'][i] = compute_jacobians_batch(state, action, next_state, env_nom, args)
memory = ReplayMemory(args.replay_size, args.seed, data)

# Training Loop
total_numsteps = 0
updates = 0

if args.pretrain:
    pretrain_sac(agent, env_nom, writer, args, data)
    agent.refresh_optim()
    agent.reset_agent_to_best()
    sys.exit()

test_init = False


for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    states_arr = []
    actions_arr = []
    state = env.reset()

    while not done and not test_init:
        states_arr.append(state)
        if args.start_steps > total_numsteps and not args.pretrain and not args.load:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                if args.jacobian:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, stats = agent.update_parameters_jac(memory, args.batch_size, updates)
                else:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                if not args.test:
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    if args.jacobian:
                        writer.add_scalar('loss/jac_act', stats['jac_a_loss'], updates)
                        writer.add_scalar('loss/jac_state', stats['jac_s_loss'], updates)
                        writer.add_scalar('loss/critic_loss', stats['critic_loss'], updates)
                        if 'act_coeff' in stats:
                            writer.add_scalar('entropy_temprature/act_coeff', stats['act_coeff'], updates)
                            writer.add_scalar('entropy_temprature/state_coeff', stats['state_coeff'], updates)
                            writer.add_scalar('grads/grad_q', stats['grad_q'], updates)
                            writer.add_scalar('grads/grad_action', stats['grad_action'], updates)
                            writer.add_scalar('grads/grad_state', stats['grad_state'], updates)
                        writer.add_scalar('grads/policy', stats['policy_grad'], updates)
                        writer.add_scalar('grads/critic_grad', stats['critic_grad'], updates)
                        writer.add_scalar('grads/nq_jac_state', stats['nq_jac_state'].abs().mean(), updates)
                        writer.add_scalar('grads/nq_jac_action', stats['nq_jac_act'].abs().mean(), updates)
                        writer.add_scalar('grads_qf/nq_st_med', stats['nq_jac_state'].abs().mean(dim=-1).median(), updates)
                        writer.add_scalar('grads_qf/nq_act_med', stats['nq_jac_act'].abs().mean(dim=-1).median(), updates)
                        writer.add_scalar('grads_qf/nq_st_max', stats['nq_jac_state'].abs().mean(dim=-1).max(), updates)
                        writer.add_scalar('grads_qf/nq_act_max', stats['nq_jac_act'].abs().mean(dim=-1).max(), updates)
                updates += 1
        with torch.no_grad():
            next_state, reward, done, info = env.step(action) # Step


        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if (episode_steps == env._max_episode_steps and not info["done_inf"]) else float(not done)

        if not args.offline:
            # jac/grads are set to 0 when args.jacobian is False or done_inf is True
            jac_ssa, grad_rsa = compute_jacobian_online(state, action, next_state, env_nom, args, info['done_inf'])
            memory.push(state, action, reward, next_state, mask, jac_ssa, grad_rsa) # Append transition to memory


        state = next_state
        actions_arr.append(action)

    if total_numsteps > args.num_steps:
        break

    if not args.test and not test_init:
        writer.add_scalar('reward/train', episode_reward, i_episode)
    if i_episode % 1 == 0:
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if (i_episode-1) % args.eval_interval == 0 and args.eval is True:
        test_init=False
        avg_reward = 0.
        episodes = 40
        avg_num_steps = 0.0
        episode_reward_list = []
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            num_steps = 0
            while not done:
                action = agent.select_action(state, evaluate=True)
                with torch.no_grad():
                    next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                num_steps +=1

                state = next_state
            episode_reward_list.append(episode_reward)
            avg_reward += episode_reward
            avg_num_steps += num_steps
        avg_reward /= episodes
        avg_num_steps /= episodes

        if not args.test:
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            logger.update(score=episode_reward_list, total_steps=total_numsteps)


        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Avg. Numsteps: {}".format(episodes, round(avg_reward, 2), avg_num_steps))
        print("----------------------------------------")

env.close()

