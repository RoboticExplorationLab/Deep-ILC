import numpy as np
import itertools
import torch
from replay_memory import ReplayMemory
from utils import compute_jacobian_online, compute_jacobian_batch
import ipdb
import copy

def pretrain_sac(agent, env, writer, args, data):
    memory = ReplayMemory(args.replay_size, args.seed, data)
    total_numsteps = 0
    updates = 0
    best_reward = -1e8
    best_i_episode = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    if args.pretrain_jacobian:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, stats = agent.update_parameters_jac(memory, args.batch_size, updates)
                    else:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    if not args.test:
                        writer.add_scalar('loss_pt/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss_pt/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss_pt/policy', policy_loss, updates)
                        writer.add_scalar('loss_pt/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha_pt', alpha, updates)
                        if args.pretrain_jacobian:
                            writer.add_scalar('loss_pt/jac_act', stats['jac_a_loss'], updates)
                            writer.add_scalar('loss_pt/jac_state', stats['jac_s_loss'], updates)
                            writer.add_scalar('loss_pt/critic_loss', stats['critic_loss'], updates)
                            writer.add_scalar('grads_pt/policy', stats['policy_grad'], updates)
                            writer.add_scalar('grads_pt/critic_grad', stats['critic_grad'], updates)
                            writer.add_scalar('grads_pt/nq_jac_state', stats['nq_jac_state'].abs().mean(), updates)
                            writer.add_scalar('grads_pt/nq_jac_action', stats['nq_jac_act'].abs().mean(), updates)
                            writer.add_scalar('grads_qf_pt/nq_st_med', stats['nq_jac_state'].abs().mean(dim=-1).median(), updates)
                            writer.add_scalar('grads_qf_pt/nq_act_med', stats['nq_jac_act'].abs().mean(dim=-1).median(), updates)
                            writer.add_scalar('grads_qf_pt/nq_st_max', stats['nq_jac_state'].abs().mean(dim=-1).max(), updates)
                            writer.add_scalar('grads_qf_pt/nq_act_max', stats['nq_jac_act'].abs().mean(dim=-1).max(), updates)
                    updates += 1
            with torch.no_grad():
                next_state, reward, done, info = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            if not args.offline:
                jac_ssa, grad_rsa = compute_jacobian_online(state, action, next_state, env, args, info['done_inf'], args_jacobian=args.pretrain_jacobian)
                memory.push(state, action, reward, next_state, mask, jac_ssa, grad_rsa) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        if not args.test:
            writer.add_scalar('reward_pt/train', episode_reward, i_episode)
        if i_episode % 5 == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if (i_episode-1) % args.eval_interval == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 2
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

            if avg_reward > best_reward and not args.test:
                agent.best_policy = copy.deepcopy(agent.policy)
                agent.best_critic = copy.deepcopy(agent.critic)
                agent.best_critic_target = copy.deepcopy(agent.critic_target)
                best_i_episode = i_episode
                best_reward = avg_reward
                print("Updating and saving best policy/critic!")
                agent.save_best(env_name = args.env_name, ckpt_path=args.ckpt_save_path)

            if not args.test:
                writer.add_scalar('avg_reward_pt/test', avg_reward, i_episode)


            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Numsteps: {}".format(episodes, round(avg_reward, 2), avg_num_steps), episode_reward_list)
            print("----------------------------------------")
        if best_i_episode < i_episode - 320:
            break
