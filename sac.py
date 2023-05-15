import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import copy
import ipdb

class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.state_dim = num_inputs
        self.args = args

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_sizeQ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_sizeQ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.act_running_median = None
        self.state_running_median = None

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_sizePi, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_sizePi, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def compute_action_grad_loss(self, out, targ):
        diff = torch.abs(out - targ)
        median = diff.median(dim=0)[0].detach().clone().unsqueeze(0)
        if self.act_running_median is None:
            self.act_running_median = median
        else:
            self.act_running_median = 0.95*self.act_running_median + 0.05*median
        median = self.act_running_median
        mask = (diff < 2*median).float()
        cost = torch.square(out - targ)*mask + diff*4*median*(1-mask)
        cost = torch.mean(cost)
        return cost, mask.sum()

    def compute_state_grad_loss(self, out, targ):
        diff = torch.abs(out - targ)
        median = diff.median(dim=0)[0].detach().clone().unsqueeze(0)
        if self.state_running_median is None:
            self.state_running_median = median
        else:
            self.state_running_median = 0.95*self.state_running_median + 0.05*median
        median = self.state_running_median
        mask = (diff < 2*median).float()
        cost = torch.square(out - targ)*mask + diff*4*median*(1-mask)
        cost = torch.mean(cost)
        return cost, mask.sum()

    def jac_loss(self, qf_grad, targ_grad, mask):
        qf_grad_state, qf_grad_act = qf_grad[0], qf_grad[1]
        targ_grad_state, targ_grad_act = targ_grad[:, :self.state_dim], targ_grad[:, self.state_dim:]
        jac_s_loss, jacs_mask_num = self.compute_state_grad_loss(qf_grad_state*mask, targ_grad_state*mask)
        jac_a_loss, jaca_mask_num = self.compute_action_grad_loss(qf_grad_act*mask, targ_grad_act*mask)
        return jac_s_loss, jac_a_loss, jacs_mask_num, jaca_mask_num

    def update_parameters_jac(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, jac_ssa, grad_rsa = memory.sample_jac(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device).requires_grad_(True)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device).requires_grad_(True)
        action_batch = torch.FloatTensor(action_batch).to(self.device).requires_grad_(True)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        state_act_jac = torch.FloatTensor(jac_ssa).to(self.device)
        rew_grads_batch = torch.FloatTensor(grad_rsa).to(self.device)
        stats = {}

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        next_q_grad = rew_grads_batch + mask_batch * self.gamma * (torch.autograd.grad(min_qf_next_target.sum(), next_state_batch)[0].unsqueeze(-1)*state_act_jac).sum(dim=1)
        next_q_grad = next_q_grad.detach().clone()
        next_q_value = next_q_value.detach().clone()

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        qf1_jac = torch.autograd.grad(qf1.sum(), [state_batch, action_batch], retain_graph=True, create_graph=True)
        qf2_jac = torch.autograd.grad(qf2.sum(), [state_batch, action_batch], retain_graph=True, create_graph=True)

        jac_s_loss1, jac_a_loss1, _, _ = self.jac_loss(qf1_jac, next_q_grad, mask_batch)
        jac_s_loss2, jac_a_loss2, jacs2_mask_num, jaca2_mask_num = self.jac_loss(qf2_jac, next_q_grad, mask_batch)
        jac_s_loss = jac_s_loss1 + jac_s_loss2
        jac_a_loss = jac_a_loss1 + jac_a_loss2

        if updates % 20 ==0:
            grad_q = torch.autograd.grad(qf_loss, self.critic.linear2.weight, retain_graph=True)[0].norm()
            grad_action = torch.autograd.grad(jac_a_loss, self.critic.linear2.weight, retain_graph=True)[0].norm() #_unmask
            grad_state = torch.autograd.grad(jac_s_loss, self.critic.linear2.weight, retain_graph=True)[0].norm() #_unmask
            if updates ==0:
                self.act_coeff = grad_q/grad_action
                self.state_coeff = grad_q/grad_state
            else:
                self.act_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_action)
                self.state_coeff = 0.3*self.state_coeff + 0.7*(grad_q/grad_state)

            stats['grad_q'] = grad_q.item()
            stats['grad_action'] = grad_action.item()
            stats['grad_state'] = grad_state.item()
            stats['act_coeff'] = self.act_coeff.item()
            stats['state_coeff'] = self.state_coeff.item()

        critic_loss = qf_loss + jac_a_loss*self.act_coeff/2 + self.args.jac_s_coeff*jac_s_loss*self.state_coeff/2

        stats['jac_s_loss'] = jac_s_loss.item()
        stats['jac_a_loss'] = jac_a_loss.item()
        stats['critic_loss'] = critic_loss.item()
        stats['qf_loss'] = qf_loss.item()
        stats['nq_jac_state'] = next_q_grad[:,:self.state_dim].data.cpu()
        stats['nq_jac_act'] = next_q_grad[:,self.state_dim:].data.cpu()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        stats['critic_grad'] = self.critic.linear2.weight.grad.data.norm().item()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        stats['policy_grad'] = self.policy.linear2.weight.grad.data.norm().item()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), stats

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    def refresh_optim(self):
        self.critic_optim = Adam(self.critic.parameters(), lr=self.args.lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.args.lr)

    def save_best(self, env_name, ckpt_path=None):
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.best_policy.state_dict(),
                    'critic_state_dict': self.best_critic.state_dict(),
                    'critic_target_state_dict': self.best_critic_target.state_dict()}, ckpt_path)

    def reset_agent_to_best(self):
        self.critic = copy.deepcopy(self.best_critic)
        self.policy = copy.deepcopy(self.best_policy)
        self.critic_target = copy.deepcopy(self.best_critic_target)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

    # Load model parameters
    def load_nets(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

