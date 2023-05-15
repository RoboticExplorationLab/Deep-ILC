import numpy as np
import torch
from utils import rk4, deg2rad, Spaces, Spaces_np
import ipdb

class Cartpole:
    def __init__(self, mc=1.0, mp=0.2, l=0.5, g=9.81, b=0.01, sig=5, mu=0.0, deadband=0.0, u_max=np.inf, max_steps=50, dt = 0.05, targ_pos=torch.tensor([0,np.pi,0,0]), x_window=[1, deg2rad(180), 0.5, 0.5], device=torch.device('cpu')):
        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = g
        self.b = b
        self.sig = sig
        self.mu = mu
        self.dt = dt
        self.deadband = deadband
        self.u_max = u_max
        self.state_dim = 4
        self.control_dim = 1
        self._max_episode_steps = max_steps
        self.num_steps = 0
        self.bsz = 1
        self.x_window = torch.tensor(x_window).to(device).unsqueeze(0)
        self.targ_pos = torch.tensor(targ_pos).to(device).unsqueeze(0)
        self.Qlqr = torch.tensor([0.2,2.0,1e-2,1e-2]).to(device).unsqueeze(0)/10
        self.Rlqr = torch.tensor([1e-3]).to(device).unsqueeze(0)
        self.observation_space = Spaces_np((4,))
        self.action_space = Spaces_np((1,), np.array([40.0]), np.array([-40.0]))
        self.term_rew = -10
        self.clip_rew = -10
        self.device = device

    def normalize_theta(self, x=None):
        if x is None:
            x = self.x
        theta = x[:, 1] + (-torch.div(x[:, 1], (2*np.pi), rounding_mode='trunc')*(2*np.pi) + (x[:,1]<0).float()*(2*np.pi)).detach()
        x[:, 1] = theta
        return x

    def step(self, u):
        self.num_steps += 1
        done_inf = False
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                self.normalize_theta()
                reward = self.reward(self.x, u).detach().cpu().numpy().squeeze()
                if reward < self.clip_rew:
                    x = self.reset()
                    done_inf = True
                x_out = self.x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                self.normalize_theta()
                reward = self.reward(self.x, u).cpu().numpy()
                if reward.sum() < self.clip_rew:
                    x = self.reset()
                    done_inf = True
                x_out = self.x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            self.x = rk4(dynamics, self.x, [0, self.dt])
            self.normalize_theta()
            reward = self.reward(self.x, u)
            if reward.sum() < self.clip_rew:
                x = self.reset()
                done_inf = True
            x_out = self.x
        done = self.num_steps >= self._max_episode_steps or done_inf
        return x_out, reward, done, {'done_inf':done_inf}

    def stepx(self, x, u):
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            x = torch.tensor(x).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                x = x.unsqueeze(0)
                dynamics = lambda y: self.dynamics(y, u)
                x = rk4(dynamics, x, [0, self.dt])
                x = self.normalize_theta(x)
                reward = self.reward(x, u).cpu().numpy().squeeze()
                x_out = x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                x = rk4(dynamics, x, [0, self.dt])
                x = self.normalize_theta(x)
                reward = self.reward(x, u).cpu().numpy()
                x_out = x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            x = rk4(dynamics, x, [0, self.dt])
            x = self.normalize_theta(x)
            reward = self.reward(x, u)
            x_out = x
        done = self.num_steps >= self._max_episode_steps
        return x_out, reward, done, {'done_inf':False}

    def dynamics(self, x, u):
        mc = self.mc  # mass of the cart in kg (10)
        mp = self.mp   # mass of the pole (point mass at the end) in kg
        l = self.l   # length of the pole in m
        g = self.g  # gravity m/s^2

        q = x[:,:2]
        qd = x[:,2:4]

        s = torch.sin(q[:,1])
        c = torch.cos(q[:,1])
        ones = torch.ones_like(c)
        zeros = torch.zeros_like(c)

        H = torch.stack([
                torch.stack([(mc+mp)*ones, mp*l*c], dim=-1),
                torch.stack([mp*l*c, mp*l*l*ones], dim=-1)], dim=1)
        C = torch.stack([
                torch.stack([zeros, -mp*qd[:,1]*l*s], dim=-1),
                torch.stack([zeros, zeros], dim=-1)], dim=1)
        G = torch.stack([zeros, mp*g*l*s], dim=-1)
        B = torch.stack([ones, zeros], dim=-1)

        # Friction
        Fn = (mc + mp) * g  # normal force (approximate)
        sig = self.sig 
        mu = self.mu 
        cart_friction = torch.tanh(sig * qd[:,0]) * mu * Fn  # approximate coloumb friction
        viscous_friction = self.b * qd          # viscous friction
        friction = torch.stack([cart_friction + viscous_friction[:,0], viscous_friction[:,1]], dim=-1)

        # Control 
        deadband = self.deadband
        u_max = self.u_max 
        u_true = torch.sign(u)*torch.clamp(u.abs(), deadband, u_max)
        qdd = -(torch.linalg.inv(H)*((C*qd.unsqueeze(1)).sum(dim=-1) + G + friction - B*u_true[:,:1]).unsqueeze(1)).sum(dim=-1)
        return torch.cat([qd, qdd], dim=-1)

    def reward(self, x, u):
        err_state = (x - self.targ_pos)

        theta = err_state[:, 1] + (-torch.div(err_state[:, 1], (2*np.pi), rounding_mode='trunc')*(2*np.pi) + (err_state[:,1]<0).float()*(2*np.pi)).detach()
        # Look at the shortest path to the desired state i.e. desired angle
        mask = (theta > np.pi).float()
        theta = (1-mask)*theta + mask*(theta - 2*np.pi)
        err_state = torch.stack([err_state[:, 0], theta, err_state[:, 2], err_state[:, 3]], dim=1)
        cost = torch.sum(err_state * self.Qlqr * err_state, dim=1) #+ torch.sum(u * 1.0e-3 * u, dim=1)
        return -cost + 1

    def reset_torch(self, bsz=None, x_window=None):
        self.num_steps = 0
        if x_window is None:
            x_window = self.x_window
        elif len(x_window.shape) == 1:
            x_window = x_window.unsqueeze(0)
        if bsz is None:
            bsz = self.bsz
        self.x = (torch.rand((bsz, self.state_dim))*2-1)*x_window + self.targ_pos
        return self.x

    def reset(self, bsz=None, x_window=None):
        self.num_steps = 0
        x = self.reset_torch(bsz, x_window).detach().cpu().numpy().squeeze()
        return x
