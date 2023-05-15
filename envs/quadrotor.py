import numpy as np
import torch
from utils import rk4, deg2rad, Spaces, Spaces_np, w2pdotkinematics_mrp, quat2mrp, euler_to_quaternion, mrp2quat, quatrot
import ipdb

class Quadrotor:
    # think about mrp vs quat for various things - play with Q cost 
    def __init__(self, mass=0.5, J=[0.0023, 0.0023, 0.004], gravity=[0,0,-9.81], motor_dist=0.1750, kf=1.0, km=0.0245, bodyframe=False, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.25, cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu')):
        self.m = mass
        J = np.array(J)
        if len(J.shape) == 1:
            self.J = torch.diag(torch.FloatTensor(J)).unsqueeze(0).to(device)
        else:
            self.J = torch.FloatTensor(J).unsqueeze(0).to(device)
        self.Jinv = torch.linalg.inv(self.J).to(device)
        self.g = torch.tensor(gravity).to(device).unsqueeze(0)
        self.motor_dist = motor_dist
        self.kf = kf
        self.km = km
        self.bodyframe = bodyframe
        self.ned = ned
        self.cross_A_x = cross_A_x
        self.cross_A_y = cross_A_y
        self.cross_A_z = cross_A_z
        self.cross_A = torch.tensor(np.array([cross_A_x, cross_A_y, cross_A_y])).to(device).unsqueeze(0)
        self.state_dim = 3 + 3*3
        self.control_dim = 4
        self._max_episode_steps = max_steps
        self.num_steps = 0
        self.bsz = 1
        self.dt = dt
        self.cd = torch.tensor(cd).unsqueeze(0).to(device)
        self.Qlqr = torch.tensor([10.0]*3 + [10.0]*3 + [1.0]*6).to(device).unsqueeze(0)
        self.Rlqr = torch.tensor([1e-4]*self.control_dim).to(device).unsqueeze(0)
        self.observation_space = Spaces_np((self.state_dim,))
        self.action_space = Spaces_np((self.control_dim,), np.array([12.0]*self.control_dim), np.array([0.0]*self.control_dim)) #12.0
        self.device = device
        self.x_window = torch.tensor([5.0,5.0,5.0,deg2rad(70),deg2rad(70),deg2rad(70),1.0,1.0,1.0,0.5,0.5,0.5]).to(device)
        self.targ_pos = torch.zeros(self.state_dim)
        
    def forces(self, x, u):
        q = x[:, 3:7]
        F = torch.sum(torch.maximum(self.kf*u, torch.zeros_like(u)), dim=-1)
        g = self.g
        if self.ned:
            F = -F
            g = -g
        F = torch.stack([torch.zeros_like(F), torch.zeros_like(F), F], dim=-1)
        df = -torch.sign(x[:,3:6])*0.5*1.27*(x[:,3:6]**2)*self.cd*self.cross_A

        f = self.m*g + quatrot(q,F) + df # need to check q*F
        return f

    def moments(self, x, u):
        L = self.motor_dist
        F = torch.maximum(self.kf*u, torch.zeros_like(u))
        M = self.km*u
        tau1 = L*(F[:,1]-F[:,3])
        tau2 = L*(F[:,2]-F[:,0])
        tau3 = M[:,0]-M[:,1]+M[:,2]-M[:,3]
        if self.ned:
            tau2 = -tau2
            tau3 = -tau3
        tau = torch.stack([tau1, tau2, tau3], dim=-1)
        return tau

    def wrenches(self, x, u):
        F = self.forces(x, u)
        M = self.moments(x, u)
        return F, M

    def parse_state(self, x):
        r = x[:, :3]
        m = x[:, 3:6] # mrp2quat?
        v = x[:, 6:9]
        w = x[:, 9:]
        return r, m, v, w

    def dynamics(self, x, u):
        r, m, v, w = self.parse_state(x) # possibly do mrp2quat
        q = mrp2quat(m)
        F, tau = self.wrenches(x, u)
        mdot = w2pdotkinematics_mrp(m, w)
        if self.bodyframe:
            xdot = quatrot(q, v)  # quaternion multiplication (check!)
            vdot = quatrot(qinv(q), F/m) - torch.cross(w, v, dim=-1)
        else:
            xdot = v
            vdot = F/m
        ipdb.set_trace()
        wdot = (self.Jinv*(tau - torch.cross(w, (self.J*(w.unsqueeze(1))).sum(dim=-1), dim=-1)).unsqueeze(1)).sum(dim=-1)
        return torch.cat([xdot, mdot, vdot, wdot], dim=-1)

    def step(self, u):
        self.num_steps += 1
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                reward = self.reward(self.x, u).cpu().numpy().squeeze()
                x_out = self.x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                reward = self.reward(self.x, u).cpu().numpy()
                x_out = self.x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            self.x = rk4(dynamics, self.x, [0, self.dt])
            reward = self.reward(self.x, u)
            x_out = self.x
        done = self.num_steps >= self._max_episode_steps
        return x_out, reward, done, {}

    def stepx(self, x, u):
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            x = torch.tensor(x).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                x = x.unsqueeze(0)
                dynamics = lambda y: self.dynamics(y, u)
                x = rk4(dynamics, x, [0, self.dt])
                reward = self.reward(x, u).cpu().numpy().squeeze()
                x_out = x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                x = rk4(dynamics, x, [0, self.dt])
                reward = self.reward(x, u).cpu().numpy()
                x_out = x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            x = rk4(dynamics, x, [0, self.dt])
            reward = self.reward(x, u)
            x_out = x
        done = self.num_steps >= self._max_episode_steps
        return x_out, reward, done, {}

    def reward(self, x, u):
        cost = (((x - self.targ_pos)**2)*self.Qlqr/2).sum(dim=-1) + ((u**2)*self.Rlqr/2).sum(dim=-1)
        return -cost + 1

    def reset_torch(self, bsz=None, x_window=None):
        self.num_steps = 0
        if x_window is None:
            x_window = self.x_window
        elif len(x_window.shape) == 1:
            x_window = x_window.unsqueeze(0)
        if bsz is None:
            bsz = self.bsz
        self.x = (torch.rand((bsz, self.state_dim))*2-1)*self.x_window #+ self.targ_pos
        self.x = torch.cat([self.x[:,:3], quat2mrp(euler_to_quaternion(self.x[:, 3:6])), self.x[:, 6:]], dim=-1) #quat2mrp
        return self.x

    def reset(self, bsz=None, x_window=None):
        self.num_steps = 0
        x = self.reset_torch(bsz, x_window).detach().cpu().numpy().squeeze()
        return x
