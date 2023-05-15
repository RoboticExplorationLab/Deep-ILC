import numpy as np
import torch
from utils import rk4, deg2rad, Spaces, Spaces_np, w2pdotkinematics_mrp, quat2mrp, euler_to_quaternion, mrp2quat, quatrot, mrp2rot
import ipdb

class RexQuadrotor:
    # think about mrp vs quat for various things - play with Q cost 
    def __init__(self, mass=2.0, J=[[0.01566089, 0.00000318037, 0.0],[0.00000318037, 0.01562078, 0.0], [0.0, 0.0, 0.02226868]], 
                    gravity=[0,0,-9.81], motor_dist=0.28, kf=0.0244101, bf=-30.48576, km=0.00029958, bm=-0.367697, 
                    quad_min_throttle = 1148.0, quad_max_throttle = 1832.0, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.5, 
                    cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu')):
        self.m = mass
        J = np.array(J)
        if len(J.shape) == 1:
            self.J = torch.diag(torch.FloatTensor(J)).unsqueeze(0).to(device)
        else:
            self.J = torch.FloatTensor(J).unsqueeze(0).to(device)
        self.Jinv = torch.linalg.inv(self.J).to(device)
        self.g = torch.FloatTensor(gravity).to(device).unsqueeze(0)
        self.motor_dist = motor_dist
        self.kf = kf
        self.km = km
        self.bf = bf
        self.bm = bm
        self.Bf = torch.zeros((1,3)).to(device)
        self.Bf[0,2] = 4*bf
        self.quad_min_throttle = quad_min_throttle
        self.quad_max_throttle = quad_max_throttle
        self.ned = ned
        self.cross_A_x = cross_A_x
        self.cross_A_y = cross_A_y
        self.cross_A_z = cross_A_z
        self.cross_A = torch.FloatTensor(np.array([cross_A_x, cross_A_y, cross_A_y])).to(device).unsqueeze(0)
        self.state_dim = 3 + 3*3
        self.control_dim = 4
        self._max_episode_steps = max_steps
        self.num_steps = 0
        self.bsz = 1
        self.dt = dt
        self.cd = torch.tensor(cd).unsqueeze(0).to(device)
        self.ss = torch.tensor([[1.,1,0], [1.,-1,0], [-1.,-1,0], [-1.,1,0]]).to(device).unsqueeze(0)
        self.ss = self.ss/self.ss.norm(dim=-1).unsqueeze(-1)
        self.Qlqr = torch.tensor([10.0]*3 + [10.0]*3 + [1.0]*6).to(device).unsqueeze(0)
        self.Rlqr = torch.tensor([1e-4]*self.control_dim).to(device).unsqueeze(0)
        self.observation_space = Spaces_np((self.state_dim,))
        self.act_scale = 100.0
        self.action_space = Spaces_np((self.control_dim,), np.array([18.3]*self.control_dim), np.array([11.5]*self.control_dim)) #12.0
        self.device = device
        self.x_window = torch.tensor([5.0,5.0,5.0,deg2rad(70),deg2rad(70),deg2rad(70),0.5,0.5,0.5,0.25,0.25,0.25]).to(device)
        self.targ_pos = torch.zeros(self.state_dim)
        
    def forces(self, x, u):
        m = x[:, 3:6]
        q = mrp2quat(-m)
        F = torch.sum(self.kf*u, dim=-1)
        g = self.g
        F = torch.stack([torch.zeros_like(F), torch.zeros_like(F), F], dim=-1)
        df = -torch.sign(m)*0.5*1.27*(m*m)*self.cd*self.cross_A

        f = F + df + quatrot(q, self.m * g)  + self.Bf
        return f

    def moments(self, x, u):
        L = self.motor_dist        
        zeros = torch.zeros_like(u)
        F = torch.maximum(self.kf*u, zeros)
        M = self.km*u
        tau1 = zeros[:,0]
        tau2 = zeros[:,0]
        tau3 = M[:,0]-M[:,1]+M[:,2]-M[:,3]
        torque = torch.stack([tau1, tau2, tau3], dim=-1)
        ss = self.ss
        if self.ss.dtype != x.dtype:
            ss = self.ss.to(x)
        torque += torch.cross(self.motor_dist * ss, torch.stack([zeros, zeros, self.kf * u + self.bf], dim=-1), dim=-1).sum(dim=1)
        return torque

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
        u = self.act_scale * u
        p, m, v, w = self.parse_state(x) # possibly do mrp2quat
        q = mrp2quat(m)
        F, tau = self.wrenches(x, u)
        mdot = w2pdotkinematics_mrp(m, w)
        pdot = quatrot(q, v)
        vdot = F/self.m - torch.cross(w, v)
        wdot = (self.Jinv*(tau - torch.cross(w, (self.J*(w.unsqueeze(1))).sum(dim=-1), dim=-1)).unsqueeze(1)).sum(dim=-1)
        return torch.cat([pdot, mdot, vdot, wdot], dim=-1)

    def step(self, u):
        self.num_steps += 1
        done_inf = False
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                reward = self.reward(self.x, u).cpu().numpy().squeeze()
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward) or np.isnan(reward) or reward < -500:
                    x = self.reset()
                    done_inf = True
                    reward = 0
                x_out = self.x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                reward = self.reward(self.x, u).cpu().numpy()
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward.sum()) or np.isnan(reward.sum()) or reward.sum() < -500:
                    x = self.reset()
                    done_inf = True
                    reward = np.array([0.0])
                x_out = self.x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            self.x = rk4(dynamics, self.x, [0, self.dt])
            reward = self.reward(self.x, u)
            if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or torch.isinf(reward.sum()) or torch.isnan(reward.sum()) or reward.sum() < -500:
                x = self.reset()
                done_inf = True
                reward = torch.tensor([0.0]).to(self.x)
            x_out = self.x
        done = self.num_steps >= self._max_episode_steps or done_inf
        if np.isinf(reward) or np.isnan(reward):
            ipdb.set_trace()
        return x_out, reward, done, {'done_inf':done_inf}

    def stepx(self, x, u):
        done_inf = False
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            x = torch.tensor(x).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                x = x.unsqueeze(0)
                dynamics = lambda y: self.dynamics(y, u)
                x = rk4(dynamics, x, [0, self.dt])
                reward = self.reward(x, u).cpu().numpy().squeeze()
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = -1000
                    x = self.x
                x_out = x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                x = rk4(dynamics, x, [0, self.dt])
                reward = self.reward(x, u).cpu().numpy()
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = np.array([-1000])
                    x = self.x
                x_out = x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            x = rk4(dynamics, x, [0, self.dt])
            reward = self.reward(x, u)
            if torch.isnan(x).sum() or torch.isinf(x).sum():
                self.reset()
                done_inf = True
                reward = torch.tensor([-1000]).to(self.x)
                x = self.x
            x_out = x
        done = self.num_steps >= self._max_episode_steps or done_inf
        return x_out, reward, done, {'done_inf':done_inf}


    def reward(self, x, u):
        cost = (((x - self.targ_pos)**2)*self.Qlqr/2).sum(dim=-1)/100 + ((u**2)*self.Rlqr/2).sum(dim=-1)/10
        mask = (cost > 500)
        rew = torch.exp(-cost/2+2)
        rew[mask] = -cost[mask]
        # if torch.isnan(rew).sum() or torch.isinf(rew).sum():
        #     ipdb.set_trace()
            
        return rew

    def reset_torch(self, bsz=None, x_window=None):
        self.num_steps = 0
        if x_window is None:
            x_window = self.x_window
        elif len(x_window.shape) == 1:
            x_window = x_window.unsqueeze(0)
        if bsz is None:
            bsz = self.bsz
        self.x = (torch.rand((bsz, self.state_dim))*2-1)*self.x_window 
        self.x = torch.cat([self.x[:,:3], quat2mrp(euler_to_quaternion(self.x[:, 3:6])), self.x[:, 6:]], dim=-1) #quat2mrp
        return self.x

    def reset(self, bsz=None, x_window=None):
        self.num_steps = 0
        x = self.reset_torch(bsz, x_window).detach().cpu().numpy().squeeze()
        return x
