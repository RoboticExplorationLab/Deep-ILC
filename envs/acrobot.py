import numpy as np
import torch
import math
from numpy import sin, cos, pi
import ipdb
from matplotlib import pyplot as plt 
from utils import *


class AcrobotEnv(torch.nn.Module):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    # dt = 0.05
    # dt = 0.01

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    AVAIL_TORQUE = torch.tensor([-1.0, 0.0, +1])

    torque_noise_max = 0.0

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 1

    num_obs = 4
    num_act = 1

    def __init__(self, batch_size=64, device=torch.device('cpu'), continuous=True, dt=0.01, T=400, l1=1.0, m1=1.0):
        super().__init__()
        self.viewer = None
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        self.continuous = continuous
        self.actions_disc = torch.arange(-6,7,3.0).unsqueeze(-1).to(device)
        self.state = None
        self.device = device
        self.bsz = self.batch_size = batch_size
        self.gs = torch.zeros((1,4)).to(device)
        self.gs[0,0] += np.pi/2
        self.init_state = torch.zeros((1,4)).to(device)
        self.init_state[0,0] -= np.pi/2

        self.observation_space = Spaces_np((4,))
        self.action_space = Spaces_np((1,), np.array([25,]), np.array([-25,]))
        self.env_name = 'acrobot_'
        self.action_coeffs = 0.05
        self.dt = dt
        self.num_steps = 0
        self.T = T

        self.LINK_LENGTH_1 = l1  # [m]
        self.LINK_LENGTH_2 = 1.0  # [m]
        self.LINK_MASS_1 = m1  #: [kg] mass of link 1
        self.LINK_MASS_2 = 1.0  #: [kg] mass of link 2
        self.LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
        self.LINK_MOI = 1.0  #: moments of inertia for both links
        self.LINK_i1=.2
        self.LINK_i2=.8
        self.normalize = True
        self.x_window =  torch.tensor([np.pi, np.pi, self.MAX_VEL_1/4, self.MAX_VEL_2/9]).unsqueeze(0).to(self.device)
        self.state_dim = self.num_obs
        self.control_dim = self.num_act
        self._max_episode_steps = T
        self.term_rew = 0.0



    def step(self, u):
        self.num_steps += 1
        num_steps_rew = self.term_rew
        done_inf = False
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                self.x = self.postprocess_state(self.x)
                reward = self.reward(self.x, u).cpu().numpy().squeeze()
                if (-torch.cos(self.x[:, 0]) - torch.cos(self.x[:, 1] + self.x[:, 0])).item() > 1.0:
                    x = self.reset()
                    done_inf = True
                    reward = -self.term_rew# np.minimum(reward, -num_steps_rew)
                x_out = self.x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                self.x = self.postprocess_state(self.x)
                reward = self.reward(self.x, u).cpu().numpy()
                if (-torch.cos(self.x[:, 0]) - torch.cos(self.x[:, 1] + self.x[:, 0])).item() > 1.0:
                    x = self.reset()
                    done_inf = True
                    reward = np.array([-self.term_rew])
                x_out = self.x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            self.x = rk4(dynamics, self.x, [0, self.dt])
            self.x = self.postprocess_state(self.x)
            reward = self.reward(self.x, u)
            if (-torch.cos(self.x[:, 0]) - torch.cos(self.x[:, 1] + self.x[:, 0])).item() > 1.0:
                x = self.reset()
                done_inf = True
                reward = torch.tensor([-self.term_rew]).to(self.x)
            x_out = self.x
        done_inf = done = self.num_steps >= self._max_episode_steps or done_inf
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
        done_inf = done = self.num_steps >= self._max_episode_steps or done_inf
        return x_out, reward, done, {'done_inf':done_inf}

    def postprocess_state(self, ns):
        if self.normalize:
            ns_th0 = angle_normalize(ns[:, 0], 0, 2*math.pi)
            ns_th1 = angle_normalize(ns[:, 1], -math.pi, math.pi)
        else:
            ns_th0 = ns[:, 0]
            ns_th1 = ns[:, 1]
        ns_vel2 = torch.clamp(ns[:, 2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns_vel3 = torch.clamp(ns[:, 3], -self.MAX_VEL_2, self.MAX_VEL_2)
        ns = torch.stack([ns_th0, ns_th1, ns_vel2, ns_vel3], dim=1)
        return ns

    def reward(self, ns, u):
        reward = torch.sin(ns[:, 0]) + torch.sin(ns[:, 0] + ns[:, 1])
        return reward

    def reset_torch(self, bsz=None, x_window=None):
        self.num_steps = 0
        if x_window is None:
            x_window = self.x_window
        elif len(x_window.shape) == 1:
            x_window = x_window.unsqueeze(0)
        if bsz is None:
            bsz = self.bsz

        self.x = (torch.rand((bsz, 4))*2-1).to(self.device)*x_window
        self.x[:, 0] += np.pi

        return self.x

    def reset(self, bsz=None, x_window=None):
        self.num_steps = 0
        x = self.reset_torch(bsz, x_window).detach().cpu().numpy().squeeze()
        return x


    def _get_obs(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        )


    def dynamics(self, x, u):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        i1 = self.LINK_i1
        i2 = self.LINK_i2
        g = 9.8
        s0 = x
        act = u
        tau = act[:,0]
        th1 = s0[:, 0]
        th2 = s0[:, 1]
        th1d = s0[:, 2]
        th2d = s0[:, 3]
        g = 9.8
        TAU = torch.stack([torch.zeros_like(tau),tau], dim=1).unsqueeze(-1)

        m11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*torch.cos(th2)) + i1 + i2
        m22 = m2*lc2**2 + i2
        m12 = m2*(lc2**2 + l1*lc2*torch.cos(th2)) + i2
        M = torch.stack([torch.stack([m11, m12], dim=-1), torch.stack([m12, m22*torch.ones_like(m12)], dim=-1)], dim=-2)
        # print("1:M:", M)
        h1 = -m2*l1*lc2*torch.sin(th2)*th2d**2 - 2*m2*l1*lc2*torch.sin(th2)*th2d*th1d
        h2 = m2*l1*lc2*torch.sin(th2)*th1d**2
        H = torch.stack([h1,h2],dim=-1).unsqueeze(-1)
        # print("1:H:", H)

        phi1 = (m1*lc1+m2*l1)*g*torch.cos(th1) + m2*lc2*g*torch.cos(th1+th2)
        phi2 = m2*lc2*g*torch.cos(th1+th2)
        PHI = torch.stack([phi1, phi2], dim=-1).unsqueeze(-1)
        # print("1:PHI:", PHI)

        d2th = torch.linalg.solve(M,(TAU - H - PHI)).squeeze(-1)
        # print("1:d2th:", d2th)
        return torch.stack([th1d, th2d, d2th[:,0], d2th[:,1]], dim=1)

    def get_frame(self, s, ax=None):
        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]
        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2

        if ax is None:
            fig, ax = plt.subplots(figsize=(2*bound,2*bound))
        else:
            fig = ax.get_figure()

        ax.plot((0,p1[0]), (0, p1[1]), color='k')
        ax.plot((p1[0],p2[0]), (p1[1], p2[1]), color='k')
        ax.set_xlim((-bound*1.1, bound*1.1))
        ax.set_ylim((-bound*1.1, bound*1.1))
        return fig, ax

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, 0.1, -0.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
