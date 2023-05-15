import numpy as np
import torch
from utils import rk4, deg2rad, Spaces, Spaces_np, mrp2rot, Polynomial, arotate, brotate, w2pdotkinematics_mrp
import ipdb

class YakPlane:
    # reset, action limits (probably in dynamics 0-255), cost weights
    # check input/output against julia version
    # also check jacobians/gradients
    # finite horizon task with quadratic objective?
    # need an objective that only has gradients zero at origin or optimum
    # early termination -1000 reward -> 
    # correct termination -> 0 value -> very good
        # but might incentivise bad behavior ? 
        # maybe better to not mask terminal state?
        # might be a good idea to try both!
    # other vals for term rew
    # sample along trajectory?
    # autotuning, reduce alpha etc. 
    # fake jacaobian 
    # might need huber loss etc.. 
    # other rewards for tracking?
        # only compute rew on position
    # problem is that we need a supervised target at the end - 
    # problem is definitely with the reward/state rep/
    def __init__(self, g=9.81, rho=1.2, m=0.075, J=[4.8944e-04, 6.3778e-04, 7.9509e-04], 
                Jm=.007*(.0075**2) + .002*(.14**2)/12, b=45/100, l_in=6/100, cr=13.5/100, 
                ct=8/100, ep_ail=0.63, trim_ail=106, g_ail=(15*np.pi/180)/100, b_elev =16/100,
                cr_elev=6/100, ct_elev=4/100, r_elev=22/100,
                ep_elev = 0.88, #flap effectiveness (Phillips P.41)
                trim_elev = 106, #control input for zero deflection
                g_elev = (20*np.pi/180)/100, #maps control input to deflection angle
                b_rud = 10.5/100,
                cr_rud = 7/100,
                ct_rud = 3.5/100,
                r_rud = 24/100, #rudder moment arm (m)
                z_rud = 2/100, #height of rudder center of pressure (m)

                ep_rud = 0.76, #flap effectiveness (Phillips P.41)
                trim_rud = 106, #control input for zero deflection
                g_rud = (35*np.pi/180)/100, #maps from control input to deflection angle

                trim_thr = 24, #control input for zero thrust (deadband)
                g_thr = 0.006763, #maps control input to Newtons of thrust
                g_mot = 3000*2*np.pi/60*7/255,
                lin=False,
                dt = 0.05,
                max_steps = 50,
                device=torch.device('cpu'),
                dtype=torch.float): #maps control input to motor rad/sec
    # , device=torch.device('cpu')):
        self.g = g
        self.rho = rho
        self.m = m

        J = torch.tensor(J).to(device).to(dtype)
        self.J = torch.diag(J).unsqueeze(0)
        self.Jinv = torch.diag(1/J).unsqueeze(0)
        self.Jm = Jm

        self.b = b
        self.l_in = l_in
        self.cr = cr
        self.ct = ct
        self.cm = (ct + cr)/2
        self.S = b*self.cm
        self.S_in = 2*l_in*cr
        self.S_out = self.S-self.S_in
        self.Rt = ct/cr
        self.r_ail = (b/6)*(1+2*self.Rt)/(1+self.Rt)

        self.ep_ail = ep_ail
        self.trim_ail = trim_ail
        self.g_ail = g_ail

        self.b_elev = b_elev
        self.cr_elev = cr_elev
        self.ct_elev = ct_elev
        self.cm_elev = (ct_elev + cr_elev)/2 #mean elevator chord (m)
        self.S_elev = self.b_elev*self.cm_elev #planform area of elevator (m^2)
        self.Ra_elev = (self.b_elev**2)/self.S_elev #wing aspect ratio (dimensionless)
        self.r_elev = r_elev

        self.ep_elev = ep_elev
        self.trim_elev = trim_elev
        self.g_elev = g_elev

        self.b_rud = b_rud
        self.cr_rud = cr_rud
        self.ct_rud = ct_rud
        self.cm_rud = (ct_rud + cr_rud)/2 #mean rudder chord (m)
        self.S_rud = b_rud*self.cm_rud #planform area of rudder (m^2)
        self.Ra_rud = (b_rud**2)/self.S_rud #wing aspect ratio (dimensionless)
        self.r_rud = r_rud
        self.z_rud = z_rud

        self.ep_rud = ep_rud
        self.trim_rud = trim_rud
        self.g_rud = g_rud

        self.trim_thr = trim_thr
        self.g_thr = g_thr
        self.g_mot = g_mot

        self.lin = lin


        self.r_ailvec = torch.tensor([0,  self.r_ail, 0]).to(device).unsqueeze(0).to(dtype)
        self.l_invec = torch.tensor([0,  self.l_in, 0]).to(device).unsqueeze(0).to(dtype)
        self.r_elevvec = torch.tensor([-self.r_elev, 0, 0]).to(device).unsqueeze(0).to(dtype)
        self.rudvec = torch.tensor([-self.r_rud,  0, -self.z_rud]).to(device).unsqueeze(0).to(dtype)
        self.state_dim = 12
        self.control_dim = 4
        self._max_episode_steps = max_steps
        self.num_steps = 0
        self.bsz = 1

        # Initial condition
        p0 = [0.997156, 0., 0.075366] # initial orientation
        x0 = torch.tensor([-5,0,1.5]+ p0 + [5,0,0] + [0,0,0]).to(device).unsqueeze(0).to(dtype)
        self.x0 = x0

        # Final condition
        xf = x0.clone()
        pf = torch.tensor([5,0,1.5]).to(device).to(dtype)
        xf[0,:3] = pf
        xf[:, 6] = 0.0
        self.targ_pos = xf.to(dtype)

        x_window = [0.5]*3
        self.x_window = torch.tensor(x_window).to(device).unsqueeze(0).to(dtype)
        self.Qlqr = torch.tensor([1.0]*3 + [0.2]*3 + [0.1]*3 + [0.2]*3).to(device).unsqueeze(0).to(dtype)
        self.Rlqr = torch.tensor([1e-4]*self.control_dim).to(device).unsqueeze(0).to(dtype)

        self.observation_space = Spaces_np((self.state_dim,))
        self.action_space = Spaces_np((self.control_dim,), np.array([255.0]*self.control_dim), np.array([0.0]*self.control_dim))
        self.device = device
        self.dt = dt
        self.clip_rew = -50
        self.term_rew = 0#200
        self.ref_traj = torch.tensor(np.load('../BilinearControl.jl/data/airplane_ref.npy'))

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
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward) or np.isnan(reward) or reward < self.clip_rew:
                    x = self.reset()
                    done_inf = True
                    reward = -self.term_rew
                x_out = self.x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                self.x = rk4(dynamics, self.x, [0, self.dt])
                reward = self.reward(self.x, u).cpu().numpy()
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward.sum()) or np.isnan(reward.sum()) or reward.sum() < self.clip_rew:
                    x = self.reset()
                    done_inf = True
                    reward = np.array([-self.term_rew])
                x_out = self.x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            self.x = rk4(dynamics, self.x, [0, self.dt])
            reward = self.reward(self.x, u)
            if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or torch.isinf(reward.sum()) or torch.isnan(reward.sum()) or reward.sum() < self.clip_rew:
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
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = -self.term_rew
                    x = self.x
                x_out = x.squeeze().detach().cpu().numpy()
            else:
                dynamics = lambda y: self.dynamics(y, u)
                x = rk4(dynamics, x, [0, self.dt])
                reward = self.reward(x, u).cpu().numpy()
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = np.array([-self.term_rew])
                    x = self.x
                x_out = x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            dynamics = lambda y: self.dynamics(y, u)
            x = rk4(dynamics, x, [0, self.dt])
            reward = self.reward(x, u)
            if torch.isnan(x).sum() or torch.isinf(x).sum():
                self.reset()
                done_inf = True
                reward = torch.tensor([-self.term_rew]).to(self.x)
                x = self.x
            x_out = x
        done_inf = done = self.num_steps >= self._max_episode_steps or done_inf
        return x_out, reward, done, {'done_inf':done_inf}

    def propwash(self, thr):
        if not self.lin:
            zeros = torch.zeros_like(thr)
            trim_thr = 24; # control input for zero thrust (deadband)
            thr_mask = thr > trim_thr
            v = torch.stack([5.568*(thr**0.199) - 8.859, zeros, zeros], dim=-1)*(thr_mask).float()
        else:
            zeros = torch.zeros_like(thr)
            v = torch.stack([0.06*thr, zeros, zeros], dim=-1)
        return v

    def dynamics(self, x, u):
        r = x[:,:3]
        q = x[:,3:6]
        v = x[:,6:9]
        w = x[:,9:]

        Q = mrp2rot(q)

        # control input
        thr  = u[:,0]; #Throttle command (0-255 as sent to RC controller)
        ail  = u[:,1]; #Aileron command (0-255 as sent to RC controller)
        elev = u[:,2]; #Elevator command (0-255 as sent to RC controller)
        rud  = u[:,3]; #Rudder command (0-255 as sent to RC controller)

        #Note that body coordinate frame is:
        # x: points forward out nose
        # y: points out right wing tip
        # z: points down

        # ------- Input Checks -------- #
        thr  = torch.clamp(thr,  0, 255)
        ail  = torch.clamp(ail,  0, 255)
        elev = torch.clamp(elev, 0, 255)
        rud  = torch.clamp(rud,  0, 255)

        # ---------- Map Control Inputs to Angles ---------- #
        delta_ail = (ail-self.trim_ail)*self.g_ail
        delta_elev = (elev-self.trim_elev)*self.g_elev
        delta_rud = (rud-self.trim_rud)*self.g_rud

        # ---------- Aerodynamic Forces (body frame) ---------- #
        v_body = (Q*v.unsqueeze(-1)).sum(1) #body-frame velocity
        v_rout = v_body + torch.cross(w, self.r_ailvec)
        v_lout = v_body + torch.cross(w, -self.r_ailvec)
        v_rin  = v_body + torch.cross(w, self.l_invec) + self.propwash(thr)
        v_lin  = v_body + torch.cross(w, -self.l_invec) + self.propwash(thr)
        v_elev = v_body + torch.cross(w, self.r_elevvec) + self.propwash(thr)
        v_rud  = v_body + torch.cross(w, self.rudvec) + self.propwash(thr)

        # --- Outboard Wing Sections --- #
        a_rout = torch.atan2(v_rout[:, 2], v_rout[:, 0]);
        a_lout = torch.atan2(v_lout[:, 2], v_rout[:, 0]);
        a_eff_rout = a_rout + self.ep_ail*delta_ail; #effective angle of attack
        a_eff_lout = a_lout - self.ep_ail*delta_ail; #effective angle of attack
        zeros = torch.zeros_like(a_rout)

        F_rout = -self.p_dyn(v_rout).unsqueeze(-1)*.5*self.S_out*torch.stack([self.Cd_wing(a_eff_rout), zeros, self.Cl_wing(a_eff_rout)], dim=-1)
        F_lout = -self.p_dyn(v_lout).unsqueeze(-1)*.5*self.S_out*torch.stack([self.Cd_wing(a_eff_lout), zeros, self.Cl_wing(a_eff_lout)], dim=-1)

        F_rout = arotate(a_rout,F_rout); #rotate to body frame
        F_lout = arotate(a_lout,F_lout); #rotate to body frame

        # --- Inboard Wing Sections (Includes Propwash) --- #
        a_rin = torch.atan2(v_rin[:, 2], v_rin[:, 0])
        a_lin = torch.atan2(v_lin[:, 2], v_lin[:, 0])
        a_eff_rin = a_rin + self.ep_ail*delta_ail; #effective angle of attack
        a_eff_lin = a_lin - self.ep_ail*delta_ail; #effective angle of attack

        F_rin = -self.p_dyn(v_rin).unsqueeze(-1)*.5*self.S_in*torch.stack([self.Cd_wing(a_eff_rin), zeros, self.Cl_wing(a_eff_rin)], dim=-1)
        F_lin = -self.p_dyn(v_lin).unsqueeze(-1)*.5*self.S_in*torch.stack([self.Cd_wing(a_eff_lin), zeros, self.Cl_wing(a_eff_lin)], dim=-1)

        F_rin = arotate(a_rin,F_rin); #rotate to body frame
        F_lin = arotate(a_lin,F_lin); #rotate to body frame

        # --- Elevator --- #
        a_elev = torch.atan2(v_elev[:, 2], v_elev[:, 0])
        a_eff_elev = a_elev + self.ep_elev*delta_elev #effective angle of attack

        F_elev = -self.p_dyn(v_elev).unsqueeze(-1)*self.S_elev*torch.stack([self.Cd_elev( a_eff_elev), zeros, self.Cl_plate(a_eff_elev)], dim=-1);

        F_elev = arotate(a_elev,F_elev); #rotate to body frame

        # --- Rudder --- #
        a_rud = torch.atan2(v_rud[:, 1], v_rud[:, 0])
        a_eff_rud = a_rud - self.ep_rud*delta_rud; #effective angle of attack

        F_rud = -self.p_dyn(v_rud).unsqueeze(-1)*self.S_rud * torch.stack([self.Cd_rud(a_eff_rud), self.Cl_plate(a_eff_rud), zeros], dim=-1);

        F_rud = brotate(a_rud,F_rud); #rotate to body frame

        # --- Thrust --- #
        thr_mask = (thr > self.trim_thr).float().unsqueeze(-1)
        zeros = torch.zeros_like(thr)
        F_thr = torch.stack([(thr-self.trim_thr)*self.g_thr, zeros, zeros], dim=-1)
        w_mot = torch.stack([self.g_mot*thr, zeros, zeros], dim=-1)*(thr_mask)

        # ---------- Aerodynamic Torques (body frame) ---------- #
        T_rout = torch.cross(torch.tensor([0, self.r_ail, 0]).to(thr).unsqueeze(0),F_rout)
        T_lout = torch.cross(torch.tensor([0, -self.r_ail, 0]).to(thr).unsqueeze(0),F_lout)

        T_rin = torch.cross(torch.tensor([0, self.l_in, 0]).to(thr).unsqueeze(0),F_rin)
        T_lin = torch.cross(torch.tensor([0, -self.l_in, 0]).to(thr).unsqueeze(0),F_lin)

        T_elev = torch.cross(torch.tensor([-self.r_elev, 0, 0]).to(thr).unsqueeze(0),F_elev)

        T_rud = torch.cross(torch.tensor([-self.r_rud, 0, -self.z_rud]).to(thr).unsqueeze(0),F_rud)

        # ---------- Add Everything Together ---------- #
        F_aero = F_rout + F_lout + F_rin + F_lin + F_elev + F_rud + F_thr;
        F = (Q*F_aero.unsqueeze(dim=1)).sum(dim=-1) - torch.tensor([0, 0, self.m*self.g]).to(thr).unsqueeze(0)

        T = T_rout + T_lout + T_rin + T_lin + T_elev + T_rud + torch.cross(((self.J*w.unsqueeze(1)).sum(dim=-1) + (self.Jm*w_mot.unsqueeze(1)).sum(dim=-1)),w)

        rdot = v
        qdot = w2pdotkinematics_mrp(q, w)
        vdot = F/self.m
        wdot = (self.Jinv*T.unsqueeze(dim=1)).sum(dim=-1)
        ns = torch.cat([rdot, qdot, vdot, wdot], dim=-1)
        
        return torch.cat([rdot, qdot, vdot, wdot], dim=-1)

    def reward(self, x, u):
        nearest_x_id = torch.argmin((x[:,:3].unsqueeze(1) - self.ref_traj[:,:3].unsqueeze(0)).norm(dim=-1), dim=1)
        nearest_x_id = torch.min(nearest_x_id, torch.tensor(self.num_steps))
        nearest_x = self.ref_traj[nearest_x_id]
        targ_pos = nearest_x
        cost = (((x - self.targ_pos)**2)*self.Qlqr/2).sum(dim=-1)/10# + ((u**2)*self.Rlqr/2).sum(dim=-1)/10
        mask = cost > -self.clip_rew
        rew = torch.exp(-cost/2+2) 
        rew[mask] = -cost[mask]
        return rew

    def reset_torch(self, bsz=None, x_window=None):
        self.num_steps = 0
        if x_window is None:
            x_window = self.x_window
        elif len(x_window.shape) == 1:
            x_window = x_window.unsqueeze(0)
        if bsz is None:
            bsz = self.bsz
        self.x = self.x0.clone()
        self.x[:,:3] = (torch.rand((bsz, 3))*2-1)*x_window

        return self.x

    def reset(self, bsz=None, x_window=None):
        self.num_steps = 0
        x = self.reset_torch(bsz, x_window).detach().cpu().numpy().squeeze()
        return x


    #"Dynamic pressure"
    def p_dyn(self, v):
        return 0.5*self.rho*((v*v).sum(dim=-1))
    
    """ Lift coefficient (alpha in radians)
    3rd order polynomial fit to glide-test data
    Good to about ±20°
    """
    def Cl_wing(self, a):
        if not self.lin:
            a = torch.clamp(a, -0.5*np.pi, 0.5*np.pi)
            cl = -27.52*a**3 - .6353*a**2 + 6.089*a
            Cl_coef = torch.tensor(list(reversed([-9.781885297556400, 38.779513049043175, -52.388499489940138, 19.266141214863080, 15.435976905745736, -13.127972418509980, -1.155316115022734, 3.634063117174400, -0.000000000000001]))).to(a)
            cl = Polynomial(Cl_coef,a)
        else:
            a = torch.clamp(a, -0.5*np.pi, 0.5*np.pi)
            cl = 2*np.pi*a  # flat plate
            cl = 6*a
        return cl
    """ Lift coefficient (alpha in radians)
    Ideal flat plate model used for wing and rudder
    """
    def Cl_plate(self,a):
        a = torch.clamp(a, -0.5*np.pi, 0.5*np.pi)
        return 2*np.pi*a

    """ Drag coefficient (alpha in radians)
    2nd order polynomial fit to glide-test data
    Good to about ±20°
    """
    def Cd_wing(self, a):
        if not self.lin:
            a = torch.clamp(a, -0.5*np.pi, 0.5*np.pi)
            cd = 2.08*(a**2) + .0612
            Cd_coef = torch.tensor(list(reversed([5.006504698588220, -15.506103756150457, 11.861112337513331, 5.809973499615182, -8.905261747777024, -0.745202453390095, 3.835222476977072, 0.003598511005068, 0.064645268865914]))).to(a)
            cd = Polynomial(Cd_coef, a)
        else:
            a = torch.clamp(a, -0.5*np.pi, 0.5*np.pi)
            cd = 0.05 + a**2
        return cd
    

    """ Drag coefficient (alpha in radians)
    Induced drag for a tapered finite wing
        From phillips P.55
    """
    def Cd_elev(self, a):
        a = torch.clamp(a, -0.5*np.pi, 0.5*np.pi)
        cd = (4*np.pi*(a**2))/self.Ra_elev
        return cd
    
    """ Drag coefficient (alpha in radians)
    Induced drag for a tapered finite wing
    From Phillips P.55
    """
    def Cd_rud(self, a):
        a = torch.clamp(a, -0.5*np.pi, 0.5*np.pi)
        cd = (4*np.pi*(a**2))/self.Ra_rud;
        return cd
    