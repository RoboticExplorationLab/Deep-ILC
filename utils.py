import math
import numpy as np
import torch
from torch.autograd.functional import jacobian
from dmc_grad import *
import ipdb


class Spaces:
    def __init__(self,shape, high=0, low=0):
        self.shape = shape
        self.high = torch.tensor(high)
        self.low = torch.tensor(low)
    def sample(self,):
        return torch.rand(self.shape)*(self.high-self.low) + self.low

class Spaces_np:
    def __init__(self,shape, high=0, low=0):
        self.shape = shape
        self.high = high
        self.low = low
    def sample(self,):
        return np.random.rand(*self.shape)*(self.high-self.low) + self.low


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def rk4(derivs, y0, t):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function

    Example 1 ::
        ## 2D system
        def derivs(x):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    This would then require re-adding the time variable to the signature of derivs.

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    yout = []
    yout.append(y0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]
        k1 = derivs(y0)
        k2 = derivs(y0 + dt2 * k1)
        k3 = derivs(y0 + dt2 * k2)
        k4 = derivs(y0 + dt * k3)
        yout.append(y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1]

def euler(derivs, y0, t):

    yout = []
    yout.append(y0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        y0 = yout[i]
        k1 = derivs(y0)
        yout.append(y0 + dt * k1)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1]


def compute_jacobian_batch(state, action, next_state, env, args, done_inf, args_jacobian=True):
    if not args_jacobian or done_inf:
        state_dim = state.shape[-1]
        act_dim = action.shape[-1]
        bsz = state.shape[0]
        return torch.zeros((bsz, state_dim, state_dim+act_dim)).to(state), torch.zeros((bsz, state_dim+act_dim)).to(state)
    def func(inp):
        x = inp[:, :state.shape[-1]]
        u = inp[:, state.shape[-1]:]
        ns, _, _, _ = env.stepx(x, u)
        ns_rew = ns + (next_state - ns).detach().clone()
        rew = env.reward(ns_rew, u).unsqueeze(-1)
        return torch.cat([ns, rew], dim=-1).sum(dim=0)
    jac_full = jacobian(func, torch.cat([state, action], dim=-1), vectorize=True).permute(1,0,2)
    jac_ssa = jac_full[:,:-1]
    grad_rsa = jac_full[:,-1]
    return jac_ssa, grad_rsa

def compute_jacobian_dflex(state, action, next_state, env, args, done_inf, args_jacobian=True):
    state_dim = state.shape[-1]
    act_dim = action.shape[-1]
    if not args_jacobian or done_inf:
        return torch.zeros((1, state_dim, state_dim+act_dim)).to(state), torch.zeros((1, state_dim+act_dim)).to(state)
    def func(x, u):
        ns, _, _, _ = env.stepx(x, u)

        ns_rew = ns[-1:] + (next_state - ns[-1:]).detach().clone()
        rew = env.reward(ns_rew, u[-1:]).unsqueeze(-1)
        idx = list(np.arange(ns.shape[-1]))
        out = ns[:-1][idx,idx].sum() + rew.sum()
        return out

    x = torch.cat([state]*(state_dim+1), dim=0).requires_grad_(True)
    u = torch.cat([action]*(state_dim+1), dim=0).requires_grad_(True)
    jac_full = torch.autograd.grad(func(x, u), [x, u])
    jac_full = torch.cat(jac_full, dim=-1)
    jac_ssa = jac_full[:-1].unsqueeze(0)
    grad_rsa = jac_full[-1].unsqueeze(0)
    return jac_ssa, grad_rsa


def compute_jacobian_zeroth(state, action, next_state, env, args, done_inf, args_jacobian=True):
    state_dim = state.shape[-1]
    action_dim = action.shape[-1]
    if not args_jacobian or done_inf:
        return torch.zeros((1, state_dim, state_dim+action_dim)).to(state), torch.zeros((1, state_dim+action_dim)).to(state)

    def func(x, u):
        ns, _, _, _ = env.stepx(x, u)
        ns_rew = ns + (next_state - ns[-1:]).detach().clone()
        rew = env.reward(ns_rew, u[-1:]).unsqueeze(-1)
        idx = list(np.arange(ns.shape[-1]))
        out = torch.cat([ns, rew],dim=-1)
        return out

    eps = 1e-6
    eps_state = torch.cat([torch.eye(state_dim), -torch.eye(state_dim), torch.zeros(2*action_dim+1, state_dim)], dim=0)*eps
    eps_action = torch.cat([torch.zeros(2*state_dim, action_dim), torch.eye(action_dim), -torch.eye(action_dim), torch.zeros(1, action_dim)], dim=0)*eps
    x = torch.cat([state]*(2*(state_dim+action_dim)+1), dim=0) + eps_state
    u = torch.cat([action]*(2*(state_dim+action_dim)+1), dim=0) + eps_action
    out = func(x, u)

    jac_state = (out[:state_dim] - out[state_dim:2*state_dim])/(2*eps)
    jac_action = (out[2*state_dim:2*state_dim+action_dim] - out[2*state_dim+action_dim:-1])/(2*eps)
    jac_full = torch.cat([jac_state, jac_action], dim=0).t().to(state).detach()
    jac_ssa = jac_full[:-1].unsqueeze(0)
    grad_rsa = jac_full[-1].unsqueeze(0)
    return jac_ssa, grad_rsa

def compute_jacobian_online(state, action, next_state, env, args, done_inf, args_jacobian=None, args_zeroth=None):
    if args_jacobian is None:
        args_jacobian = args.jacobian
    if args_zeroth is None:
        args_zeroth = args.zeroth
    if state.dtype == np.float64 or state.dtype == np.float32:
        state = torch.FloatTensor(state).to(env.device)
        action = torch.FloatTensor(action).to(env.device)
        next_state = torch.FloatTensor(next_state).to(env.device)
    if args.dmc_env:
        jac_ssa, grad_rsa = compute_jacobian_dmc(state.unsqueeze(0), action.unsqueeze(0), next_state.unsqueeze(0), env, args, done_inf, args_jacobian)
    elif args.dflex_env:
        jac_ssa, grad_rsa = compute_jacobian_dflex(state.unsqueeze(0), action.unsqueeze(0), next_state.unsqueeze(0), env, args, done_inf, args_jacobian)
    else:    
        if args_zeroth:
            jac_ssa, grad_rsa = compute_jacobian_zeroth(state.unsqueeze(0), action.unsqueeze(0), next_state.unsqueeze(0), env, args, done_inf, args_jacobian)
        else:
            jac_ssa, grad_rsa = compute_jacobian_batch(state.unsqueeze(0), action.unsqueeze(0), next_state.unsqueeze(0), env, args, done_inf, args_jacobian)
    return jac_ssa[0].cpu().numpy(), grad_rsa[0].cpu().numpy()

def deg2rad(deg):
    return (deg%360)*(2*np.pi)/360


def quatrot(q, r):
    qv = q[:, 1:]
    qs = q[:, 0:1]
    r = r.expand_as(qv)
    if q.dtype != r.dtype:
        r = r.to(q)
    
    cross_product = torch.cross(qv, r, dim=-1)
    qr = (qs**2 - (qv*qv).sum(dim=-1, keepdim=True)) * r + 2 * qv * (qv * r).sum(dim=-1, keepdim=True) + 2 * qs * cross_product
    return qr

def getqvx(qv, bsz):
    qvx = torch.zeros((bsz, 3, 3)).to(qv)
    qvx[:, 0, 1] = -qv[:, 2]
    qvx[:, 0, 2] = qv[:, 1]
    qvx[:, 1, 2] = -qv[:, 0]
    qvx = qvx - (qvx.transpose(1,2))
    return qvx

def getLq(qv, qs, qvx, bsz):
    Lq = torch.zeros((bsz, 4, 4)).to(qv)
    Lq[:, 0, 0] += qs
    Lq[:, 0, 1:] += -qv
    Lq[:, 1:, 0] += qv
    Lq[:, 1:, 1:] += qs.unsqueeze(-1).unsqueeze(-1) + qvx
    return Lq

def getRq(qv, qs, qvx, bsz):
    Rq = torch.zeros((bsz, 4, 4)).to(qv)
    Rq[:, 0, 0] += qs
    Rq[:, 0, 1:] += -qv
    Rq[:, 1:, 0] += qv
    Rq[:, 1:, 1:] += qs.unsqueeze(-1).unsqueeze(-1) - qvx
    return Rq

def quatinv(q):
    q[1:] = -q[1:]
    return q

def w2qdotkinematics_quat(q, w):
    qs = q[:, 0]
    qv = q[:, 1:]
    bsz = q.shape[0]
    qvx = getqvx(qv, bsz)
    Lq = getLq(qv, qs, qvx, bsz)
    qw = (Lq[:, :, 1:] * w.unsqueeze(1)).sum(dim=-1)/2
    return qdot

def Polynomial(coeffs, a):
    exp = torch.arange(len(coeffs)).unsqueeze(0).to(a)
    basis = a.unsqueeze(-1)**exp
    out = (coeffs.unsqueeze(0)*basis).sum(dim=-1)
    return out


# "Rotate by angle of attack"
def arotate(a,r):
    sa,ca = torch.sin(a), torch.cos(a)
    zeros, ones = torch.zeros_like(a), torch.ones_like(a)
    R1 = torch.stack([ca, zeros, -sa], dim=-1)
    R2 = torch.stack([zeros, ones, zeros], dim=-1)
    R3 = torch.stack([sa, zeros, ca], dim=-1)
    R = torch.stack([R1, R2, R3], dim=1)
    out = (R*r.unsqueeze(1)).sum(dim=-1)
    return out

# "Rotate by sideslip angle"
def brotate(b,r):
    sb,cb = torch.sin(b), torch.cos(b)
    zeros, ones = torch.zeros_like(b), torch.ones_like(b)
    # R = @SMatrix [
    #     cb -sb 0;
    #     sb  cb 0;
    #     0    0 1]
    R1 = torch.stack([cb, -sb, zeros], dim=-1)
    R2 = torch.stack([sb, cb, zeros], dim=-1)
    R3 = torch.stack([zeros, zeros, ones], dim=-1)
    R = torch.stack([R1, R2, R3], dim=1)
    out = (R*r.unsqueeze(1)).sum(dim=-1)
    return out

def mrp2rot(m):
    q = mrp2quat(m)
    R = quat2rot(q)
    return R

def mrp2quat(m):
    sqmag=torch.sum(m*m, dim=-1).unsqueeze(-1)
    q = torch.cat([1-sqmag, 2*m], dim=-1)/(1+sqmag)
    return q

def quat2mrp(q):
    m=q[:,1:]/(1 + q[:,0].unsqueeze(-1))
    return m

def quat2rot(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def euler_to_quaternion(e, order=['x','y','z']):
    """
    Convert Euler angles to quaternions.
    """
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = torch.stack((torch.cos(x/2), torch.sin(x/2), torch.zeros_like(x), torch.zeros_like(x)), axis=1)
    ry = torch.stack((torch.cos(y/2), torch.zeros_like(y), torch.sin(y/2), torch.zeros_like(y)), axis=1)
    rz = torch.stack((torch.cos(z/2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)
            
    # Reverse antipodal representation to have a non-negative "w"
    # if order in ['xyz']:
    result *= -1
    
    return result.reshape(original_shape)

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def w2pdotkinematics_mrp(p, w):
    # From Fundamentals of Spacecraft Attitude Determination and Control, eq (3.24)
    #     here A = (I(3) + 2*(skew(p)^2 + skew(p))/(1+p'p)) * (1+p'p)
    #            = √R * (1 + p'p)
    #     where √R*√R = (√R')*(√R') = RotMatrix(p)
    # p = params(p)
    A1 = torch.stack([1 + p[:,0]**2 - p[:,1]**2 - p[:,2]**2, 2*(p[:,0]*p[:,1] - p[:,2]),      2*(p[:,0]*p[:,2] + p[:,1])], dim=-1)
    A2 = torch.stack([2*(p[:,1]*p[:,0] + p[:,2]),   1-p[:,0]**2+p[:,1]**2-p[:,2]**2, 2*(p[:,1]*p[:,2] - p[:,0])], dim=-1)
    A3 = torch.stack([2*(p[:,2]*p[:,0] - p[:,1]),   2*(p[:,2]*p[:,1] + p[:,0]),      1-p[:,0]**2-p[:,1]**2+p[:,2]**2], dim=-1)
    A = torch.stack([A1, A2, A3], dim=1)
    return 0.25*(A*w.unsqueeze(1)).sum(dim=-1)


def angle_normalize(x, low=-math.pi, high=math.pi):
    return (((x - low) % (high-low)) + low)

