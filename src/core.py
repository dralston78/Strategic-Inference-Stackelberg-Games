import json
import os
import random
from datetime import datetime
from wordfreq import top_n_list
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass

@dataclass
class SystemParams():
    A_L, A_F = None, None
    B_L, B_F = None, None
    sig_L, sig_F = None, None
    M_true = None
    lamb_F = None
    lamb_L = None
    Q_L, Q_F = None, None
    Q_L_T = None
    R_L, R_F = None, None
    X_L_0, X_F_0 = None, None
    T = None
    N = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = None
    time_grid = None

class DeepLSTMController(nn.Module):
    def __init__(self, hidden=48, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size, self.num_layers = hidden, num_layers
        self.rnn = nn.LSTM(2, hidden, num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.SiLU(), nn.Linear(hidden//2, 1))
        self.h = None

    def reset_state(self, B):
        z = torch.zeros(self.num_layers, B, self.hidden_size, device=self.head[0].weight.device)
        self.h = (z.clone(), z.clone())

    def forward(self, tau, x):
        if self.h is None or self.h[0].size(1) != tau.size(0): self.reset_state(tau.size(0))
        out, self.h = self.rnn(
            torch.stack([tau, x], -1).unsqueeze(1), self.h
        )
        return self.head(out.squeeze(1)).squeeze(-1)
    
def compute_a_grid(params):
    if getattr(params, 'a_grid', None) is not None:
        return params.a_grid
    else:
        def rhs(t, a): return (2 * params.B_F**2 / params.R_F) * a**2 - 2 * params.A_F * a - params.Q_F / 2
        sol = solve_ivp(rhs, (params.T, 0.0), [0.0],
                        t_eval=np.linspace(params.T, 0.0, params.N + 1),
                        rtol=1e-8, atol=1e-8)
        # return torch.tensor(sol.y[0][::-1].copy(), dtype=torch.float32, device=params.device)
        params.a_grid = torch.tensor(sol.y[0][::-1].copy(), dtype=torch.float32, device=params.device)
        return params.a_grid

def compute_f_grid(params):
    a_grid = compute_a_grid(params)
    f_grid = params.A_F - 2 * (params.B_F**2) / params.R_F * a_grid
    return f_grid

def compute_b_grid(params, a_grid, X_L):
    B, L = X_L.shape
    b_next = X_L.new_zeros(B)
    vals = [b_next]
    for k in range(L - 1, 0, -1):
        db = (2 * params.B_F**2 / params.R_F) * a_grid[k] * b_next - params.A_F * b_next + params.Q_F * params.M_true * X_L[:, k]
        b_next = b_next - params.dt * db
        vals.append(b_next)
    return torch.stack(vals[::-1], 1)

def make_nn_policy(net, params):
    def policy(k, x):
        if k == 0:
            if hasattr(net, "reset_state"):
                net.reset_state(x.size(0))
        tau = torch.full_like(x, (k * params.dt) / params.T)
        return net(tau, x)
    return policy


class LeaderLBSolver:
    def __init__(self, params, F):
        self.params = params
        self.F_grid = F(params.time_grid)
        self.f_grid = compute_f_grid(params)

        self._build_static_tensors()
        self._compute_ric_functions()                        
        
    def _build_static_tensors(self):
        params = self.params
        self.Bv = torch.tensor([[params.B_L], [0.0], [0.0]], device=params.device)
        self.BBT = self.Bv @ self.Bv.T 
        self.B_flat = self.Bv.squeeze()

        self.Cv = torch.tensor([[params.sig_L], [0.0], [0.0]], device=params.device)
        self.C_flat = self.Cv.squeeze()

        F_int = torch.cumsum(self.f_grid[:-1], 0) * params.dt
        F_int = torch.cat([torch.tensor([0.0], device=params.device), F_int])
        self.h_grid = params.Q_F * torch.exp(F_int)
        self.k_grid = torch.exp(-2.0 * F_int)

        self.Q_diag = torch.tensor(
            [[0.5 * params.Q_L, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=params.device,
        )

    def _A_mat(self, j):
        """\mathcal{A}(t_j)"""
        params = self.params
        return torch.tensor(
            [
                [params.A_L, 0.0, 0.0],
                [-self.h_grid[j], 0.0, 0.0],
                [0.0, self.k_grid[j], 0.0],
            ],
            device=params.device,
        )

    def _compute_ric_functions(self):
        params = self.params
        # N, dt, device = params.N, params.dt, params.device
        # lam_L, R_L = params.lam_L, params.R_L

        # allocate
        self.L_t = torch.zeros(params.N + 1, 3, 3, device=params.device)
        self.M_t = torch.zeros(params.N + 1, 3, device=params.device)
        self.N_t = torch.zeros(params.N + 1, 1, device=params.device)

        int_K = int_trapz(self.k_grid, dx=params.dt)  # ∫_0^T k(t) dt

        L_T = torch.zeros(3, 3, device=params.device)
        L_T[0, 0] = 0.5 * params.Q_L_T
        L_T[1, 1] = -(params.B_F**4 / (params.sig_F**2 * params.R_F**2)) * params.lamb_L * int_K
        L_T[1, 2] = (params.B_F**4 / (params.sig_F**2 * params.R_F**2)) * params.lamb_L
        L_T[2, 1] = (params.B_F**4 / (params.sig_F**2 * params.R_F**2)) * params.lamb_L
        self.L_t[-1] = L_T

        self.M_t[-1] = torch.tensor(
            [-params.Q_L_T * self.F_grid[-1], 0.0, 0.0], device=params.device
        )
        self.N_t[-1] = 0.5 * params.Q_L_T * self.F_grid[-1] ** 2

        for j in range(params.N, 0, -1):
            Lj, Mj, Aj = self.L_t[j], self.M_t[j], self._A_mat(j)
            Qj = self.Q_diag.clone()
            Qj[1, 1] = -(params.B_F**4 / (params.sig_F**2 * params.R_F**2)) * params.lamb_L * self.k_grid[j]
            dL = -(
                Lj @ Aj
                + Aj.T @ Lj
                - (2.0 / params.R_L) * Lj @ self.BBT @ Lj
                + Qj
            )
            self.L_t[j - 1] = Lj - params.dt * dL
            vec_F = torch.tensor([-params.Q_L * self.F_grid[j], 0.0, 0.0], device=params.device)
            dM = -(
                Aj.T @ Mj
                - (2.0 / params.R_L) * Lj @ self.BBT @ Mj
                + vec_F
            )
            self.M_t[j - 1] = Mj - params.dt * dM
            BT_M = (self.B_flat * Mj).sum()
            dN = (
                (BT_M**2) / (2.0 * params.R_L)
                - self.Cv.T @ Lj @ self.Cv
                - 0.5 * params.Q_L * self.F_grid[j] ** 2
            )
            self.N_t[j - 1] = self.N_t[j] - params.dt * dN

        asym = (self.L_t - self.L_t.transpose(-1, -2)).abs().max()
        if asym > 1e-4:
            raise ValueError(f"L_t not symmetric; max asym = {asym:.2e}")

    def u_star(self, k, r):
        params = self.params
        proj = 2.0 * torch.einsum("ij,kj->ki", self.L_t[k], r) + self.M_t[k]
        return -(1.0 / params.R_L) * torch.einsum("ki,i->k", proj, self.B_flat)
    
    def simulate(self, batch, seed):
        torch.manual_seed(seed)
        params = self.params
        control = self.u_star

        Xs = torch.empty(batch, params.N + 1, device=params.device)
        Us = torch.empty(batch, params.N, device=params.device)

        X = torch.full((batch,), params.X_L_0, device=params.device)
        Y = torch.zeros(batch, device=params.device)
        Z = torch.zeros(batch, device=params.device)
        Xs[:, 0] = X

        dW = torch.randn((batch, params.N), device=params.device) * (params.dt**0.5)

        for k in range(params.N):
            r = torch.stack((X, Y, Z), dim=1) 
            u = control(k, r).reshape(-1)
            Us[:, k] = u

            Y = Y + (-self.h_grid[k] * X) * params.dt
            Z = Z + self.k_grid[k] * Y * params.dt
            X += (params.A_L * X + params.B_L * u) * params.dt + params.sig_L * dW[:, k]
            Xs[:, k + 1] = X
        return Xs, Us

def exact_leader_total_cost(X_L, U_L, params, F):
    """Variance minimization task"""
    F_grid = F(torch.arange(params.N + 1, device=params.device))
    a_grid = compute_a_grid(params)
    b = compute_b_grid(params, a_grid, X_L)
    g = b / params.M_true
    cost = (params.B_F**4 / (params.sig_F**2 * params.R_F**2)) * (params.lamb_L / int_trapz(g**2, dx=params.dt)).mean() \
            + (params.Q_L / 2) * int_trapz((X_L - F_grid)**2, dx=params.dt).mean() \
            + (params.R_L / 2) * int_trapz(U_L**2, dx=params.dt).mean()
    return cost

def lb_leader_total_cost(X_L, U_L, params, F):
    """Fisher Information Maximization task"""
    F_grid = F(torch.arange(params.N + 1, device=params.device))
    a_grid = compute_a_grid(params)
    b = compute_b_grid(params, a_grid, X_L)
    g = b / params.M_true
    cost = -(params.B_F**4 / (params.sig_F**2 * params.R_F**2)) * params.lamb_L * int_trapz(g**2, dx=params.dt).mean() \
            + (params.Q_L / 2) * int_trapz((X_L - F_grid)**2, dx=params.dt).mean() \
            + (params.R_L / 2) * int_trapz(U_L**2, dx=params.dt).mean()
    return cost

def simulate_leader(control, params, batch, seed):
    torch.manual_seed(seed)
    dW = torch.randn(batch, params.N, device=params.device) * (params.dt**0.5)
    X_L = torch.empty(batch, params.N + 1, device=params.device)
    U_L = torch.empty(batch, params.N, device=params.device)
    X_L[:, 0] = torch.full((batch,), params.X_L_0, device=params.device)
    
    for k in range(params.N):
        u = control(k, X_L[:, k])
        U_L[:, k] = u

        X_L[:, k + 1] = X_L[:, k] + (params.A_L * X_L[:, k] + params.B_L * u) * params.dt + params.sig_L * dW[:, k]
    return X_L, U_L

def simulate_follower(X_L, params, batch, seed):
    """Use exploratory policy of the follower"""
    torch.manual_seed(seed)
    X_F = torch.empty(batch, params.N + 1, device=params.device)
    X_F[:, 0] = torch.full((batch,), params.X_F_0, device=params.device)
    U_F = torch.empty(batch, params.N, device=params.device)
    dW = torch.randn(batch, params.N, device=params.device) * (params.dt**0.5)
    a_grid = compute_a_grid(params)
    b_grid = compute_b_grid(params, a_grid, X_L)
    for k in range(params.N):
        U_F[:, k] = torch.tensor((-params.B_F / params.R_F) * (2 * a_grid[k] * X_F[:, k] + b_grid[:, k]), device=params.device)
        X_F[:, k + 1] = X_F[:, k] + (params.A_F * X_F[:, k] + params.B_F * U_F[:, k]) * params.dt + params.sig_F * dW[:, k]
    return X_F, U_F

def int_trapz(y, dx):
    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    if len(y.shape) > 2:
        raise ValueError("Input tensor must be 1D or 2D.")
    return dx * (0.5 * (y[:, 0] + y[:, -1]) + y[:, 1:-1].sum(dim=1))

def train_net(net, params, F, exact, epochs, batch_size, lr, starting_spread, plateau_factor, plateau_patience, plateau_thresh, min_lr, print_every):

    common_words = top_n_list('en', n=20000,)
    run_words = random.sample(common_words, 2)
    run_id = "_".join(run_words)
    base_dir = "src/runs"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)

    params_serializable = {}
    for (key, value) in vars(params).items():
        if isinstance(value, torch.Tensor):
            continue
        elif isinstance(value, torch.device):
            params_serializable[key] = str(value)
        else: 
            try:
                json.dumps(value)
                params_serializable[key] = value
            except TypeError:
                params_serializable[key] = str(value)
    
    run_details = {
        "system_params": params_serializable,
        "training_params": {
            "exact":exact,
            "epochs":epochs,
            "batch_size":batch_size,
            "lr":lr,
            "starting_spread":starting_spread,
            "plateau_factor":plateau_factor,
            "plateau_patience": plateau_patience,
            "plateau_thresh":plateau_thresh,
            "min_lr":min_lr,
            "print_every":print_every
        },
        "net_architecture":str(net),
        "run_time":datetime.now().isoformat()
    }
    with open(os.path.join(run_dir, "run_details.json"), "w") as f:
        json.dump(run_details, f, indent=4)


    net.to(params.device)
    opt = optim.AdamW(net.parameters(), lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=plateau_factor, patience=plateau_patience,
        threshold=plateau_thresh, min_lr=min_lr, verbose=False)

    # w_pen = 0.1 + 0.9 * torch.arange(N, device=device) / N
    w_pen = 1
    losses = []
    best_losses = []
    for upd in range(1, epochs + 1):
        opt.zero_grad()
        dW = torch.randn(batch_size, params.N, device=params.device) * params.dt**0.5
        X = torch.empty((batch_size, 1), device=params.device).uniform_(params.X_L_0 - starting_spread, params.X_L_0 + starting_spread)
        U = []
        if hasattr(net, 'reset_state'):
            net.reset_state(batch_size)

        for k in range(params.N):
            xk = X[:, -1]
            u_k = net(torch.full_like(xk, (k * params.dt) / params.T), xk)
            U.append(u_k)
            X = torch.cat([X, (xk + (params.A_L * xk + params.B_L * u_k) * params.dt + params.sig_L * dW[:, k]).unsqueeze(1)], 1)

        U = torch.stack(U, 1)
        if exact:
            loss = exact_leader_total_cost(X, U, params, F)
        else:
            loss = lb_leader_total_cost(X, U, params, F)
        loss.backward()
            
        clip_grad = 1.0
        nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        opt.step()

        loss_val = None
        if upd % print_every == 0:
            with torch.no_grad():
                X_eval, U_eval = simulate_leader(make_nn_policy(net, params), params, batch=10000, seed=42)
                if exact:
                    loss_val = exact_leader_total_cost(X_eval, U_eval, params, F)
                    print(f"[{upd}] MC‑Exact = {loss_val:+.3e} LR = {opt.param_groups[0]['lr']:.2e} ")
                else:
                    loss_val = lb_leader_total_cost(X_eval, U_eval, params, F)
                    print(f"[{upd}] MC‑LB = {loss_val:+.3e} LR = {opt.param_groups[0]['lr']:.2e} ")

        loss_val = loss_val.item()
        if len(best_losses) < 10 or loss_val < best_losses[-1][0]:
            sd_clone = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_losses.append((loss_val, upd, sd_clone))
            best_losses = sorted(best_losses, key=lambda x: x[0])[:10]

        sched.step(loss_val)
        losses.append(loss_val)


    for (loss_val, upd, sd_clone) in best_losses:
        fname = f"loss_{loss_val:.3e}_upd_{upd}.pt"
        torch.save(sd_clone, os.path.join(run_dir, fname))

    torch.save(net.state_dict(), os.path.join(run_dir, "final_weights.pt"))
    # save losses
    with open(os.path.join(run_dir, "losses.json"), "w") as f:
        json.dump({
            "losses": [float(loss) for loss in losses],
        }, f, indent=4)
    return net

@torch.no_grad()
def estimate_I_batch(X_L_paths, params):
    # 1 / variance of hat{M}
    a_grid = compute_a_grid(params)
    b = compute_b_grid(params, a_grid, X_L_paths)
    g2 = (b / params.M_true) ** 2
    I_paths = (params.B_F**4)/(params.sig_F**2 * params.R_F**2) * int_trapz(g2, dx=params.dt)
    return I_paths.mean().item()
