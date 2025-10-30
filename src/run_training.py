import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" ########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from core import *


### Training parameters ###
exact = False #################################
epochs = 10000 #################################
batch_size = 512
lr = 0.001 #####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)

def F(x):
    return 0.1 * torch.sin(2*np.pi * x / params.T)

params = SystemParams()
params.A_L, params.A_F = -1.0, -1.0
params.B_L, params.B_F = 1.0, 1.0
params.sig_L, params.sig_F = 0.1, 0.1
params.M_true = 1.0
params.lamb_F = 1.0
params.lamb_L = 0.75 ############################################
params.Q_L_T = 1.0
params.Q_L, params.Q_F = 1.0, 1.0
params.R_L, params.R_F = 1.0, 1.0
params.X_L_0, params.X_F_0 = 0.1, 0.1
params.T, params.N = 0.5, 50
params.device = device
params.time_grid = torch.linspace(0, params.T, params.N+1, device=params.device)
params.dt = params.T / params.N

leader_analytic = LeaderLBSolver(params, F)

ARCH = "lstm"
net = DeepLSTMController(hidden=64, num_layers=2)

net = train_net(
    net, 
    params,
    F, 
    exact, 
    epochs, 
    batch_size, 
    lr,
    starting_spread=.0,
    plateau_factor=0.5, 
    plateau_patience=50, 
    plateau_thresh=6e-4, 
    min_lr=1e-7, 
    print_every=1
)