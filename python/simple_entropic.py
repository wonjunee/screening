# %%

# %%
import argparse
import os
import numpy as np
import time
import math
import sys
import copy
import matplotlib.pyplot as plt
from screening import HelperClass

from matplotlib.gridspec import GridSpec

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

cuda = True if torch.cuda.is_available() else False
image_folder = 'images-epsilon-0'
os.makedirs(image_folder, exist_ok=True)

# %% [markdown]
# # Importing functions (Poisson solver, finite difference schemes, c-transform, etc)

# %%

# %% [markdown]
# # Defining functions

# %%
# performing c transform
# output: modified psi
def c_transform(psi, phi, cost):
  n,m = psi.shape[0], phi.shape[0]
  psi[:] = torch.min(cost + phi.view((1,m*m)), dim=1).values.view((n,n))
  

def c_transform_forward(phi, psi, cost):
  N = psi.shape[0]
  phi[:] = np.max(- cost + psi.reshape((N,1)), axis=0)

# push forward
def approx_push(nu, psi, phi, cost, epsilon: float, dx: float, dy: float, yMax: float):
  n = psi.shape[0]
  m = phi.shape[0]
  mat   = torch.exp((psi.view((n*n,1)) - phi.view((1,m*m)) - cost) / epsilon) # mat = (n*n, n*n) matrix
  mat  /= torch.sum(mat, dim=1).view((n*n,1)) * (dy*dy)
  nu[:] = torch.sum(mat, dim=0).view((m,m)) * (dx*dx)

# %%
# centered difference
def compute_dx(output, phi, dy):
  m = int(np.sqrt(phi.shape[0]))
  phi = phi.reshape((m,m))
  output[:,1:-1] = (phi[:,2:] - phi[:,:-2])/(2.0*dy)
  output[:,0]    = (phi[:,1] - phi[:,0])/(1.0*dy)
  output[:,-1]   = (phi[:,-1] - phi[:,-2])/(1.0*dy)

# centered difference
def compute_dy(output, phi, dy):
  m = int(np.sqrt(phi.shape[0]))
  phi = phi.reshape((m,m))
  output[1:-1,:] = (phi[2:,:] - phi[:-2,:])/(2.0*dy)
  output[0,:]    = (phi[1,:] - phi[0,:])/(1.0*dy)
  output[-1,:]   = (phi[-1,:] - phi[-2,:])/(1.0*dy)

# parameters
# grid size n x n
n = 128
m = 128

# step size for the gradient ascent
L = 5000

# epsilon for pushforward
eps = 1e-3
max_iteration = 10000000
size = 1.5
xMin = -size
yMin = -size
Xx,Xy =np.meshgrid(np.linspace(0.5/n,1-0.5/n,n), np.linspace(0.5/n,1-0.5/n,n))
Xx = (Xx - 0.5) * 2 * size
Xy = (Xy - 0.5) * 2 * size
Yx,Yy =np.meshgrid(np.linspace(0.5/m,1-0.5/m,m), np.linspace(0.5/m,1-0.5/m,m))
Yx = (Yx - 0.5) * 2 * size
Yy = (Yy - 0.5) * 2 * size

dx = Xx[0,1] - Xx[0,0]
dy = Yx[0,1] - Yx[0,0]

Xv = np.zeros((n*n,2))
Xv[:,0] = Xx.reshape((n*n,))
Xv[:,1] = Xy.reshape((n*n,))

Yv = np.zeros((m*m,2))
Yv[:,0] = Yx.reshape((m*m,))
Yv[:,1] = Yy.reshape((m*m,))

cost_np = ((Xv.reshape((n*n,1,2)) - Yv.reshape((1,m*m,2)))**2).sum(2) * 0.5

print("size of cost: ", cost_np.shape)
b_np       = 0.5 * (Yx**2 + Yy**2) * 0
b_np     = b_np.flatten()
psi_np = 0.5 * (Xx**2 + Xy**2) - 0.9 * (Xx**2 + Xy**2) ** 0.5
psi_np     = psi_np.flatten()
phi_np     = np.zeros((m*m)).astype('float64')
nu_np      = np.zeros((m*m)).astype('float64')
psi_eps_np = np.zeros((n*n)).astype('float64')
rho_np     = np.zeros((m*m)).astype('float64')
Tx         = np.zeros((n,n)).astype('float64')
Ty         = np.zeros((n,n)).astype('float64')

# phi  = torch.from_numpy(phi).type(torch.float32)
cost = torch.from_numpy(cost_np)
phi  = torch.from_numpy(phi_np)
psi  = torch.from_numpy(psi_np)
nu   = torch.from_numpy(nu_np)
psi_eps   = torch.from_numpy(psi_eps_np)

mu_np  = np.zeros((n,n)).astype('float64')
mu_radius = 1.0
mu_np[(Xx**2 + Xy**2 < mu_radius**2)] = 1
mu_np /= mu_np.sum() * dx * dx
mu_np  = mu_np.flatten()

c_transform_forward(phi_np, psi_np, cost_np)
# c_transform_forward_cpp(phi_np, psi_np, cost_np)

fig,ax = plt.subplots(1,2,figsize=(10,5))

# c_transform_epsilon_cpp(psi_eps_np, psi_np, phi_np, cost_np, eps, dx, dy, yMax)
# compute_nu_and_rho_cpp(nu_np, rho_np, psi_eps_np, phi_np,  cost_np, b_np, epsilon, dx, dy, yMax)

def compute_ctransform_eps(psi_eps_np, psi_np, phi_np, cost_np, eps, dy):
  psi_eps_np[:] = (np.exp( (  - phi_np.reshape((1,-1)) - cost_np)/eps ) * dy*dy).sum(axis=1)
  psi_eps_np[:] = - eps * np.log(psi_eps_np)

compute_ctransform_eps(psi_eps_np, psi_np, phi_np, cost_np, eps, dy)
# plan = np.exp( (psi_eps_np.reshape((n*n,1)) - phi_np.reshape((1,m*m)) - cost_np)/eps )
# nu_np[:] = (plan * mu_np.reshape((n*n,1))).sum(axis=0) * dx*dx

plan = np.exp( (psi_eps_np.reshape((-1,1)) - phi_np.reshape((1,-1)) - cost_np)/eps )
# plan /= plan.sum(1).reshape((n*n,1))* dy * dy
nu_np[:] = (plan * mu_np.reshape((n*n,1))).sum(0) * dx*dx

ax[0].contourf(Yx,Yy,nu_np.reshape((m,m)))
ax[0].set_title(f'sum: {np.sum(nu_np)*dy*dy}')
ax[1].imshow(plan, origin='lower')
ax[1].set_title(f'sum: {plan.sum()*dx*dx*dy*dy}')

plt.show()

u = np.zeros((n,n))

if cuda:
  phi  = phi.cuda()
  psi  = psi.cuda()
  cost = cost.cuda()
  nu   = nu.cuda()
it = 0

# %%
J_list = []
L = 1000
eps = 1e-3

start_time = time.time()

while it < max_iteration:
  # c_transform(psi_np, phi_np, cost)
  # c_transform_epsilon_cpp(psi_eps_np, psi_np, phi_np, cost_np, eps, dx, dy, yMax)
  compute_ctransform_eps(psi_eps_np, psi_np, phi_np, cost_np, eps, dy)
  plan = np.exp( (psi_eps_np.reshape((-1,1)) - phi_np.reshape((1,-1)) - cost_np)/eps )
  # plan /= plan.sum(1).reshape((n*n,1))* dy * dy
  nu_np[:] = (plan * mu_np.reshape((n*n,1))).sum(0) * dx*dx

  b_phi = phi_np/eps
  xi = plan @ (b_phi.reshape((-1,1))) * dy*dy

  phi_np += 1.0/L * ( ((1.0 - b_phi)*nu_np).flatten() + ((xi.reshape((1,-1)) * mu_np.reshape((1,-1)))@plan).flatten() * dx * dx )
  phi_np.reshape((m,m))[(Yx**2+Yy**2)>mu_radius**2] = 0
  
  J_val = (phi_np * nu_np).sum() * dy*dy
  J_list.append(J_val)

  if it % 100 == 0:
    # for plotting point cloud version of T(x)
    def c_transform(psi, phi, cost):
      psi[:] = np.min(cost + phi.reshape((1,-1)), axis=1).flatten()
    c_transform(psi_eps_np, phi_np, cost_np)
    compute_dx(Tx, psi_eps_np, dx); compute_dy(Ty, psi_eps_np, dx); Tx = Xx-Tx; Ty = Xy-Ty

    # for plotting
    # fig,ax = plt.subplots(nrows=1,ncols=5,layout='constrained')

    fig = plt.figure(constrained_layout=True, figsize=(16,4))
    gs = GridSpec(1, 5, figure=fig)

    ax = fig.add_subplot(gs[0])
    rho_plot = np.log(1+nu_np)
    ax.contour(Yx,Yy,rho_plot.reshape(m,m), 60)
    ax.set_title(f"rho {np.min(nu_np):0.2e}, {np.max(nu_np):0.2e}")
    ax.grid()
    ax.set_aspect('equal')

    ax = fig.add_subplot(gs[1])
    ax.scatter(Tx,Ty,marker='o',alpha=0.2)
    ax.set_xlim([xMin,-xMin])
    ax.set_ylim([yMin,-yMin])
    ax.grid()
    ax.set_aspect('equal')

    ax = fig.add_subplot(gs[2])
    ax.contourf(Yx,Yy,phi_np.reshape(m,m))
    ax.set_title(f"phi {np.min(phi_np):0.2e}, {np.max(phi_np):0.2e}")
    ax.set_aspect('equal')

    ax = fig.add_subplot(gs[3])
    ax.contourf(Xx,Xy,psi_eps_np.reshape(n,n))
    ax.set_title(f"psi {np.min(psi_eps_np):0.2e}, {np.max(psi_eps_np):0.2e}")
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(gs[4])
    ax.plot(J_list, '-')
    # ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(f"J={J_val:0.2e}")
    ax.grid()
    
    plt.suptitle(f"it={it}")
    filename = f"{image_folder}/{it//100:03d}.png"
    plt.savefig(filename)
    plt.close('all')
    print(f"it: {it} J: {J_val:0.2e} time: {time.time() - start_time:0.2} s  filename: {filename}")
    
    np.save( f"{image_folder}/phi.npy", phi_np)
  
  it += 1

# %%


# %%
fig,ax = plt.subplots(1,5,figsize=(24,4))
rho_plot = np.log(1+rho_np)
ax[0].contour(Yx,Yy,rho_plot.reshape(m,m), 60)
ax[0].set_title(f"rho {np.min(rho_np):0.2e}, {np.max(rho_np):0.2e}")
ax[0].grid()

ax[1].scatter(Tx,Ty,marker='.')
ax[1].set_aspect('equal')
ax[1].set_xlim([0,2.1])
ax[1].set_ylim([0,2.1])

ax[2].contourf(Yx,Yy,phi_np.reshape(m,m))
ax[2].set_title(f"phi {np.min(phi_np):0.2e}, {np.max(phi_np):0.2e}")
ax[3].contourf(Xx,Xy,psi_eps_np.reshape(n,n))
ax[3].set_title(f"psi {np.min(psi_eps_np):0.2e}, {np.max(psi_eps_np):0.2e}")
ax[4].plot(J_list, '.-')
ax[4].set_title(f"Loss plot")