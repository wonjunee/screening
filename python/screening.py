# %%
# Run if you are on Google Colab to install the Python bindings
import os
os.system('bash compile.sh')

# %%
import os
import numpy as np
from scipy.fftpack import dctn, idctn
import tqdm
import matplotlib.pyplot as plt
from screening import HelperClass

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
image_folder = "images-other4"
os.makedirs(image_folder, exist_ok=True)
desc = "using anti identity"

# %%
# Initialize Fourier kernel
def initialize_kernel(n1, n2, dy):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    # kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel = 2*(1-np.cos(xx))/(dy*dy) + 2*(1-np.cos(yy))/(dy*dy)
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel

# 2d DCT
def dct2(a):
    return dctn(a, norm='ortho')

# 2d IDCT
def idct2(a):
    return idctn(a, norm='ortho')

# Solving Poisson
#   - Δ u = f
#   output: u = (-Δ)⁻¹ f
def solve_poisson(u, f, kernel):
    n = u.shape[0]
    u[:] = 0
    workspace = np.copy(f)
    workspace = dct2(workspace) / kernel
    workspace[0,0] = 0
    u += idct2(workspace)

def solve_poisson_bdry(u, f, bdry, kernel):
    gx0, gx1, gy0, gy1 = bdry
    n = u.shape[0]
    u[:] = 0
    workspace = np.copy(f)
    workspace[0,:]   -= gy0 * n
    workspace[n-1,:] += gy1 * n
    workspace[:,0]   -= gx0 * n
    workspace[:,n-1] += gx1 * n
    workspace = dct2(workspace) / (1+kernel)
    workspace[0,0] = 0
    u += idct2(workspace)


# %% [markdown]
# # c-transform
# 
# $$ \phi^c(x) = \inf_y c(x,y) + \phi(y) $$
# 
# $$ c(x,y) = \frac12 |y|^2$$
# 
# $$ \phi^c(x) = \inf_y \frac12 |y|^2 + \phi(y) $$

# %%
# centered difference
def compute_dx(output, phi, dy):
  output[:,1:-1] = (phi[:,2:] - phi[:,:-2])/(2.0*dy)
  # output[:,0]    = (phi[:,1] - phi[:,0])/(1.0*dy)
  # output[:,-1]   = (phi[:,-1] - phi[:,-2])/(1.0*dy)

# centered difference
def compute_dy(output, phi, dy):
  output[1:-1,:] = (phi[2:,:] - phi[:-2,:])/(2.0*dy)
  # output[0,:]    = (phi[1,:] - phi[0,:])/(1.0*dy)
  # output[-1,:]   = (phi[-1,:] - phi[-2,:])/(1.0*dy)

# %%
def compute_dx_forward(phi, tmp, dy):
  A = np.zeros_like(phi)
  A[:,:-1] = (phi[:,1:] - phi[:,:-1])/(1.0*dy)
  A[:,-1]  = (phi[:,-1] - phi[:,-2])/(1.0*dy)
  return A

def compute_dx_back(phi, tmp, dy):
  A = np.zeros_like(phi)
  A[:,1:] = (phi[:,1:] - phi[:,:-1])/(1.0*dy)
  A[:,0]  = (phi[:,1]  - phi[:,0])/(1.0*dy)
  return A

def compute_dxx(phi, tmp, dy):
  # n = phi.shape[0]
  # tmp[1:-1,0]    = phi[:,0]
  # tmp[1:-1,-1]   = phi[:,-1]
  # tmp[1:-1,1:-1] = phi
  # return (tmp[1:-1,2:] + tmp[1:-1,:-2] - 2.0 * tmp[1:-1,1:-1])/(dy*dy)

  A = compute_dx_forward(phi, tmp, dy)
  return compute_dx_back(A, tmp, dy)

def compute_dy_forward(phi, tmp, dy):
  A = np.zeros_like(phi)
  A[:-1,:] = (phi[1:,:] - phi[:-1,:])/(1.0*dy)
  A[-1,:]  = (phi[-1,:] - phi[-2,:])/(1.0*dy)
  return A

def compute_dy_back(phi, tmp, dy):
  A = np.zeros_like(phi)
  A[1:,:] = (phi[1:,:] - phi[:-1,:])/(1.0*dy)
  A[0,:]  = (phi[1,:] - phi[0,:])/(1.0*dy)
  return A

def compute_dyy(phi, tmp, dy):
  # n = phi.shape[0]
  # tmp[0,1:-1]    = phi[0,:]
  # tmp[-1,1:-1]   = phi[-1,:]
  # tmp[1:-1,1:-1] = phi
  # return (tmp[2:,1:-1] + tmp[:-2,1:-1] - 2.0 * tmp[1:-1,1:-1])/(dy*dy)

  A = compute_dy_forward(phi, tmp, dy)
  return compute_dy_back(A, tmp, dy)

# centered difference
def compute_dxy(phi, tmp, dy):
  # n = phi.shape[0]
  # tmp[0,0]     = phi[0,0]
  # tmp[0,-1]    = phi[0,-1]
  # tmp[-1,0]    = phi[-1,0]
  # tmp[-1,-1]   = phi[-1,-1]
  # tmp[1:-1,0]  = phi[:,0]
  # tmp[1:-1,-1] = phi[:,-1]
  # tmp[0,1:-1]  = phi[0,:]
  # tmp[-1,1:-1] = phi[-1,:]
  # tmp[1:-1,1:-1] = phi
  # return (tmp[2:,2:] - tmp[2:,:-2] - tmp[:-2,2:] + tmp[:-2,:-2])/(4.0*dy*dy)
  # A = compute_dx_back(phi, tmp, dy)
  A = np.zeros_like(phi)
  A[:-1,:-1] = (phi[1:,1:] - phi[1:,:-1] - phi[:-1,1:] + phi[:-1,:-1])/(dy*dy)
  A[-1,:-1]  = - (phi[-1,:-1] - phi[-1,1:] - phi[-2,:-1] + phi[-2,1:])  /(dy*dy)
  A[:-1,-1]  = - (phi[:-1,-1] - phi[:-1,-2] - phi[1:,-1] + phi[1:,-2])/(dy*dy)
  A[-1,-1]   = (phi[-1,-1] - phi[-1,-2] - phi[-2,-1] + phi[-2,-2])/(dy*dy)
  return A

# %%
from screening import c_transform_cpp, c_transform_forward_cpp,compute_first_variation_cpp #, approx_push_cpp

# performing c transform
# output: modified psi
def c_transform(psi, phi, cost):
  n = psi.shape[0]
  m = phi.shape[0]
  psi[:] = np.min(cost + phi.reshape((1,m*m)), axis=1).reshape((n,n))

def c_transform_eps(psi, phi, cost, epsilon: float, dy: float):
  n = psi.shape[0]
  m = phi.shape[0]
  psi[:] = - epsilon * np.log(( np.exp((- phi.reshape((1,n*n)) - cost) / epsilon) ).sum(axis=1)* (dy*dy) ).reshape((n,n))  # mat = (n*n, n*n) matrix

def c_transform_forward(phi, psi, cost):
  phi[:] = np.max(- cost + psi.reshape((-1,1)), axis=0)

# push forward
def approx_push(nu, psi, phi, cost, epsilon: float, dx: float, dy: float):
  n = psi.shape[0]
  m = phi.shape[0]
  mat   = np.exp((psi.reshape((n*n,1)) - phi.reshape((1,m*m)) - cost) / epsilon) # mat = (n*n, n*n) matrix
  mat  /= np.sum(mat, axis=1).reshape((n*n,1)) * (dy*dy)
  nu[:] = np.sum(mat, axis=0).reshape((m,m)) * (dx*dx)

def compute_ctransform_eps(psi_eps_np, phi_np, cost_np, eps, dy):
  psi_eps_np[:] = (np.exp( (-phi_np.reshape((1,-1)) - cost_np)/eps ) * dy*dy).sum(axis=1)
  psi_eps_np[:] = - eps * np.log(psi_eps_np + 1e-6)

def compute_rhs(phi_np, psi_np, nu_np, b, helper, dx, dy, n, m, show_image=False):

  #   initialize fx, fy
  fx = np.zeros((m,m))
  fy = np.zeros((m,m))
  phi_b = phi_np - b
  compute_dx(fx, phi_b.reshape((m,m)), dy)
  compute_dy(fy, phi_b.reshape((m,m)), dy)

  R1  = np.zeros((m,m))
  R2  = np.zeros((m,m))
  R1x = np.zeros((m,m))
  R2y = np.zeros((m,m))

  helper.compute_inverse_g(R1, R2, phi_np, psi_np, fx, fy)

  R1 *= nu_np.reshape((m,m))
  R2 *= nu_np.reshape((m,m))

  compute_dx(R1x, R1, dy)
  compute_dy(R2y, R2, dy)

  gx0 = R1[:,0]
  gx1 = R1[:,-1]

  gy0 = R2[0,:]
  gy1 = R2[-1,:]

  return nu_np.reshape((m,m)) + R1x + R2y, [-gx0,-gx1,-gy0,-gy1]


def solve_main_poisson(u, phi_np, psi_np, nu_np, b, kernel, helper, dx, dy, yMax, n, m, show_image=False):
  rhs, bdry = compute_rhs(phi_np, psi_np, nu_np, b, helper, dx, dy, n, m, show_image=show_image) # computing the right hand side
  solve_poisson_bdry(u,rhs,bdry,kernel)
  return rhs

# %%
# parameters
# grid size n x n
n = 40
m = 60

# step size for the gradient ascent
sigma = 1e-5

# epsilon for pushforward
eps = 1e-2
max_iteration = 100000
Xx,Xy =np.meshgrid(np.linspace(1+0.5/n,2-0.5/n,n), np.linspace(1+0.5/n,2-0.5/n,n))
yMax = 2.5
Yx,Yy =np.meshgrid(np.linspace(yMax*0.5/m,yMax*(1-0.5/m),m), np.linspace(yMax*0.5/m,yMax*(1-0.5/m),m))

dx = 1.0/n
dy = 1.0/m * yMax

kernel = initialize_kernel(m, m, dy)

Xv = np.zeros((n*n,2))
Xv[:,0] = Xx.reshape((n*n,))
Xv[:,1] = Xy.reshape((n*n,))

Yv = np.zeros((m*m,2))
Yv[:,0] = Yx.reshape((m*m,))
Yv[:,1] = Yy.reshape((m*m,))

cost = - np.sum(Xv.reshape((n*n,1,2)) * Yv.reshape((1,m*m,2)),axis=2)
print("size of cost: ", cost.shape)
b = 0.5 * (Yx**2 + Yy**2)
b = b.astype('float64').flatten()  

psi_np = (- 0.5*(Xx**2+Xy**2)).astype('float64')
psi_np = psi_np.flatten()
phi_checking = np.zeros((m*m))
c_transform_forward_cpp(phi_checking, psi_np, cost)
plt.imshow(phi_checking.reshape((m,m)),origin='lower')
plt.savefig("sanity_check_c_transform.png")
plt.close('all')

phi_np = np.zeros((m*m)).astype('float64')
nu_np  = np.zeros((m*m)).astype('float64')

c_transform_forward(phi_np, psi_np, cost)

plan = np.exp( (psi_np.reshape((-1,1)) - phi_np.reshape((1,-1)) - cost)/eps )
plan /= plan.sum() * dx * dx * dy * dy
nu_np[:] = plan.sum(axis=0) * dx*dx

plt.contour(Yx,Yy,nu_np.reshape((m,m)), 120, origin='lower')
plt.title(f"sum: {np.sum(nu_np) * dy * dy}")
plt.savefig("sanity_check_push_forward.png")
plt.close('all')

helper = HelperClass(psi_np, phi_np, dx, dy, n, m)

u = np.zeros((m,m)).astype('float64')

# %%

# phi_np = np.load(f'{image_folder}/phi.npy')

# fig,ax = plt.subplots(1,4,figsize=(14,4))
pbar = tqdm.tqdm(range(1000000))

J_list = []
eps= 1e-2

Tx = np.zeros((n,n))
Ty = np.zeros((n,n))

rhs = np.zeros((m,m))

for it in pbar:
  # c transform
  # c_transform(psi_np, phi_np, cost)
  compute_ctransform_eps(psi_np, phi_np, cost, eps, dy)

  # print(f"min: {torch.min(psi_np.view((n*n,1)) - phi_np.view((1,n*n)) - cost)} max: {torch.max(psi_np.view((n*n,1)) - phi_np.view((1,n*n)) - cost)}")

  # pushforward mu -> nu
  plan = np.exp( (psi_np.reshape((-1,1)) - phi_np.reshape((1,-1)) - cost)/eps )
  plan /= plan.sum() * dx * dx * dy * dy
  nu_np[:] = plan.sum(axis=0) * dx*dx

  rhs = solve_main_poisson(u, phi_np, psi_np, nu_np, b, kernel, helper, dx, dy, yMax, n, m, show_image = (it%10==0))
  error = np.mean(u**2)
  phi_np += sigma * u.flatten()
  phi_np[0] = 0

  # find the value of J
  J_val = ((phi_np - b) * nu_np).sum() * dy*dy
  J_list.append(J_val)

  # fdfd
  compute_dx(Tx, psi_np.reshape((n,n)), dx)
  compute_dy(Ty, psi_np.reshape((n,n)), dx)

  Tx = -Tx; Ty = -Ty

  skip = 100
  if it % skip == 0:
    np.save( f"{image_folder}/phi.npy", phi_np)
    # approx_push(nu_np, psi_np, phi_np, cost, 1e-3, dx, dy)
    plan = np.exp( (psi_np.reshape((-1,1)) - phi_np.reshape((1,-1)) - cost)/eps )
    plan /= plan.sum() * dx * dx * dy * dy
    nu_np[:] = plan.sum(axis=0) * dx*dx
    fig,ax = plt.subplots(1,6,figsize=(18,4),constrained_layout=True)
    ax[0].contourf(Yx,Yy,np.log(1+np.log(1+nu_np)).reshape((m,m)),60)
    ax[0].set_title("nu")
    ax[1].contourf(Yx,Yy,phi_np.reshape((m,m)))
    ax[1].set_aspect('equal')
    ax[1].set_title("phi")
    ax[2].contourf(Xx,Xy,psi_np.reshape((n,n)))
    ax[2].set_aspect('equal')
    ax[2].set_title("psi")
    ax[3].imshow(u,origin='lower')
    ax[3].set_title("u")
    ax[4].plot(J_list, 'o-')
    ax[4].set_title("J values")
    ax[5].scatter(Tx,Ty,marker='.',alpha=0.4)
    ax[5].set_aspect('equal')
    ax[5].set_title('Tx')
    plt.suptitle(f"it={it} error={error:0.2e}\nsigma: {sigma:0.2e}\n{desc}")
    filename = f"{image_folder}/{it//skip:03d}.png"
    plt.savefig(filename)
    plt.savefig(f"{image_folder}/000-status.png")
    plt.close('all')
    pbar.set_description(filename)

# %%
# !rm images/*



