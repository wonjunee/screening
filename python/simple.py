# %%
import os
# Run if you are on Google Colab to install the Python bindings
os.system('bash compile.sh')
import argparse
import numpy as np
from scipy.fftpack import dctn, idctn
import tqdm
import matplotlib.pyplot as plt
from screening import HelperClass

parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--num_iter', type=int, default=5000, help="Total number of iteration")
parser.add_argument('--plot_every', type=int, default=100, help="Frequency of saving plots")
parser.add_argument('--starting_epoch', type=int, default=0, help="Starting epoch number. If the value is nonzero than it will load the saved data.")
parser.add_argument('--saving', type=str, default='0')
# arguments for Gaussian mixture application
parser.add_argument('--lr', type=float, default=1e-3, help="Step size of the gradient ascent")
parser.add_argument('--tau', type=float, default=1.0, help="parameter for the c-transform")

args = parser.parse_args()
print(args)

image_folder = f'image-{args.saving}'
os.makedirs(image_folder, exist_ok=True)
desc = f'using tau={args.tau:0.2e}'


# %%


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
    workspace = dct2(workspace) / (kernel)
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
def compute_dx_forward(phi, dy):
  A = np.zeros_like(phi)
  A[:,:-1] = (phi[:,1:] - phi[:,:-1])/(1.0*dy)
  A[:,-1]  = (phi[:,-1] - phi[:,-2])/(1.0*dy)
  return A

def compute_dx_back(phi, dy):
  A = np.zeros_like(phi)
  A[:,1:] = (phi[:,1:] - phi[:,:-1])/(1.0*dy)
  A[:,0]  = (phi[:,1]  - phi[:,0])/(1.0*dy)
  return A

def compute_dxx(phi, dy):
  A = compute_dx_forward(phi, dy)
  return compute_dx_back(A, dy)

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

def compute_dxx(A, phi, dy):
  A[:,1:-1] = (phi[:,2:] - 2 * phi[:,1:-1] + phi[:,:-2])/(dy*dy)

def compute_dyy(A, phi, dy):
  A[1:-1,:] = (phi[2:,:] - 2 * phi[1:-1,:] + phi[:-2,:])/(dy*dy)

# centered difference
def compute_dxy(A, phi, dy):
  A[:-1,:-1] = (phi[1:,1:] - phi[1:,:-1] - phi[:-1,1:] + phi[:-1,:-1])/(dy*dy)
  A[-1,:-1]  = - (phi[-1,:-1] - phi[-1,1:] - phi[-2,:-1] + phi[-2,1:])  /(dy*dy)
  A[:-1,-1]  = - (phi[:-1,-1] - phi[:-1,-2] - phi[1:,-1] + phi[1:,-2])/(dy*dy)
  A[-1,-1]   = (phi[-1,-1] - phi[-1,-2] - phi[-2,-1] + phi[-2,-2])/(dy*dy)

# %%
from screening import c_transform_cpp, c_transform_forward_cpp,pushforward_entropic_cpp

# performing c transform
# output: modified psi
def c_transform(psi, phi, cost):
  psi[:] = np.min(cost + phi.reshape((1,-1)), axis=1)

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

def compute_rhs(phi_np, psi_np, nu_np, helper, dx, dy, n, m):

  #   initialize fx, fy
  grad1_phi = np.zeros((m,m))
  grad2_phi = np.zeros((m,m))
  phi_b = phi_np

  # dx: spatial resolution in X space
  # dy: spatial resolution in Y space

  compute_dx(grad1_phi, phi_b.reshape((m,m)), dy)
  compute_dy(grad2_phi, phi_b.reshape((m,m)), dy)

  vxx_psi = np.zeros((n,n))
  vyy_psi = np.zeros((n,n))
  vxy_psi = np.zeros((n,n))

  compute_dxx(vxx_psi, psi_np.reshape((n,n)), dx)
  compute_dyy(vyy_psi, psi_np.reshape((n,n)), dx)
  compute_dxy(vxy_psi, psi_np.reshape((n,n)), dx)

  R1  = np.zeros((m,m)).astype('float64')
  R2  = np.zeros((m,m)).astype('float64')
  R1y1 = np.zeros((m,m)).astype('float64')
  R2y2 = np.zeros((m,m)).astype('float64')

  helper.compute_inverse_g(R1, R2, phi_np, psi_np, grad1_phi, grad2_phi, vxx_psi, vyy_psi, vxy_psi, dx, dy, n, m, tau)

  R1 *= nu_np.reshape((m,m))
  R2 *= nu_np.reshape((m,m))

  compute_dx(R1y1, R1, dy)
  compute_dy(R2y2, R2, dy)

  gx0 = R1[:,0]
  gx1 = R1[:,-1]

  gy0 = R2[0,:]
  gy1 = R2[-1,:]

  return nu_np.reshape((m,m)) + R1y1 + R2y2, [-gx0,-gx1,-gy0,-gy1]


def solve_main_poisson(u, phi_np, psi_np, nu_np, kernel, helper, dx, dy, n, m):
  rhs, bdry = compute_rhs(phi_np, psi_np, nu_np, helper, dx, dy, n, m) # computing the right hand side
  solve_poisson_bdry(u,rhs,bdry,kernel)
  return rhs

# %%
# parameters
# grid size n x n
n = 128
m = 128

# step size for the gradient ascent
sigma = args.lr
tau = args.tau
max_iteration = 1000000

# epsilon for pushforward
eps = 1e-2
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

print(f'dx: {dx}, dy: {dy}')

kernel = initialize_kernel(m, m, dy)

Xv = np.zeros((n*n,2))
Xv[:,0] = Xx.reshape((n*n,))
Xv[:,1] = Xy.reshape((n*n,))

Yv = np.zeros((m*m,2))
Yv[:,0] = Yx.reshape((m*m,))
Yv[:,1] = Yy.reshape((m*m,))

cost = None
# cost = ((Xv.reshape((n*n,1,2)) - Yv.reshape((1,m*m,2)))**2).sum(2) * 1.0/(2.0 * tau)


psi_np = np.zeros((n*n)).astype('float64')

psi_np = 0.5 * (Xx**2 + Xy**2) - 0.9 * (Xx**2 + Xy**2) ** 0.5
psi_np = psi_np.flatten().astype('float64')

# psi_np.reshape((n,n))[(Xx**2 + Xy**2 > 1)] = 0

phi_np = np.zeros((m*m)).astype('float64')
c_transform_forward_cpp(phi_np, psi_np, dx, dy, n, m)
plt.imshow(phi_np.reshape((m,m)),origin='lower')
plt.savefig("sanity_check_c_transform_cpp.png")
plt.close('all')

nu_np  = np.zeros((m*m)).astype('float64')
mu_np  = np.zeros((n,n)).astype('float64')
mu_radius = 1.0
mu_np[(Xx**2 + Xy**2 < mu_radius**2)] = 1
mu_np /= mu_np.sum() * dx * dx
mu_np  = mu_np.flatten()

helper = HelperClass(dx, dy, n, m, xMin, yMin)

# helper.pushforward(nu_np, psi_np, mu_np)
pushforward_entropic_cpp(nu_np, mu_np, psi_np, phi_np, eps, dx, dy, -1.2, -1.2, n, m); nu_np /= nu_np.sum()*(dy*dy)
# plan = np.exp( (psi_np.reshape((-1,1)) - phi_np.reshape((1,-1)) - cost)/eps )
# plan /= plan.sum(axis=1).reshape((n*n,1))* dy * dy
# plan = plan * mu_np.reshape((n*n,1))
# nu_np[:] = (plan).sum(axis=0) * dx*dx
fig, ax = plt.subplots(1,1)
ax.contourf(Yx,Yy,nu_np.reshape((m,m)), 60)
ax.set_title(f"sum: {np.sum(nu_np) * dy * dy}")
plt.savefig("sanity_check_push_forward_cpp.png")
plt.close('all')

# %%

u = np.zeros((m,m)).astype('float64')

phi_np = phi_np.reshape((m,m))
phi_np[(Yx**2+Yy**2)>mu_radius**2] = 0
phi_np = phi_np.flatten()

# %%

# if starting epoch > 0 then load the saved data.
if args.starting_epoch > 0:
    phi_np = np.load(f'{image_folder}/phi.npy')

J_list = []

Tx = np.zeros((n,n))
Ty = np.zeros((n,n))
rhs = np.zeros((m,m))

pbar = tqdm.tqdm(range(args.starting_epoch, args.num_iter))

def compute_c_transform(psi_np, phi_np, dx, dy, n, m, xMin=-1.2, yMin=-1.2, method='nonentropic', cost=None):
  if method == 'nonentropic':
    c_transform_cpp(psi_np, phi_np, dx, dy, n, m, xMin, yMin)
  else:
    compute_ctransform_eps(psi_np, phi_np, cost, eps, dy) # entropic

def compute_pushforward(nu_np, mu_np, psi_np, phi_np, dx, dy, n, m, cost=None, eps=1e-2, method='nonentropic'):
  if method == 'nonentropic':
    helper.pushforward(nu_np, psi_np, mu_np)
  else:
    pushforward_entropic_cpp(nu_np, mu_np, psi_np, phi_np, eps, dx, dy, xMin, yMin, n, m); nu_np /= nu_np.sum()*(dy*dy)
                          
    # plan = np.exp( (psi_np.reshape((-1,1)) - phi_np.reshape((1,-1)) - cost)/eps )
    # plan /= plan.sum(axis=1).reshape((n*n,1))* dy * dy
    # plan = plan * mu_np.reshape((n*n,1))
    # nu_np[:] = (plan).sum(axis=0) * dx*dx

# phi_np*=0

phi_np = np.load( f"{image_folder}/phi.npy")

for it in pbar:
  compute_c_transform(psi_np, phi_np, dx, dy, n, m, method='nonentropic')
  compute_pushforward(nu_np, mu_np, psi_np, phi_np, dx, dy, n, m, cost=cost, eps=eps, method='entropic')

  nu_np.reshape((m,m))[(Yx**2+Yy**2)>mu_radius**2] = 0

  rhs = solve_main_poisson(u, phi_np, psi_np, nu_np, kernel, helper, dx, dy, n, m)
  error = np.mean(u**2)
  phi_np += sigma * u.flatten()
  phi_np = phi_np.reshape((m,m))
  phi_np[(Yx**2+Yy**2)>mu_radius**2] = 0
  phi_np = phi_np.flatten()

  # find the value of J
  J_val = (phi_np * nu_np).sum() * dy*dy
  J_list.append(J_val)

  compute_dx(Tx, psi_np.reshape((n,n)), dx)
  compute_dy(Ty, psi_np.reshape((n,n)), dx)

  Tx = Xx - args.tau * Tx; Ty = Xy - args.tau * Ty

  if it % args.plot_every == 0:
    np.save( f"{image_folder}/phi.npy", phi_np)
    fig,ax = plt.subplots(1,6,figsize=(18,4),constrained_layout=True)
    ax[0].contourf(Yx,Yy,np.log(1+np.log(1+nu_np)).reshape((m,m)),60)
    ax[0].set_title("nu")
    ax[0].set_aspect('equal')
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
    ax[5].set_xlim([-1.2,1.2])
    ax[5].set_ylim([-1.2,1.2])
    plt.suptitle(f"it={it} error={error:0.2e}\nsigma: {sigma:0.2e}\n{desc}")
    filename = f"{image_folder}/{it//args.plot_every}.png"
    plt.savefig(filename)
    plt.savefig(f"{image_folder}/000-status.png")
    plt.close('all')
  pbar.set_description(f'{filename} error={error:0.2e}')
