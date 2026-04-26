import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
Nx = 100  # Number of grid points in x
Ny = 100  # Number of grid points in y
Lx = 2 * np.pi  # Length of domain in x
Ly = 2 * np.pi  # Length of domain in y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
nu = 0.01  # Viscosity
dt = 0.001  # Time step
T = 1.0  # Total time
Nt = int(T / dt)  # Number of time steps

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial conditions
u = np.sin(X) * np.cos(Y)
v = -np.cos(X) * np.sin(Y)

# Function to compute derivatives
def compute_derivatives(u, v, dx, dy):
    # First derivatives
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    vx = np.zeros_like(v)
    vy = np.zeros_like(v)
    
    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    uy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    vx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    vy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    
    # Second derivatives
    uxx = np.zeros_like(u)
    uyy = np.zeros_like(u)
    vxx = np.zeros_like(v)
    vyy = np.zeros_like(v)
    
    uxx[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
    uyy[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / (dy**2)
    vxx[:, 1:-1] = (v[:, 2:] - 2*v[:, 1:-1] + v[:, :-2]) / (dx**2)
    vyy[1:-1, :] = (v[2:, :] - 2*v[1:-1, :] + v[:-2, :]) / (dy**2)
    
    return ux, uy, uxx, uyy, vx, vy, vxx, vyy

# Time integration
for n in range(Nt):
    ux, uy, uxx, uyy, vx, vy, vxx, vyy = compute_derivatives(u, v, dx, dy)
    
    # Update u and v using explicit Euler
    u_new = u - dt * (u * ux + v * uy) + nu * dt * (uxx + uyy)
    v_new = v - dt * (u * vx + v * vy) + nu * dt * (vxx + vyy)
    
    # Apply periodic boundary conditions
    u_new[:, 0] = u_new[:, -2]
    u_new[:, -1] = u_new[:, 1]
    u_new[0, :] = u_new[-2, :]
    u_new[-1, :] = u_new[1, :]
    
    v_new[:, 0] = v_new[:, -2]
    v_new[:, -1] = v_new[:, 1]
    v_new[0, :] = v_new[-2, :]
    v_new[-1, :] = v_new[1, :]
    
    u = u_new
    v = v_new

# Plot the final solution
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u, cmap='viridis')
ax1.set_title('u velocity')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, v, cmap='plasma')
ax2.set_title('v velocity')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('v')

plt.tight_layout()
plt.show()

print("Simulation completed successfully")