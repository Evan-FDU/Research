import numpy as np
import matplotlib.pyplot as plt

# Parameters
Nx = 200  # Number of grid points in x
Ny = 100  # Number of grid points in y
Lx = 10.0  # Length of domain in x
Ly = 5.0  # Length of domain in y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
nu = 0.01  # Viscosity
dt = 0.01  # Time step
T = 1.0  # Total time
Nt = int(T / dt)  # Number of time steps
Re = 100  # Reynolds number (for reference)

# Cylinder parameters
cx, cy = Lx / 4, Ly / 2  # Center of cylinder
radius = 0.5  # Radius of cylinder

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial conditions
u = np.ones((Ny, Nx)) * 1.0  # Uniform inflow
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))

# Function to check if point is inside cylinder
def inside_cylinder(i, j):
    return (x[j] - cx)**2 + (y[i] - cy)**2 < radius**2

# Apply no-slip boundary condition on cylinder
for i in range(Ny):
    for j in range(Nx):
        if inside_cylinder(i, j):
            u[i, j] = 0.0
            v[i, j] = 0.0

# Function to compute derivatives
def compute_derivatives(u, v, dx, dy):
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    vx = np.zeros_like(v)
    vy = np.zeros_like(v)
    uxx = np.zeros_like(u)
    uyy = np.zeros_like(u)
    vxx = np.zeros_like(v)
    vyy = np.zeros_like(v)
    
    # Central differences for interior points
    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    uy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    vx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    vy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    
    uxx[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
    uyy[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / (dy**2)
    vxx[:, 1:-1] = (v[:, 2:] - 2*v[:, 1:-1] + v[:, :-2]) / (dx**2)
    vyy[1:-1, :] = (v[2:, :] - 2*v[1:-1, :] + v[:-2, :]) / (dy**2)
    
    return ux, uy, uxx, uyy, vx, vy, vxx, vyy

# Function to solve Poisson equation for pressure (simple Jacobi iteration)
def solve_poisson(div, dx, dy, tol=1e-6, max_iter=1000):
    p = np.zeros_like(div)
    for _ in range(max_iter):
        p_old = p.copy()
        p[1:-1, 1:-1] = ((p[1:-1, 2:] + p[1:-1, :-2]) * dy**2 +
                          (p[2:, 1:-1] + p[:-2, 1:-1]) * dx**2 -
                          div[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
        # Boundary conditions for p
        p[:, 0] = p[:, 1]  # Left
        p[:, -1] = 0  # Right (outflow)
        p[0, :] = p[1, :]  # Bottom
        p[-1, :] = p[-2, :]  # Top
        
        if np.max(np.abs(p - p_old)) < tol:
            break
    return p

# Time integration using fractional step method
for n in range(Nt):
    if n % 10 == 0:
        print(f"Time step {n}/{Nt}")
    # Step 1: Compute intermediate velocities
    # Step 1: Compute intermediate velocities
    ux, uy, uxx, uyy, vx, vy, vxx, vyy = compute_derivatives(u, v, dx, dy)
    
    u_star = u + dt * (-u * ux - v * uy + nu * (uxx + uyy))
    v_star = v + dt * (-u * vx - v * vy + nu * (vxx + vyy))
    
    # Apply boundary conditions to u_star, v_star
    for i in range(Ny):
        for j in range(Nx):
            if inside_cylinder(i, j):
                u_star[i, j] = 0.0
                v_star[i, j] = 0.0
    
    # Inflow boundary
    u_star[:, 0] = 1.0
    v_star[:, 0] = 0.0
    
    # Outflow boundary (Neumann)
    u_star[:, -1] = u_star[:, -2]
    v_star[:, -1] = v_star[:, -2]
    
    # Top and bottom (free slip or no slip, here free slip)
    u_star[0, :] = u_star[1, :]
    u_star[-1, :] = u_star[-2, :]
    v_star[0, :] = 0.0
    v_star[-1, :] = 0.0
    
    # Step 2: Compute divergence
    div = (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2*dx) + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2*dy)
    div = np.pad(div, ((1,1),(1,1)), mode='constant')
    
    # Step 3: Solve for pressure
    p = solve_poisson(div, dx, dy)
    
    # Step 4: Project velocities
    px = (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx)
    py = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dy)
    px = np.pad(px, ((1,1),(1,1)), mode='constant')
    py = np.pad(py, ((1,1),(1,1)), mode='constant')
    
    u = u_star - dt * px
    v = v_star - dt * py
    
    # Reapply boundary conditions
    for i in range(Ny):
        for j in range(Nx):
            if inside_cylinder(i, j):
                u[i, j] = 0.0
                v[i, j] = 0.0
    u[:, 0] = 1.0
    v[:, 0] = 0.0

# Plot the final solution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, np.sqrt(u**2 + v**2), cmap='viridis')
plt.colorbar()
plt.title('Velocity Magnitude')
plt.xlabel('x')
plt.ylabel('y')
circle = plt.Circle((cx, cy), radius, color='red', fill=False)
plt.gca().add_artist(circle)

plt.subplot(1, 2, 2)
plt.streamplot(X, Y, u, v, density=1.5)
plt.title('Streamlines')
circle = plt.Circle((cx, cy), radius, color='red', fill=False)
plt.gca().add_artist(circle)

plt.tight_layout()
plt.savefig('cylinder_flow.png')
# plt.show()

print("DNS simulation of 2D cylinder flow completed successfully")