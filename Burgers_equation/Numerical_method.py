"This code uses the finite difference method to solve the burgers in 2D."
"The boundary condition is periodic boundary condition."

import numpy as np
import matplotlib.pyplot as plt


L = 1.0  
T = 2.0  
Nx = 50  # 
Ny = 50  # 
Nt = 200 
nu = 0.005 

dx = L / (Nx - 1)  
dy = L / (Ny - 1)  
dt = T / Nt  

# initialize the grid
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
u_new = np.zeros((Nx, Ny))
v_new = np.zeros((Nx, Ny))

if dt > 0.25 * min(dx, dy)**2 / nu:
    raise ValueError("Time step is too large for stability.")

# set the initial condition
for i in range(Nx):
    for j in range(Ny):
        u[i, j] = 0.5 * (np.sin(4 * np.pi * x[i]) + np.sin(4 * np.pi * y[j]))
        v[i, j] = 0.5 * (np.cos(4 * np.pi * x[i]) + np.cos(4 * np.pi * y[j]))

# boundary conditions
def apply_boundary_conditions(u, v):
    # fixed BCs
    #u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
    #v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0
    # periodic BCs
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

    v[0, :] = v[-2, :]
    v[-1, :] = v[1, :]
    v[:, 0] = v[:, -2]
    v[:, -1] = v[:, 1]

    return u, v

# save the data for each time step
time_steps_data = []

# time stepping loop
for n in range(Nt):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            uxx = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            uyy = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            vxx = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2
            vyy = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2

            ux = (u[i+1, j] - u[i-1, j]) / (2 * dx)
            uy = (u[i, j+1] - u[i, j-1]) / (2 * dy)
            vx = (v[i+1, j] - v[i-1, j]) / (2 * dx)
            vy = (v[i, j+1] - v[i, j-1]) / (2 * dy)
            
            u_new[i, j] = u[i, j] + dt * (nu * (uxx + uyy) - u[i, j] * ux - v[i, j] * uy)
            v_new[i, j] = v[i, j] + dt * (nu * (vxx + vyy) - u[i, j] * vx - v[i, j] * vy)
    

            #u_new[i, j] = u[i, j] + dt * (nu * (uxx + uyy) - u[i, j] * (u[i, j] - u[i-1, j]) / dx - v[i, j] * (u[i, j] - u[i, j-1]) / dy)
            #v_new[i, j] = v[i, j] + dt * (nu * (vxx + vyy) - u[i, j] * (v[i, j] - v[i-1, j]) / dx - v[i, j] * (v[i, j] - v[i, j-1]) / dy)
    #print('time step value:', n, ' u:', np.mean(u_new), ' v:', np.mean(v_new))
    u_new, v_new = apply_boundary_conditions(u_new, v_new)
    u = u_new.copy()
    v = v_new.copy()

    if np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isinf(u)) or np.any(np.isinf(v)):
        print(f"Numerical instability detected at time step {n}.")
    
    time_steps_data.append((u.copy(), v.copy()))


def plotfigure(X, Y, Z, Mymodel, time, save_path):
    plt.tricontour(X, Y, Z, 15, cmap='jet')
    plt.tricontourf(X, Y, Z, 15, cmap='jet')
    cbar=plt.colorbar()
    plt.plot(X, Y, 'ko', ms=2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    #plt.savefig(save_path + Mymodel + '-Temperature' + str(time) + '.png')
    plt.show()
    plt.close()
    return 

# plot the solution at a specific time step
def plot_timestep(timestep):

    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()/(Nx)
    Y = Y.flatten()/(Nx)

    if timestep < 0 or timestep >= Nt:
        print("time step out of range!")
        return
    
    plotfigure(X, Y, time_steps_data[timestep][0].flatten(), "burgers_u", timestep, "./Burgers_equation/")
    #plotfigure(X, Y, time_steps_data[timestep][1].flatten(), "burgers_v", timestep, "./Burgers_equation/")
    # plt.figure(figsize=(8, 6))
    # plt.contourf(x, y, time_steps_data[timestep], 20, cmap='hot')
    # plt.colorbar()
    # plt.title(f"Solution of the PDE at timestep {timestep}")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

plot_timestep(0)
plot_timestep(50)
plot_timestep(100)
plot_timestep(150)
plot_timestep(199)