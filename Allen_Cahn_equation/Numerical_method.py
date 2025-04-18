"This code uses the finite difference method to solve the Allen Cahn equation in 2D."
"The boundary condition is zero flux boundary condition."

import numpy as np
import matplotlib.pyplot as plt


L = 1.0  
T = 2.0  
Nx = 50  # 
Ny = 50  # 
Nt = 200  
alpha = 0.001  

dx = L / (Nx - 1)  
dy = L / (Ny - 1)  
dt = T / Nt  

# initialize the grid
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
u = np.zeros((Nx, Ny))
u_new = np.zeros((Nx, Ny))

# set the initial condition
for i in range(Nx):
    for j in range(Ny):
        u[i, j] = 0.5 * (np.sin(4 * np.pi * x[i]) + np.sin(4 * np.pi * y[j]))

# boundary conditions
def apply_boundary_conditions(u):

    # u_x(t, 0, y) = -0.1  =>  (u[2, :] - 4*u[1, :] + 3*u[0, :]) / (2*dx) = -0.1
    u[0, :] = u[1, :] + 0 * dx
    #u[0, :] = (4*u[1,:]-u[2,:]-0.1*2*dx)/3

    # u_x(t, 1, y) = -0.1  =>  (3*u[-1, :] - 4*u[-2, :] + u[-3, :]) / (2*dx) = -0.1
    u[-1, :] = u[-2, :] + 0 * dx
    #u[-1, :] = (4*u[-2,:]-u[-3,:]-0.1*2*dx)/3

    # u_y(t, x, 0) = 0.1  =>  (u[:, 2] - 4*u[:, 1] + 3*u[:, 0]) / (2*dy) = 0.1
    u[:, 0] = u[:, 1] - 0 * dy
    #u[:, 0] = (4*u[:,1]-u[:,2]-0.1*2*dy)/3

    # u_y(t, x, 1) = 0.1  =>  (3*u[:, -1] - 4*u[:, -2] + u[:, -3]) / (2*dy) = 0.1
    u[:, -1] = u[:, -2] - 0 * dy
    #u[:, -1] = (4*u[:, -2]-u[:, -3]-0.1*2*dy)/3

    return u

# save the data for each time step
time_steps_data = []

# time stepping loop
for n in range(Nt):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            uxx = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            uyy = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            u_new[i, j] = u[i, j] + dt * (alpha * (uxx + uyy)+u[i, j]-u[i, j]**3)
    
    # update and apply the boundary conditions
    u_new = apply_boundary_conditions(u_new)
    u = u_new.copy()
    
    # save the data of current time step
    time_steps_data.append(u.copy())

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
    
    plotfigure(X, Y, time_steps_data[timestep].flatten(), "Heat_equation", timestep, "./Heat_equation/")
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