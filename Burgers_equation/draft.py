import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 1.0  
T = 2.0  
Nx = 50  
Ny = 50  
Nt = 200  
nu = 0.01  # 粘性系数

dx = L / (Nx - 1)  
dy = L / (Ny - 1)  
dt = T / Nt  

# 初始化网格
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
u_new = np.zeros((Nx, Ny))
v_new = np.zeros((Nx, Ny))

# 设置初始条件
for i in range(Nx):
    for j in range(Ny):
        u[i, j] = -np.sin(np.pi * x[i]) * np.cos(np.pi * y[j])
        v[i, j] = np.cos(np.pi * x[i]) * np.sin(np.pi * y[j])

# 边界条件
def apply_boundary_conditions(u, v):
    # 周期性边界条件
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

    v[0, :] = v[-2, :]
    v[-1, :] = v[1, :]
    v[:, 0] = v[:, -2]
    v[:, -1] = v[:, 1]
    
    return u, v

# 保存每个时间步的数据
time_steps_data = []

# 时间步循环
for n in range(Nt):
    u_new = u.copy()
    v_new = v.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            uxx = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            uyy = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            vxx = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2
            vyy = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2
            
            # 中心差分计算
            u_new[i, j] = u[i, j] + dt * (nu * (uxx + uyy) - u[i, j] * (u[i, j] - u[i-1, j]) / dx - v[i, j] * (u[i, j] - u[i, j-1]) / dy)
            v_new[i, j] = v[i, j] + dt * (nu * (vxx + vyy) - u[i, j] * (v[i, j] - v[i-1, j]) / dx - v[i, j] * (v[i, j] - v[i, j-1]) / dy)
    
    # 更新并应用边界条件
    u_new, v_new = apply_boundary_conditions(u_new, v_new)
    u = u_new.copy()
    v = v_new.copy()
    
    # 检查数值稳定性
    if np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isinf(u)) or np.any(np.isinf(v)):
        print(f"Numerical instability detected at time step {n}.")
        break
    
    # 保存当前时间步的数据
    time_steps_data.append((u.copy(), v.copy()))

def plot_figure(X, Y, Z, title, time):
    plt.tricontour(X, Y, Z, 15, cmap='jet')
    plt.tricontourf(X, Y, Z, 15, cmap='jet')
    plt.colorbar()
    plt.plot(X, Y, 'ko', ms=2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(f"{title} at time step {time}")
    plt.show()
    plt.close()

# 绘制某一时间步的解
def plot_timestep(timestep):
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    if timestep < 0 or timestep >= len(time_steps_data):
        print("time step out of range!")
        return
    
    u, v = time_steps_data[timestep]
    plot_figure(X, Y, u.flatten(), "U-velocity", timestep)
    plot_figure(X, Y, v.flatten(), "V-velocity", timestep)

plot_timestep(0)
plot_timestep(50)
plot_timestep(100)
plot_timestep(150)
plot_timestep(199)
