from math import pi
import matplotlib.pylab as plt
import matplotlib
import numpy as np
import timeit
from numpy import genfromtxt
import os
import scipy.io as scio

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import torch.optim as optim

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(0)

class MLP(nn.Module):

    def __init__(self, in_size, h_sizes, out_size):

        super(MLP, self).__init__()

        # Hidden layers
        self.input = nn.Linear(in_size, h_sizes[0])

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):

        x = torch.tanh(self.input(x))

        # Feedforward
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        output = self.out(x)

        return output

model = MLP(3, [30, 20, 30, 20], 2).to(device)
# model_v = MLP(3, [30, 20, 30, 20], 1).to(device)

# xavier initialization of weights
for param_tensor in model.state_dict():
    if model.state_dict()[param_tensor].dim() > 1:
        nn.init.xavier_normal_(model.state_dict()[param_tensor])


Mymodel= 'PCNN' 
start = timeit.default_timer()
# Parameters
learning_rate = 5e-4
error_threshold = 1e-3
num_epochs = int(45000)
display_step = int(20)

num_tsample = 41
num_xsample = 41#65
num_ysample = 41#65
total_time = 2.0

# input
data_dir = '/home/gli12/project/ASME2024/Data/burgers_ic_2001x2x64x64.mat'
my_data = scio.loadmat(data_dir)

pixel_data = np.array([(x, y, pixel_value) for (y, row) in enumerate(my_data['uv'][0][0]) for (x, pixel_value) in enumerate(row)])
X = pixel_data[:,0]/63.0
Y = pixel_data[:,1]/63.0
Z = pixel_data[:,2]

# for data loss
results_u = np.array([(t, x, y, pixel_value) 
            for t in range(my_data['uv'][:,0,...][::20].shape[0]) 
            for (y, row) in enumerate(my_data['uv'][:,0,...][::20][t]) 
            for (x, pixel_value) in enumerate(row)])
results_v = np.array([(t, x, y, pixel_value) 
            for t in range(my_data['uv'][:,1,...][::20].shape[0]) 
            for (y, row) in enumerate(my_data['uv'][:,0,...][::20][t]) 
            for (x, pixel_value) in enumerate(row)])

results_u = results_u[::2] # downsample for x, y
results_v = results_v[::2] # downsample for x, y

T = results_u[:,0]/50.0
X = results_u[:,1]/63.0
Y = results_u[:,2]/63.0
ZU = results_u[:,3]
ZV = results_v[:,3]

# Convert numpy arrays to torch tensors and move to device
def to_tensor(data, requires_grad=False):
    return torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad).to(device)

T = to_tensor(np.reshape(T, (T.size, 1)), True)
X = to_tensor(np.reshape(X, (X.size, 1)), True)
Y = to_tensor(np.reshape(Y, (Y.size, 1)), True)
ZU = to_tensor(np.reshape(ZU, (ZU.size, 1)))
ZV = to_tensor(np.reshape(ZV, (ZV.size, 1)))
# Uniform sample function
def uniform_sample(t_min, t_max, x_min, x_max, y_min, y_max, n_t, n_x, n_y):
    t = np.linspace(t_min, t_max, n_t)
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    t, x, y = np.meshgrid(t, x, y)
    tx = t.reshape(-1, 1)
    xy = np.hstack((tx, x.reshape(-1, 1), y.reshape(-1, 1)))

    mask1 = (xy[:, 1] - x_min) == 0
    mask2 = (x_max - xy[:, 1]) == 0
    mask3 = (xy[:, 2] - y_min) == 0
    mask4 = (y_max - xy[:, 2]) == 0
    mask5 = (xy[:, 0] - t_min) == 0
    mask6 = mask1 + mask2 + mask3 + mask4 + mask5
    s_xlb = xy[mask1]
    s_xub = xy[mask2]
    s_ylb = xy[mask3]
    s_yub = xy[mask4]
    s_0 = xy[mask5]
    s_i = xy[np.logical_not(mask6)]

    return s_i, s_xlb, s_xub, s_ylb, s_yub, s_0

# Use the uniform_sample function to generate points
s_i, s_xlb, s_xub, s_ylb, s_yub, s_0 = uniform_sample(0.0, total_time, 0.0, 1.0, 0.0, 1.0, num_tsample, num_xsample, num_ysample)

# Convert numpy arrays to torch tensors and expand dimensions
T_i = to_tensor(s_i[:, 0]).unsqueeze(1).requires_grad_(True)
X_i = to_tensor(s_i[:, 1]).unsqueeze(1).requires_grad_(True)
Y_i = to_tensor(s_i[:, 2]).unsqueeze(1).requires_grad_(True)

T_xlb = to_tensor(s_xlb[:, 0]).unsqueeze(1).requires_grad_(True)
X_xlb = to_tensor(s_xlb[:, 1]).unsqueeze(1).requires_grad_(True)
Y_xlb = to_tensor(s_xlb[:, 2]).unsqueeze(1).requires_grad_(True)

T_xub = to_tensor(s_xub[:, 0]).unsqueeze(1).requires_grad_(True)
X_xub = to_tensor(s_xub[:, 1]).unsqueeze(1).requires_grad_(True)
Y_xub = to_tensor(s_xub[:, 2]).unsqueeze(1).requires_grad_(True)

T_ylb = to_tensor(s_ylb[:, 0]).unsqueeze(1).requires_grad_(True)
X_ylb = to_tensor(s_ylb[:, 1]).unsqueeze(1).requires_grad_(True)
Y_ylb = to_tensor(s_ylb[:, 2]).unsqueeze(1).requires_grad_(True)

T_yub = to_tensor(s_yub[:, 0]).unsqueeze(1).requires_grad_(True)
X_yub = to_tensor(s_yub[:, 1]).unsqueeze(1).requires_grad_(True)
Y_yub = to_tensor(s_yub[:, 2]).unsqueeze(1).requires_grad_(True)

T_0 = to_tensor(s_0[:, 0]).unsqueeze(1).requires_grad_(True)
X_0 = to_tensor(s_0[:, 1]).unsqueeze(1).requires_grad_(True)
Y_0 = to_tensor(s_0[:, 2]).unsqueeze(1).requires_grad_(True)

# Construct model
def uv(t, x, y):
    u_v = model(torch.cat((t,x,y),1))
    return u_v[:,0].unsqueeze(dim=1), u_v[:,1].unsqueeze(dim=1)

def ux(t, x, y):
    gu, gv = uv(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False).to(device)
    u_x = grad(gu, x, create_graph=True, grad_outputs=grad_outputs)[0]
    v_x = grad(gv, x, create_graph=True, grad_outputs=grad_outputs)[0]
    return u_x, v_x

def uy(t, x, y):
    gu, gv = uv(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False).to(device)
    u_y = grad(gu, y, create_graph=True, grad_outputs=grad_outputs)[0]
    v_y = grad(gv, y, create_graph=True, grad_outputs=grad_outputs)[0]
    return u_y, v_y

def f_PDE(t, x, y):
    gu, gv = uv(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False).to(device)

    u_t = grad(gu, t, create_graph=True, grad_outputs=grad_outputs)[0]
    u_x = grad(gu, x, create_graph=True, grad_outputs=grad_outputs)[0]
    u_xx = grad(u_x, x, create_graph=True, grad_outputs=grad_outputs)[0]
    u_y = grad(gu, y, create_graph=True, grad_outputs=grad_outputs)[0]
    u_yy = grad(u_y, y, create_graph=True, grad_outputs=grad_outputs)[0]

    v_t = grad(gv, t, create_graph=True, grad_outputs=grad_outputs)[0]
    v_x = grad(gv, x, create_graph=True, grad_outputs=grad_outputs)[0]
    v_xx = grad(v_x, x, create_graph=True, grad_outputs=grad_outputs)[0]
    v_y = grad(gv, y, create_graph=True, grad_outputs=grad_outputs)[0]
    v_yy = grad(v_y, y, create_graph=True, grad_outputs=grad_outputs)[0]

    f1 = u_t - (1/200)*(u_xx+u_yy) + gu*u_x + gv*u_y
    f2 = v_t - (1/200)*(v_xx+v_yy) + gu*v_x + gv*v_y
    return f1+f2

# Define loss and optimizer
MSE = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

error = 1.0
a = np.zeros((1,10))
b = np.zeros(1)

def train():
    # Train the model
    for epoch in range(num_epochs):
        # data loss
        cost1 = MSE(uv(T, X, Y)[0], ZU) + MSE(uv(T, X, Y)[1], ZV)
        # differential operator
        cost2 = (f_PDE(T_i, X_i, Y_i) ** 2).mean()
        # intial condition
        uu, vv = uv(T_0, X_0, Y_0)
        cost3 = ((uu - 0.5 * (torch.sin(4.0 * pi * X_0) + torch.sin(4.0 * pi * Y_0))) ** 2).mean() + \
                ((vv - 0.5 * (torch.cos(4.0 * pi * X_0) + torch.cos(4.0 * pi * Y_0))) ** 2).mean()
        # boundary condition - Periodic BC
        uxl, vxl = uv(T_xlb, X_xlb, Y_xlb)  # (tbx, xl, ybx)
        uxr, vxr = uv(T_xub, X_xub, Y_xub)  # (tbx, xr, ybx)
        uyl, vyl = uv(T_ylb, X_ylb, Y_ylb)  # (tby, xby, yl)
        uyr, vyr = uv(T_xub, X_xub, Y_xub)  # (tby, xby, yr)

        uxl_x, vxl_x = ux(T_xlb, X_xlb, Y_xlb)  # (tbx, xl, ybx)
        uxr_x, vxr_x = ux(T_xub, X_xub, Y_xub)  # (tbx, xr, ybx)
        uyl_y, vyl_y = uy(T_ylb, X_ylb, Y_ylb)  # (tby, xby, yl)
        uyr_y, vyr_y = uy(T_xub, X_xub, Y_xub)  # (tby, xby, yr)
        cost4 = (((uxl - uxr) ** 2).mean() + ((uyl - uyr) ** 2).mean() + ((vxl - vxr) ** 2).mean()
                 + ((vyl - vyr) ** 2).mean() + ((uxl_x - uxr_x) ** 2).mean() + ((uyl_y - uyr_y) ** 2).mean()
                 + ((vxl_x - vxr_x) ** 2).mean() + ((vyl_y - vyr_y) ** 2).mean())

        cost = (cost2 ** 2 + cost3 ** 2 + cost4 ** 2) / (cost2 + cost3 + cost4)
        ncost = 1.0 / 3.0 * (cost2 + cost3 + cost4)
        error = ncost.item()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % display_step == 0:
            print(
                'Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
                    epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))

        if error < error_threshold:
            print(
                'Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
                    epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))

    stop = timeit.default_timer()
    print("Running time")
    runnint_time = stop - start
    print(runnint_time)
    np.savetxt('./Burgers_equation/Adam_Training_hist.csv', a, delimiter=",")
    np.savetxt('./Burgers_equation/Adam_Forces.csv', b, delimiter=",")
    f = open('Training_time_GDA.txt','w')
    f.write(str(runnint_time))
    f.close()
    # Save the model checkpoint
    torch.save(model.state_dict(), './Burgers_equation/PCNN_burgers.pkl')
    print("Running time")
    return


def get_data_time(t):

    pixel_data_u = np.array([(x, y, pixel_value) for (y, row) in enumerate(my_data['uv'][t][0]) for (x, pixel_value) in enumerate(row)])
    pixel_data_v = np.array([(x, y, pixel_value) for (y, row) in enumerate(my_data['uv'][t][1]) for (x, pixel_value) in enumerate(row)])
    ZU = pixel_data_u[:,2]
    ZV = pixel_data_v[:,2]

    return ZU, ZV

def evaluation():
    # Evaluation
    time= 4.0
    L_xsample = 64

    L_ysample = 64
    pixel_data = np.array([(x, y, pixel_value) for (y, row) in enumerate(my_data['uv'][0][-1]) for (x, pixel_value) in enumerate(row)])
    X = pixel_data[:,0]/63.0
    Y = pixel_data[:,1]/63.0

    X= torch.Tensor(np.reshape(X,(X.size,1))).to(device)
    Y= torch.Tensor(np.reshape(Y,(Y.size,1))).to(device)
    T1 = torch.ones(L_xsample * L_ysample, 1)*time

    model.load_state_dict(torch.load('/home/gli12/project/ASME2024/Burgers_equation/PCNN_burgers.pkl'))

    ZU_truth, ZV_truth = get_data_time(int(time/0.002))

    T1 = (torch.ones(L_xsample * L_ysample, 1)*time).to(device)
    ZU, ZV = model(torch.cat((T1, X, Y),1)).cpu().detach().numpy()[:,0], model(torch.cat((T1, X, Y),1)).cpu().detach().numpy()[:,1]
    #ZV = model(torch.cat((T1, X, Y),1)).cpu().detach().numpy()
    print('rmse at t=4.0: ', np.sqrt(np.mean((ZU - ZU_truth) ** 2)), np.sqrt(np.mean((ZV - ZV_truth) ** 2)))

    X=np.squeeze(X.cpu().detach().numpy())
    Y=np.squeeze(Y.cpu().detach().numpy())

    # plot u truth
    matplotlib.rcParams.update({'font.size': 20})
    plt.tricontour(X, Y, ZU.squeeze(), 15, cmap='jet')
    plt.tricontourf(X, Y, ZU.squeeze(), 15, cmap='jet')
    cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    #cbar.set_label('Burgers U')
    plt.plot(2*X, 2*Y, 'ko', ms=3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title(Mymodel + '-Burgers U at t=' + str(time))
    plt.savefig('./Burgers_equation/' + Mymodel + ' Burgers U' + str(time) + '.png')
    plt.show()
    plt.close()

    plt.tricontour(X, Y, ZV.squeeze(), 15, cmap='jet')
    plt.tricontourf(X, Y, ZV.squeeze(), 15, cmap='jet')
    cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    #cbar.set_label('Burgers U')
    plt.plot(2*X, 2*Y, 'ko', ms=3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title(Mymodel + '-Burgers V at t=' + str(time))
    plt.savefig('./Burgers_equation/' + Mymodel + ' Burgers V' + str(time) + '.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    train()
    evaluation()
    print('done')