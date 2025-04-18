from math import pi
import matplotlib.pylab as plt
import matplotlib
import numpy as np
import timeit
from numpy import genfromtxt

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
import torch.optim as optim
from torch import Tensor
import sys
sys.path.append('/home/gli12/project/ASME2024')
from plot_figure import plotfigure
#from adamgda import AdamGDA


# Device configuration
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
#cuda = True if torch.cuda.is_available() else False

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


model = MLP(3, [30, 20, 30, 20], 1)#.cuda()

# model = MLP(3, [10,10,10,10], 1)

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# xavier initialization of weights
for param_tensor in model.state_dict():
    if model.state_dict()[param_tensor].dim() > 1:
        nn.init.xavier_normal_(model.state_dict()[param_tensor])

#weigths of different losses
w = torch.randn(4, 1, requires_grad=True)
nn.init.xavier_normal_(w)

Mymodel="PCNN_AllenCahn"
start = timeit.default_timer()
# Parameters
learning_rate = 5e-4
error_threshold = 1e-3
num_epochs = int(9e4+1)
display_step = int(100)

num_tsample = 21
num_xsample = 31
num_ysample = 31
total_time = 1.0

# input
my_data_train = genfromtxt('./Data/allen_cahn_flux0.1_100%.csv', delimiter=',', skip_header=1)
X_train = my_data_train[:, 0]
Y_train = my_data_train[:, 1]
Z_train = my_data_train[:, 3]
T_train = my_data_train[:, 2] 

# my_data = genfromtxt('./Allen Cahn equation/allen chan flux.csv', delimiter=',', skip_header=1)

# T = my_data[:,0]
# X = my_data[:,1]
# Y = my_data[:,2]
# Z = my_data[:,3]

# Convert numpy arrays to torch tensors
# Convert numpy arrays to torch tensors and move to device
def to_tensor(data, requires_grad=False):
    return torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad).to(device)

# T = to_tensor(np.reshape(T, (T.size, 1)), True)
# X = to_tensor(np.reshape(X, (X.size, 1)), True)
# Y = to_tensor(np.reshape(Y, (Y.size, 1)), True)
# Z = to_tensor(np.reshape(Z, (Z.size, 1)))

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

T_train = to_tensor(T_train).unsqueeze(1).requires_grad_(True)
X_train = to_tensor(X_train).unsqueeze(1).requires_grad_(True)
Y_train = to_tensor(Y_train).unsqueeze(1).requires_grad_(True)
Z_train = to_tensor(Z_train).unsqueeze(1).requires_grad_(True)

# Construct model
def u0(x,y):
    u0 = 0.5 * (torch.sin(4.0 * np.pi * x) + torch.sin(4.0 * np.pi * y))
    return u0

def u(t, x, y):
    uo = model(torch.cat((t,x,y),1))
    #if t.eq(0.0).all():
    #    uo[:,0] = u0(x,y)[:,0]
    return uo

def ux(t, x, y):
    g = u(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False)#.cuda()
    u_x = grad(g, x, create_graph=True, grad_outputs=grad_outputs)[0]
    return u_x

def uy(t, x, y):
    g = u(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False)#.cuda()
    u_y = grad(g, y, create_graph=True, grad_outputs=grad_outputs)[0]
    return u_y

def f(t, x, y):
    g = u(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False)#.cuda()
    u_t = grad(g, t, create_graph=True, grad_outputs=grad_outputs)[0]
    u_x = grad(g, x, create_graph=True, grad_outputs=grad_outputs)[0]
    u_xx = grad(u_x, x, create_graph=True, grad_outputs=grad_outputs)[0]
    u_y = grad(g, y, create_graph=True, grad_outputs=grad_outputs)[0]
    u_yy = grad(u_y, y, create_graph=True, grad_outputs=grad_outputs)[0]
    f = u_t - 0.001*u_xx-0.001*u_yy+g*g*g - g
    return f

# Define loss and optimizer
MSE = nn.MSELoss()
#optimizer = AdamGDA(list(model.parameters()) + list([w]), dim_max=4, lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.2)  # 0.0005 -> 0.0001 -> 0.00002 -> 0.000004 -> 0.0000008
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.97)

error = 1.0
a = np.zeros((1,6))
b = np.zeros(1)
t = T_train.shape[0]//2
# Train the model
for epoch in range(num_epochs):

    cost1 = MSE(u(T_train[:t], X_train[:t], Y_train[:t]), Z_train[:t])
    # differential operator
    cost2 = (f(T_i, X_i, Y_i) ** 2).mean()
    # intial condition
    cost3 = ((u(T_0, X_0, Y_0) - 0.5 * (torch.sin(4.0 * pi * X_0) + torch.sin(4.0 * pi * Y_0))) ** 2).mean()
    # periodic boundary condition
    #cost4 = MSE(u(T_xlb, X_xlb, Y_xlb), u(T_xub, X_xub, Y_xub)) + MSE(ux(T_xlb, X_xlb, Y_xlb), ux(T_xub, X_xub, Y_xub)) \
    #        + MSE(u(T_ylb, X_ylb, Y_ylb), u(T_xub, X_xub, Y_xub)) + MSE(uy(T_xub, X_xub, Y_xub), uy(T_ylb, X_ylb, Y_ylb))
    # zero flux boundary condition
    cost4 =    (((ux(T_xlb, X_xlb, Y_xlb)+0.1) ** 2).mean() + ((ux(T_xub, X_xub, Y_xub)+0.1) ** 2).mean() \
               + ((uy(T_ylb, X_ylb, Y_ylb)-0.1) ** 2).mean() + ((uy(T_yub, X_yub, Y_yub)-0.1) ** 2).mean()) 

    w_norm = torch.exp(w[0]) + torch.exp(w[1]) + torch.exp(w[2]) + torch.exp(w[3])
    w1 = torch.exp(w[0]) / w_norm
    w2 = torch.exp(w[1]) / w_norm
    w3 = torch.exp(w[2]) / w_norm
    w4 = torch.exp(w[3]) / w_norm

    cost = ( cost2 ** 2  + cost3**2 + cost4 ** 2) / ( cost2  + cost3 + cost4)
    ncost = 1.0/4.0 * (cost1 + cost2 + cost3 + cost4)
    error = ncost.item()

    if epoch % display_step == 0: # optimizer.start and 
        print(
            'Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
                epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))
        a = np.append(a, [[epoch, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()]], axis=0)

    if error < error_threshold:
        print(
            'Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}, '.format(
                epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))
        a = np.append(a, [[epoch, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()]], axis=0)

    # optimize
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    scheduler.step()
    if error < error_threshold:
    #     print('norm(f): {:.5f}'.format(torch.norm(optimizer.forces0)))
    #     b = np.append(b, [torch.norm(optimizer.forces0)], axis=0)
         break
    # if epoch % display_step == 0:
    #     print('norm(f): {:.5f}'.format(torch.norm(optimizer.forces0)))
    #     b = np.append(b, [torch.norm(optimizer.forces0)], axis=0)

stop = timeit.default_timer()
torch.save(model.state_dict(), './Allen_Cahn_equation/GDA.pkl')
print("Running time")
runnint_time = stop - start
print(runnint_time)
np.savetxt('./Allen_Cahn_equation/Adam_Training_hist.csv', a, delimiter=",")
np.savetxt('./Allen_Cahn_equation/Adam_Forces.csv', b, delimiter=",")
f = open('./Allen_Cahn_equation/Training_time_GDA.txt','w')
f.write(str(runnint_time))
f.close()

# plot loss 
plt.figure()
plt.plot(a[1:, 0], a[1:, 2], 'y', label='data loss')
plt.plot(a[1:, 0], a[1:, 3], 'r', label='PDE loss')
plt.plot(a[1:, 0], a[1:, 4], 'g', label='IC loss')
plt.plot(a[1:, 0], a[1:, 5], 'b', label='BC loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title(Mymodel + ' Error')
plt.yscale('log')
plt.legend()
plt.savefig('./Allen_Cahn_equation/' + Mymodel + '_loss.png')
plt.show()
plt.close()

# Evaluation
time=1.0
my_data = genfromtxt('./Data/allen_cahn_flux0.1_100%.csv', delimiter=',', skip_header=1) # using comsol datae
X = my_data[:, 0]
Y = my_data[:, 1]
Z_truth1= my_data[:,22]
Z_truth0= my_data[:,2]
Z_truth2= my_data[:,-1]
Z_truth0_5= my_data[:,12]
Z_truth1_5= my_data[:,32]

# Convert numpy arrays to torch tensors
X= Tensor(np.reshape(X,(X.size,1)))
Y= Tensor(np.reshape(Y,(Y.size,1)))

L_xsample = 33
L_ysample = 33

T0 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*0.0)
T0_5 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*0.5)
T1 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*1.0)
T1_5 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*1.5)
T2 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*2.0)

# load data for evluation
model.load_state_dict(torch.load('./Allen_Cahn_equation/GDA.pkl'))
def u(t, x, y):
    uo = model(torch.cat((t,x,y),1))
    if t.eq(0.0).all():
        uo[:,0] = u0(x,y)[:,0]
    return uo

Z0 = u(T0, X, Y).cpu().detach().numpy().squeeze()
Z0_5 = model(torch.cat((T0_5, X, Y),1)).cpu().detach().numpy().squeeze()
Z1 = model(torch.cat((T1, X, Y),1)).cpu().detach().numpy().squeeze()
Z1_5 = model(torch.cat((T1_5, X, Y),1)).cpu().detach().numpy().squeeze()
Z2 = model(torch.cat((T2, X, Y),1)).cpu().detach().numpy().squeeze()

print('rmse at t=0.0s:', np.sqrt(np.mean((Z0 - Z_truth0) ** 2)))
print('rmse at t=0.5s:', np.sqrt(np.mean((Z0_5 - Z_truth0_5) ** 2)))
print('rmse at t=1.0s:', np.sqrt(np.mean((Z1 - Z_truth1) ** 2)))
print('rmse at t=1.5s:', np.sqrt(np.mean((Z1_5 - Z_truth1_5) ** 2)))
print('rmse at t=2.0s:', np.sqrt(np.mean((Z2 - Z_truth2) ** 2)))

X=np.squeeze(X.cpu().detach().numpy())
Y=np.squeeze(Y.cpu().detach().numpy())
Z1=np.squeeze(Z1)

c = np.zeros(shape=(L_xsample * L_ysample, 3))
c[:, 0] = X
c[:, 1] = Y
c[:, 2] = Z1
np.savetxt(Mymodel +'-'+ str(time) + '.csv', c, delimiter=",")

matplotlib.rcParams.update({'font.size': 16})
# plot figure prediction
plotfigure(X, Y, Z0, Mymodel, 0.0, './Allen_Cahn_equation/')
plotfigure(X, Y, Z0_5, Mymodel, 0.5, './Allen_Cahn_equation/')
plotfigure(X, Y, Z1, Mymodel, 1.0, './Allen_Cahn_equation/')
plotfigure(X, Y, Z1_5, Mymodel, 1.5, './Allen_Cahn_equation/')
plotfigure(X, Y, Z2, Mymodel, 2.0, './Allen_Cahn_equation/')

plotfigure(X, Y, Z_truth0, 'Truth ', 0.0, './Allen_Cahn_equation/')
plotfigure(X, Y, Z_truth0_5, 'Truth ', 0.5, './Allen_Cahn_equation/')
plotfigure(X, Y, Z_truth1, 'Truth ', 1.0, './Allen_Cahn_equation/')
plotfigure(X, Y, Z_truth1_5, 'Truth ', 1.5, './Allen_Cahn_equation/')
plotfigure(X, Y, Z_truth2, 'Truth ', 2.0, './Allen_Cahn_equation/')

print('Done')