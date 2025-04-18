import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad, Variable
import matplotlib.pyplot as plt
import matplotlib
import timeit
from numpy import genfromtxt
import os
from dualdimer import DualDimer
# import python file
import sys
sys.path.append('/home/gli12/project/ASME2024')
from plot_figure import plotfigure

# Device configuration
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.manual_seed(0)

# MLP model definition
class MLP(nn.Module):
    def __init__(self, in_size, h_sizes, out_size):
        super(MLP, self).__init__()
        self.input = nn.Linear(in_size, h_sizes[0])
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        for layer in self.hidden:
            x = torch.tanh(layer(x))
            # x = layer(x) # not working properly
        output = self.out(x)
        return output

# Model instance and move to device
model = MLP(3, [30, 20, 30, 20], 1).to(device)

# Initialize weights
for param_tensor in model.state_dict():
    if model.state_dict()[param_tensor].dim() > 1:
        nn.init.xavier_normal_(model.state_dict()[param_tensor])

# Training parameters
learning_rate = 5e-4
error_threshold = 1e-4
num_epochs = int(9e4 + 1)
display_step = int(500)

# Sampling parameters
num_tsample = 21
num_xsample = 11
num_ysample = 11
total_time = 1.0

# Load data
#my_data = genfromtxt('./Heat equation/Training_Heat_FEM.csv', delimiter=',')
# my_data = genfromtxt('./Heat_equation/heat equation flux.csv', delimiter=',', skip_header=1)
# X = my_data[:, 0][:my_data.shape[0]//2]
# Y = my_data[:, 1][:my_data.shape[0]//2]
# Z = my_data[:, 2][:my_data.shape[0]//2] 
# T = my_data[:, 0][:my_data.shape[0]//2] 

# convert to tensor
# T = to_tensor(np.reshape(T, (T.size, 1)), True)
# X = to_tensor(np.reshape(X, (X.size, 1)), True)
# Y = to_tensor(np.reshape(Y, (Y.size, 1)), True)
# Z = to_tensor(np.reshape(Z, (Z.size, 1)))

# load training data
my_data_train = genfromtxt('./Heat_equation/heat equation flux_0.5_training.csv', delimiter=',', skip_header=1)
X_train = my_data_train[:, 0]
Y_train = my_data_train[:, 1]
Z_train = my_data_train[:, 3]
T_train = my_data_train[:, 2] 

# Convert numpy arrays to torch tensors and move to device
def to_tensor(data, requires_grad=False):
    return torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad).to(device)

def normalize(data):
    return 2*(data - data.min()) / (data.max() - data.min()) -1

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

T_xlb = to_tensor(s_xlb[:, 0]).unsqueeze(1).requires_grad_(True) # left boundary
X_xlb = to_tensor(s_xlb[:, 1]).unsqueeze(1).requires_grad_(True)
Y_xlb = to_tensor(s_xlb[:, 2]).unsqueeze(1).requires_grad_(True)

T_xub = to_tensor(s_xub[:, 0]).unsqueeze(1).requires_grad_(True) # right boundary
X_xub = to_tensor(s_xub[:, 1]).unsqueeze(1).requires_grad_(True)
Y_xub = to_tensor(s_xub[:, 2]).unsqueeze(1).requires_grad_(True)

T_ylb = to_tensor(s_ylb[:, 0]).unsqueeze(1).requires_grad_(True) # bottom boundary
X_ylb = to_tensor(s_ylb[:, 1]).unsqueeze(1).requires_grad_(True)
Y_ylb = to_tensor(s_ylb[:, 2]).unsqueeze(1).requires_grad_(True)

T_yub = to_tensor(s_yub[:, 0]).unsqueeze(1).requires_grad_(True) # top boundary
X_yub = to_tensor(s_yub[:, 1]).unsqueeze(1).requires_grad_(True)
Y_yub = to_tensor(s_yub[:, 2]).unsqueeze(1).requires_grad_(True)

T_0 = to_tensor(s_0[:, 0]).unsqueeze(1).requires_grad_(True)
X_0 = to_tensor(s_0[:, 1]).unsqueeze(1).requires_grad_(True)
Y_0 = to_tensor(s_0[:, 2]).unsqueeze(1).requires_grad_(True)

T_train = to_tensor(T_train).unsqueeze(1).requires_grad_(True)
X_train = to_tensor(X_train).unsqueeze(1).requires_grad_(True)
Y_train = to_tensor(Y_train).unsqueeze(1).requires_grad_(True)
Z_train = to_tensor(Z_train).unsqueeze(1).requires_grad_(True)

# Define differential operator functions
def u(t, x, y):
    inputs = torch.cat((t, x, y), 1)
    return model(inputs)

def ux(t, x, y):
    g = u(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False)
    u_x = grad(g, x, create_graph=True, grad_outputs=grad_outputs)[0]
    return u_x

def uy(t, x, y):
    g = u(t, x, y)
    grad_outputs = Variable(torch.ones(t.shape[0], 1), requires_grad=False)
    u_y = grad(g, y, create_graph=True, grad_outputs=grad_outputs)[0]
    return u_y

def f(t, x, y):
    g = u(t, x, y)
    g_t = grad(g.sum(), t, create_graph=True)[0]
    g_x = grad(g.sum(), x, create_graph=True)[0]
    g_xx = grad(g_x.sum(), x, create_graph=True)[0]
    g_y = grad(g.sum(), y, create_graph=True)[0]
    g_yy = grad(g_y.sum(), y, create_graph=True)[0]
    return g_t - 0.01 * g_xx - 0.01 * g_yy

# Loss function and optimizer
MSE = nn.MSELoss()

#weigths of different losses
w = torch.randn(4, 1, requires_grad=True)#.to(device)
nn.init.xavier_normal_(w)
#optimizer = DualDimer(list(model.parameters()) + list([w]), dim_max=4, lr=learning_rate)
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

# Training loop
start = timeit.default_timer()
error = 1.0
a = np.zeros((1, 6))

for epoch in range(num_epochs):
    if epoch > 50000:
        optimizer = optimizer_lbfgs
    else:
        optimizer = optimizer_adam
    def closure():
        optimizer.zero_grad()
        cost1 = MSE(u(T_train, X_train, Y_train), Z_train)
        cost2 = (f(T_i, X_i, Y_i) ** 2).mean()
        cost3 = ((u(T_0, X_0, Y_0) - 0.5 * (torch.sin(4.0 * np.pi * X_0) + torch.sin(4.0 * np.pi * Y_0))) ** 2).mean()
        #cost4 = (ux(T_xlb, X_xlb, Y_xlb) ** 2).mean() + (ux(T_xub, X_xub, Y_xub) ** 2).mean() \
        #         + (uy(T_ylb, X_ylb, Y_ylb) ** 2).mean() + (uy(T_xub, X_xub, Y_xub) ** 2).mean()
        cost4 =    ((ux(T_xlb, X_xlb, Y_xlb)) ** 2).mean() + ((ux(T_xub, X_xub, Y_xub)) ** 2).mean() \
                    + ((uy(T_ylb, X_ylb, Y_ylb)) ** 2).mean() + ((uy(T_yub, X_yub, Y_yub)) ** 2).mean()   
        
        # w_norm = torch.exp(w[0]) + torch.exp(w[1]) + torch.exp(w[2]) + torch.exp(w[3])
        # w1 = torch.exp(w[0]) / w_norm
        # w2 = torch.exp(w[1]) / w_norm
        # w3 = torch.exp(w[2]) / w_norm
        # w4 = torch.exp(w[3]) / w_norm

        cost =  ( cost2 ** 2 + cost3 ** 2 + cost4 ** 2) / ( cost2 + cost3 + cost4)
        #cost = w2 * cost2 + w3 * cost3 + w4 * cost4
        ncost = 1.0/4.0 * (cost1 + cost2 + cost3 + cost4)
        
        global error
        global a

        error = ncost.item()
        
        cost.backward()
        #optimizer.step()

        if epoch % display_step == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
                epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))
            a = np.append(a, [[epoch, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()]], axis=0)

        if error < error_threshold:
            print('Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
                epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))
            a = np.append(a, [[epoch, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()]], axis=0)
            # break

        return cost
    
    optimizer.step(closure)

torch.save(model.state_dict(), './Heat_equation/PCNN_heat.pkl')


Mymodel = 'PCNN_Heat'
stop = timeit.default_timer()
print("Running time: {:.2f} seconds".format(stop - start))
np.savetxt('./Heat_equation/' + Mymodel + '_Training_hist.csv', a, delimiter=",")
f = open('./Heat_equation/' + Mymodel + '_Training_time.txt', 'w')
f.write(str(stop - start))
f.close()

# plot loss 
plt.figure()
plt.plot(a[1:, 0], a[1:, 2], 'y', label='data loss')
plt.plot(a[1:, 0], a[1:, 3], 'r', label='PDE loss')
plt.plot(a[1:, 0], a[1:, 4], 'g', label='IC loss')
plt.plot(a[1:, 0], a[1:, 5], 'b', label='BC loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.title(Mymodel + ' Error')
plt.yscale('log')
plt.legend()
plt.savefig('./Heat_equation/' + Mymodel + '_loss.png')
plt.show()
plt.close()

# Evaluation
time=1.0
L_xsample = 33
L_ysample = 33
my_data = genfromtxt('./Heat_equation/heat equation flux.csv', delimiter=',', skip_header=1)
#sample_indice = np.linspace(0, my_data.shape[0]-1, L_xsample * L_ysample, dtype=int)
X = my_data[:, 0]
Y = my_data[:, 1]
Z_truth1= my_data[:,102]
Z_truth0= my_data[:,2]
Z_truth2= my_data[:,-1]
Z_truth0_5= my_data[:,52]
Z_truth1_5= my_data[:,152]
# Convert numpy arrays to torch tensors
X= to_tensor(np.reshape(X,(X.size,1)))
Y= to_tensor(np.reshape(Y,(Y.size,1)))

T0 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*0.0)
T0_5 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*0.5)
T1 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*1.0)
T1_5 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*1.5)
T2 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*2.0)

Z1 = u(T1, X, Y).detach().cpu().numpy()
Z0 = u(T0, X, Y).detach().cpu().numpy().squeeze()

# load data for evluation
model.load_state_dict(torch.load('./Heat_equation/PCNN_heat.pkl'))
Z0 = model(torch.cat((T0, X, Y),1)).cpu().detach().numpy().squeeze()
Z0_5 = model(torch.cat((T0_5, X, Y),1)).cpu().detach().numpy().squeeze()
Z1 = model(torch.cat((T1, X, Y),1)).cpu().detach().numpy().squeeze()
Z1_5 = model(torch.cat((T1_5, X, Y),1)).cpu().detach().numpy().squeeze()
Z2 = model(torch.cat((T2, X, Y),1)).cpu().detach().numpy().squeeze()

print('rmse at t=0.0s:', np.sqrt(np.mean((Z0 - Z_truth0) ** 2)))
print('rmse at t=0.5s:', np.sqrt(np.mean((Z0_5 - Z_truth0_5) ** 2)))
print('rmse at t=1.0s:', np.sqrt(np.mean((Z1 - Z_truth1) ** 2)))
print('rmse at t=1.5s:', np.sqrt(np.mean((Z1_5 - Z_truth1_5) ** 2)))
print('rmse at t=2.0s:', np.sqrt(np.mean((Z2 - Z_truth2) ** 2)))

X=np.squeeze(X.detach().cpu().numpy())
Y=np.squeeze(Y.detach().cpu().numpy())
Z1=np.squeeze(Z1)

c = np.zeros(shape=(L_xsample * L_ysample, 3))
c[:, 0] = X
c[:, 1] = Y
c[:, 2] = Z1
np.savetxt('./Heat_equation/'+Mymodel +'-'+ str(time) + '.csv', c, delimiter=",")

# plot figure prediction
matplotlib.rcParams.update({'font.size': 16})
plotfigure(X, Y, Z0, Mymodel, 0.0, './Heat_equation/')
plotfigure(X, Y, Z0_5, Mymodel, 0.5, './Heat_equation/')
plotfigure(X, Y, Z1, Mymodel, 1.0, './Heat_equation/')
plotfigure(X, Y, Z1_5, Mymodel, 1.5, './Heat_equation/')
plotfigure(X, Y, Z2, Mymodel, 2.0, './Heat_equation/')

plotfigure(X, Y, Z_truth0, 'Truth ', 0.0, './Heat_equation/')
plotfigure(X, Y, Z_truth0_5, 'Truth ', 0.5, './Heat_equation/')
plotfigure(X, Y, Z_truth1, 'Truth ', 1.0, './Heat_equation/')
plotfigure(X, Y, Z_truth1_5, 'Truth ', 1.5, './Heat_equation/')
plotfigure(X, Y, Z_truth2, 'Truth ', 2.0, './Heat_equation/')

print('Done')
















# plt.tricontour(X, Y, Z1, 15, cmap='jet')
# plt.tricontourf(X, Y, Z1, 15, cmap='jet')
# cbar=plt.colorbar()
# #cbar.set_label('Temperature')
# plt.plot(X, Y, 'ko', ms=3)
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.title(Mymodel + '-Temperature at t=' + str(time))
# plt.savefig('./Heat equation/'+Mymodel + '-Temperature' + str(time) + '.png')
# plt.show()
# plt.close()


# plt.tricontour(X, Y, Z0, 15, cmap='jet')
# plt.tricontourf(X, Y, Z0, 15, cmap='jet')
# cbar=plt.colorbar()
# #cbar.set_label('Temperature')
# plt.plot(X, Y, 'ko', ms=3)
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.title(Mymodel + '-Temperature at t=' + str(time))
# plt.savefig('./Heat equation/'+Mymodel + '-Temperature' + str(0.0) + '.png')
# plt.show()
# plt.close()