import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad, Variable
import matplotlib.pyplot as plt
import matplotlib
import timeit
from numpy import genfromtxt
import os
from math import pi
# import python file
import sys
sys.path.append('/home/gli12/project/JCISE2024')
from plot_figure import plotfigure_uv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
model = MLP(3, [30, 20, 60, 40, 30, 20], 2).to(device)

# Initialize weights
for param_tensor in model.state_dict():
    if model.state_dict()[param_tensor].dim() > 1:
        nn.init.xavier_normal_(model.state_dict()[param_tensor])
        model.state_dict()[param_tensor] *= 1.5  # amplify weight

# Training parameters
learning_rate = 5e-4
error_threshold = 1e-4
num_epochs = int(9e4 + 1)
display_step = int(100)

# Sampling parameters
num_tsample = 21
num_xsample = 31#11
num_ysample = 31#11
total_time = 1.0

# Load data
#my_data = genfromtxt('./Heat equation/Training_Heat_FEM.csv', delimiter=',')
# my_data = genfromtxt('./Burgers_equation/heat equation flux.csv', delimiter=',', skip_header=1)
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
# data 
my_data_train = genfromtxt('./Data/burgers_update1_20%_training.csv', delimiter=',', skip_header=1)
X_train = my_data_train[:, 0]
Y_train = my_data_train[:, 1]
ZU_train = my_data_train[:, 3]
ZV_train = my_data_train[:, 4]
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
ZU_train = to_tensor(ZU_train).unsqueeze(1).requires_grad_(True)
ZV_train = to_tensor(ZV_train).unsqueeze(1).requires_grad_(True)

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
    #return nn.MSELoss()(f1, torch.zeros_like(f1).cuda()) + nn.MSELoss()(f2, torch.zeros_like(f2).cuda())

# Loss function and optimizer
MSE = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.97) #(optimizer, step_size=20000, gamma=0.2)  # 0.0005 -> 0.0001 -> 0.00002 -> 0.000004 -> 0.0000008

# Training loop
start = timeit.default_timer()
error = 1.0
a = np.zeros((1, 6))

t = T_train.shape[0]//2

for epoch in range(num_epochs):
    # data loss
    cost1 = (MSE(uv(T_train[:t], X_train[:t], Y_train[:t])[0], ZU_train[:t]) + MSE(uv(T_train[:t], X_train[:t], Y_train[:t])[1], ZV_train[:t]))
    # differential operator
    cost2 = (f_PDE(T_i, X_i, Y_i) ** 2).mean()#6*f_PDE(T_i, X_i, Y_i) 
    # intial condition
    uu, vv = uv(T_0, X_0, Y_0)
    cost3 = (((uu - 0.5 * (torch.sin(4.0 * pi * X_0) + torch.cos(4.0 * pi * Y_0))) ** 2).mean() + \
                 ((vv - 0.5 * (torch.cos(4.0 * pi * X_0) + torch.sin(4.0 * pi * Y_0))) ** 2).mean())

    uxl_x, vxl_x = ux(T_xlb, X_xlb, Y_xlb)  # (tbx, xl, ybx)
    uxr_x, vxr_x = ux(T_xub, X_xub, Y_xub)  # (tbx, xr, ybx)
    uyl_y, vyl_y = uy(T_ylb, X_ylb, Y_ylb)  # (tby, xby, yl)
    uyr_y, vyr_y = uy(T_yub, X_yub, Y_yub)  # (tby, xby, yr)
    uxl, vxl = uv(T_xlb, X_xlb, Y_xlb)  # (tbx, xl, ybx)
    uxr, vxr = uv(T_xub, X_xub, Y_xub)  # (tbx, xr, ybx)
    uyl, vyl = uv(T_ylb, X_ylb, Y_ylb)  # (tby, xby, yl)
    uyr, vyr = uv(T_yub, X_yub, Y_yub)  # (tby, xby, yr)

    "Mix boundary condition"
    cost4 = (((uxl - 0.5*torch.cos(4*pi*Y_xlb))**2) + ((vxl_x)**2) \
           + ((uxr - 0.5*torch.cos(4*pi*Y_xub))**2) + ((vxr_x)**2) \
           + ((uyl_y)**2) + ((vyl - 0.5*torch.cos(4*pi*X_ylb))**2) \
           + ((uyr_y)**2) + ((vyr - 0.5*torch.cos(4*pi*X_yub))**2)).mean()
    
    cost =  ( cost1 **2 + cost2 ** 2 + cost3 ** 2 +cost4**2) / ( cost1 +cost2 + cost3 +cost4)
    #cost =  (cost1 **2 + cost4 **2 +  cost3 ** 2)/ (cost1 + cost4  + cost3)
    ncost = 1.0/4.0 * (cost1 + cost2 + cost3 + cost4)
    error = ncost.item()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step() # update weights
    scheduler.step() # update learning rate 
    

    if epoch % display_step == 0:
        print('Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
            epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))
        a = np.append(a, [[epoch, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()]], axis=0)

    if error < error_threshold:
        print('Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
            epoch, num_epochs, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()))
        a = np.append(a, [[epoch, error, cost1.item(), cost2.item(), cost3.item(), cost4.item()]], axis=0)
        break

torch.save(model.state_dict(), './Burgers_equation/PCNN_burgers.pkl')

final_learning_rate = optimizer.param_groups[0]['lr']
print(f'Training {num_epochs} epochs, final learning rate: {final_learning_rate}')


Mymodel = 'PCNN_Burgers'
stop = timeit.default_timer()
print("Running time: {:.2f} seconds".format(stop - start))
np.savetxt('./Burgers_equation/' + Mymodel + '_Training_hist.csv', a, delimiter=",")
f = open('./Burgers_equation/' + Mymodel + '_Training_time.txt', 'w')
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
plt.title(Mymodel + ' Error')
plt.yscale('log')
plt.legend()
plt.savefig('./Burgers_equation/' + Mymodel + '_loss.png')
plt.show()
plt.close()

# Evaluation
time=1.0
L_xsample = 33
L_ysample = 33
my_data = genfromtxt('./Data/burgers_update1_100%.csv', delimiter=',', skip_header=1)
#sample_indice = np.linspace(0, my_data.shape[0]-1, L_xsample * L_ysample, dtype=int)
X = my_data[:, 0]
Y = my_data[:, 1]
ZU_truth1= my_data[:,42]
ZU_truth0= my_data[:,2]
ZU_truth2= my_data[:,-2]
ZU_truth0_5= my_data[:,22]
ZU_truth1_5= my_data[:,62]

ZV_truth1= my_data[:,43]
ZV_truth0= my_data[:,3]
ZV_truth2= my_data[:,-1]
ZV_truth0_5= my_data[:,23]
ZV_truth1_5= my_data[:,63]
# Convert numpy arrays to torch tensors
X= to_tensor(np.reshape(X,(X.size,1)))
Y= to_tensor(np.reshape(Y,(Y.size,1)))

T0 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*0.0)
T0_5 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*0.5)
T1 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*1.0)
T1_5 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*1.5)
T2 = to_tensor(torch.ones(L_xsample * L_ysample, 1)*2.0)

#Z1 = u(T1, X, Y).detach().cpu().numpy()
#Z0 = u(T0, X, Y).detach().cpu().numpy().squeeze()

# load data for evluation
model.load_state_dict(torch.load('./Burgers_equation/PCNN_burgers.pkl'))
Z0 = model(torch.cat((T0, X, Y),1)).cpu().detach().numpy().squeeze()
Z0_5 = model(torch.cat((T0_5, X, Y),1)).cpu().detach().numpy().squeeze()
Z1 = model(torch.cat((T1, X, Y),1)).cpu().detach().numpy().squeeze()
Z1_5 = model(torch.cat((T1_5, X, Y),1)).cpu().detach().numpy().squeeze()
Z2 = model(torch.cat((T2, X, Y),1)).cpu().detach().numpy().squeeze()

ZU0, ZV0 = Z0[:,0], Z0[:,1]
ZU0_5, ZV0_5 = Z0_5[:,0], Z0_5[:,1]
ZU1, ZV1 = Z1[:,0], Z1[:,1]
ZU1_5, ZV1_5 = Z1_5[:,0], Z1_5[:,1]
ZU2, ZV2 = Z2[:,0], Z2[:,1]
print('U, V rmse at t=0.0s:', np.sqrt(np.mean((ZU0 - ZU_truth0) ** 2)), ' ', np.sqrt(np.mean((ZV0 - ZV_truth0) ** 2)))
print('U, V rmse at t=0.5s:', np.sqrt(np.mean((ZU0_5 - ZU_truth0_5) ** 2)), ' ', np.sqrt(np.mean((ZV0_5 - ZV_truth0_5) ** 2)))
print('U, V rmse at t=1.0s:', np.sqrt(np.mean((ZU1 - ZU_truth1) ** 2)), ' ', np.sqrt(np.mean((ZV1 - ZV_truth1) ** 2)))
print('U, V rmse at t=1.5s:', np.sqrt(np.mean((ZU1_5 - ZU_truth1_5) ** 2)), ' ', np.sqrt(np.mean((ZV1_5 - ZV_truth1_5) ** 2)))
print('U, V rmse at t=2.0s:', np.sqrt(np.mean((ZU2 - ZU_truth2) ** 2)), ' ', np.sqrt(np.mean((ZV2 - ZV_truth2) ** 2)))

X=np.squeeze(X.detach().cpu().numpy())
Y=np.squeeze(Y.detach().cpu().numpy())
Z1=np.squeeze(Z1)

c = np.zeros(shape=(L_xsample * L_ysample, 2))
c[:, 0] = X
c[:, 1] = Y
#c[:, 2] = Z1
np.savetxt('./Burgers_equation/'+Mymodel +'-'+ str(time) + '.csv', c, delimiter=",")

# plot figure prediction
matplotlib.rcParams.update({'font.size': 16})
plotfigure_uv(X, Y, ZU0, ZV0, Mymodel, 0.0, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU0_5, ZV0_5, Mymodel, 0.5, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU1, ZV1, Mymodel, 1.0, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU1_5, ZV1_5, Mymodel, 1.5, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU2, ZV2, Mymodel, 2.0, './Burgers_equation/Results/')

plotfigure_uv(X, Y, ZU_truth0, ZV_truth0, 'Truth-Burgers', 0.0, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU_truth0_5, ZV_truth0_5, 'Truth-Burgers', 0.5, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU_truth1, ZV_truth1, 'Truth-Burgers', 1.0, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU_truth1_5, ZV_truth1_5, 'Truth-Burgers', 1.5, './Burgers_equation/Results/')
plotfigure_uv(X, Y, ZU_truth2, ZV_truth2, 'Truth-Burgers', 2.0, './Burgers_equation/Results/')

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