import sys
sys.path.append('/home/gli12/project/PhyCRNet')
from phyCRNet_AM.PhyCRNet_burgers_ori_true import PhyCRNet, load_checkpoint

import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.interpolate import griddata
import sys
import torch.nn as nn
import matplotlib

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

# model_u = MLP(3, [30, 20, 30, 20], 1)
# model_v = MLP(3, [30, 20, 30, 20], 1)
model = MLP(3, [30, 20, 30, 20], 2)


data_dir = '/home/gli12/project/PhyCRNet/data/2dBurgers/burgers_ic_2001x2x64x64.mat'
my_data = scio.loadmat(data_dir)


# Load the data at specified time step
def get_data_time(my_data,t):
    pixel_data_u = np.array([(x, y, pixel_value) for (y, row) in enumerate(my_data['uv'][t][0]) for (x, pixel_value) in enumerate(row)])
    ZU = pixel_data_u[:,2]
    pixel_data_v = np.array([(x, y, pixel_value) for (y, row) in enumerate(my_data['uv'][t][1]) for (x, pixel_value) in enumerate(row)])
    ZV = pixel_data_v[:,2]
    return ZU, ZV

def get_output_phycrnet(output,output_extro,truth):
    pre_u = []
    pre_v = []
    pre_cr_ulist = []
    pre_cr_vlist = []
    truth_u = []
    truth_v = []
    cr_truth_u = []
    cr_truth_v = []
    error = []
    error_cr = []   
    L_xsample = 64
    L_ysample = 64
    x = np.linspace(0, 63, 64)/63.0
    y = np.linspace(0, 63, 64)/63.0
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X= torch.Tensor(np.reshape(X,(X.size,1)))
    Y= torch.Tensor(np.reshape(Y,(Y.size,1)))
    for i in range(100):
        # curve for pcnn
        time = (i+1)*4/100
        ZU, ZV = get_data_time(my_data, int(time/0.002))  # get ground truth u, v

        T1 = torch.ones(L_xsample * L_ysample, 1)*time

        # model_u.load_state_dict(torch.load('/home/gli12/project/PhyCRNet/PCNN_onlydata_u.pkl'))
        # model_v.load_state_dict(torch.load('/home/gli12/project/PhyCRNet/PCNN_onlydata_v.pkl'))
        # ZU1 = model_u(torch.cat((T1, X, Y),1)).cpu().detach().numpy()
        # ZV1 = model_v(torch.cat((T1, X, Y),1)).cpu().detach().numpy()
        model.load_state_dict(torch.load('/home/gli12/project/PhyCRNet/PCNN_Burger/PCNN_one_model_no_data_uv.pkl'))
        ZU1, ZV1 = model(torch.cat((T1, X, Y),1)).cpu().detach().numpy()[:,0], model(torch.cat((T1, X, Y),1)).cpu().detach().numpy()[:,1]


        pre_u.append(ZU1)
        pre_v.append(ZV1)
        truth_u.append(ZU)
        truth_v.append(ZV)
        cr_truth_u.append(truth[int(time/0.002)][0])
        cr_truth_v.append(truth[int(time/0.002)][1])   

        arr_u = np.array(pre_u).squeeze()
        arr_v = np.array(pre_v).squeeze()
        arr_truth_u = np.array(truth_u)
        arr_truth_v = np.array(truth_v)
        arr_cr_truth_u = np.array(cr_truth_u)
        arr_cr_truth_v = np.array(cr_truth_v)

        RMSE_u, RMSE_v = np.sqrt(np.sum((arr_u - arr_truth_u) ** 2/4096)/100), np.sqrt(np.sum((arr_v - arr_truth_v) ** 2/4096)/100)
        error.append([time, RMSE_u, RMSE_v])
        #print('t:{:.5f}, RMSE_u: {:.5f}, RMSE_v: {:.5f}'.format(time, RMSE_u, RMSE_v))

        RMSE_u1, RMSE_v1 = np.sqrt(np.mean((ZU - ZU1) ** 2)), np.sqrt(np.mean((ZV - ZV1) ** 2))
        print('t:{:.5f}, RMSE_u: {:.5f}, RMSE_v: {:.5f}'.format(time, RMSE_u1, RMSE_v1))

        # curve for phycrnet
        if time <= 2:
            output = output
        else:
            output = output_extro
            time = time-2

        pred_u_cr = output[int(time/0.002), 0, :, :].cpu().detach().numpy()
        pred_v_cr = output[int(time/0.002), 1, :, :].cpu().detach().numpy()
        pre_cr_ulist.append(pred_u_cr)
        pre_cr_vlist.append(pred_v_cr)
        arr_cru = np.array(pre_cr_ulist).squeeze()
        arr_crv = np.array(pre_cr_vlist).squeeze()
        RMSE_cr_u, RMSE_cr_v = np.sqrt(np.sum((arr_cru - arr_cr_truth_u) ** 2/4096)/100), np.sqrt(np.sum((arr_crv - arr_cr_truth_v) ** 2/4096)/100)
        error_cr.append([time, RMSE_cr_u, RMSE_cr_v])

    return error, error_cr

