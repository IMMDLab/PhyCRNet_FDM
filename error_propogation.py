"this code is for error propogation curve of PCNN and PhyCRNet under burgers equation, heat equation and Allen-Cahn equation"
"need to be use in phycrnet model python file due to random seed issue"
import sys
#sys.path.append('/home/gli12/project/ASME2024')
#from Burgers_equation.PhyCRNet_Burgers_update import data_to_img_new

import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.interpolate import griddata
import sys
import os
import pandas as pd
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)


def data_to_img_new(data):
    imgu = np.zeros((32, 32))
    imgv = np.zeros((32, 32))
    for xi, yi, zu, zv in zip(data[0], data[1], data[2], data[3]):
        imgu[int(yi*31),int(xi*31),] = zu
        imgv[int(yi*31),int(xi*31),] = zv

    # plt.figure()
    # plt.imshow(imgv, cmap='viridis')
    # plt.colorbar()
    # plt.savefig(fig_save_path + 'heat_First.png', dpi = 300)
    # plt.close()
    return imgu, imgv

def data_to_img_heat(data):
    imgu = np.zeros((32, 32))
    for xi, yi, zu in zip(data[0], data[1], data[2]):
        imgu[int(yi*31),int(xi*31),] = zu
    return imgu

def to_tensor(data, requires_grad=False):
    data = torch.tensor(data)
    return torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad).to(device)
    #return data.clone().detach().requires_grad_(requires_grad).to(device)

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
model_pcnn = MLP(3, [30, 20, 60, 40, 30, 20], 2).to(device)

model_heat = MLP(3, [30, 20, 30, 20], 1).to(device)


def pcnn_rmse(time): # time is the time step, which is multiples of 0.05

    L_xsample = 33
    L_ysample = 33
    my_data = genfromtxt('./Data/Diffusion_reaction_update.csv', delimiter=',', skip_header=1)
    #sample_indice = np.linspace(0, my_data.shape[0]-1, L_xsample * L_ysample, dtype=int)
    X = my_data[:, 0]
    Y = my_data[:, 1]
    # Convert numpy arrays to torch tensors
    # X= to_tensor(np.reshape(X,(X.size,1)))
    # Y= to_tensor(np.reshape(Y,(Y.size,1)))
    # T = to_tensor(torch.ones(L_xsample * L_ysample, 1)*time)
    X = torch.tensor(np.reshape(X,(X.size,1)), dtype=torch.float32).to(device)
    Y = torch.tensor(np.reshape(Y,(Y.size,1)), dtype=torch.float32).to(device)
    T = torch.tensor(torch.ones(L_xsample * L_ysample, 1)*time, dtype=torch.float32).to(device)
    # load data for evluation
    model_pcnn.load_state_dict(torch.load('./Diffusion_reaction/PCNN_Diffusion_reaction.pkl'))
    Z = model_pcnn(torch.cat((T, X, Y),1))#.cpu().detach().numpy().squeeze()
    if time == 0:
        Z[:,0] = (0.5 * (torch.sin(4.0 * np.pi * X) + torch.sin(4.0 * np.pi * Y))).squeeze()
        Z[:,1] = (0.5 * (torch.cos(4.0 * np.pi * X) + torch.cos(4.0 * np.pi * Y))).squeeze()
    timestep = int(time/0.05)
    Z = Z.cpu().detach().numpy().squeeze()
    ZU, ZV = Z[:,0], Z[:,1]
    ZU_truth, ZV_truth = my_data[:, 2*timestep+2], my_data[:, 2*timestep+3]
    RMSE_u, RMSE_v = np.sqrt(np.mean((ZU - ZU_truth) ** 2)), np.sqrt(np.mean((ZV - ZV_truth) ** 2))
    #print('U, V rmse at {:.4f} s:', time, RMSE_u, ' ', RMSE_v)

    return RMSE_u, RMSE_v, ZU, ZV, ZU_truth, ZV_truth



def phycrnet_rmse(time, output): # time is the time step, which is multiples of 0.05

    timestep = int(time/0.05)
    data = pd.read_csv('./Data/Diffusion_reaction_update.csv', skiprows=0)  # read comsol generated data
    data_timestep = np.array(data)[:, [0,1,2*timestep+2,2*timestep+3]].T 

    ZU_truth, ZV_truth = data_to_img_new(data_timestep)
    ZU, ZV = output[timestep,0].squeeze().cpu().detach().numpy(), output[timestep,1].squeeze().cpu().detach().numpy() 

    u_RMSE, v_RMSE = np.sqrt(np.mean((ZU-ZU_truth)**2)), np.sqrt(np.mean((ZV-ZV_truth)**2)) 
    #print('PhyCRNet U, V rmse at {:.4f} s:', timestep, u_RMSE, ' ', v_RMSE)

    return u_RMSE, v_RMSE, ZU, ZV, ZU_truth, ZV_truth


def error_propogation(output):

    pre_PCNN = []
    tru_PCNN = []
    pre_Phycrnet = []
    tru_Phycrnet = []
    error = []
    error_cr = []

    for i in range(1,42):

        time = (i-1)*0.05
        # curve for pcnn
        _, _, pcnn_u, pcnn_v, pcnn_u_truth, pcnn_v_truth = pcnn_rmse(time)
        pre_PCNN.append([ pcnn_u, pcnn_v])
        tru_PCNN.append([ pcnn_u_truth, pcnn_v_truth])
        arr_pre_PCNN = np.array(pre_PCNN)
        arr_tru_PCNN = np.array(tru_PCNN)
        #RMSE_pre_u, RMSE_pre_v = np.sqrt(np.sum((arr_pre_PCNN[:,0] - arr_tru_PCNN[:,0]) ** 2/arr_tru_PCNN.shape[-1])/41),\
        #                            np.sqrt(np.sum((arr_pre_PCNN[:,1] - arr_tru_PCNN[:,1]) ** 2/arr_tru_PCNN.shape[-1])/41)
        RMSE_pre_u, RMSE_pre_v = np.sqrt(np.sum(np.sum((arr_pre_PCNN[:,0,:,] - arr_tru_PCNN[:,0,:,]) ** 2,axis=1)/pcnn_u.shape[0])/41),\
                                 np.sqrt(np.sum(np.sum((arr_pre_PCNN[:,1,:,] - arr_tru_PCNN[:,1,:,]) ** 2,axis=1)/pcnn_u.shape[0])/41)
        #norm_pre_u, norm_pre_v = np.linalg.norm((arr_pre_PCNN[:,0,:,] - arr_tru_PCNN[:,0,:,]),ord=2), np.linalg.norm((arr_pre_PCNN[:,1,:,] - arr_tru_PCNN[:,1,:,]),ord=2)
        #RMSE_pre_u, RMSE_pre_v = np.sqrt(norm_pre_u**2/(arr_tru_PCNN.shape[-1]*arr_tru_PCNN.shape[-2])/41), np.sqrt(norm_pre_v**2/(arr_tru_PCNN.shape[-1]*arr_tru_PCNN.shape[-2])/41)
        error.append([time, RMSE_pre_u, RMSE_pre_v])

        # curve for phycrnet
        _, _, phycrnet_u, phycrnet_v, phycrnet_u_truth, phycrnet_v_truth = phycrnet_rmse(time, output)
        pre_Phycrnet.append([ phycrnet_u.flatten(), phycrnet_v.flatten()])
        tru_Phycrnet.append([ phycrnet_u_truth.flatten(), phycrnet_v_truth.flatten()])
        arr_pre_Phycrnet = np.array(pre_Phycrnet)
        arr_tru_Phycrnet = np.array(tru_Phycrnet)
        #RMSE_pre_u1, RMSE_pre_v1 = np.sqrt(np.sum((arr_pre_Phycrnet[:,0] - arr_tru_Phycrnet[:,0]) ** 2/(arr_tru_Phycrnet.shape[-1]*arr_tru_Phycrnet.shape[-2]))/41),\
        #                            np.sqrt(np.sum((arr_pre_Phycrnet[:,1] - arr_tru_Phycrnet[:,1]) ** 2/(arr_tru_Phycrnet.shape[-1]*arr_tru_Phycrnet.shape[-2]))/41)
        RMSE_pre_u1, RMSE_pre_v1 = np.sqrt(np.sum(np.sum((arr_pre_Phycrnet[:,0,:,] - arr_tru_Phycrnet[:,0,:,]) ** 2,axis=(1))/(arr_tru_Phycrnet.shape[-1]))/41),\
                                   np.sqrt(np.sum(np.sum((arr_pre_Phycrnet[:,1,:,] - arr_tru_Phycrnet[:,1,:,]) ** 2,axis=(1))/(arr_tru_Phycrnet.shape[-1]))/41)
        #norm_pre_u1, norm_pre_v1 = np.linalg.norm((arr_pre_Phycrnet[:,0,:,].squeeze() - arr_tru_Phycrnet[:,0,:,].squeeze()),ord=2),\
        #                           np.linalg.norm((arr_pre_Phycrnet[:,1,:,].squeeze() - arr_tru_Phycrnet[:,1,:,].squeeze()),ord=2)
        #RMSE_pre_u1, RMSE_pre_v1 = np.sqrt(norm_pre_u1**2/(arr_tru_Phycrnet.shape[-1])/41), np.sqrt(norm_pre_v1**2/(arr_tru_Phycrnet.shape[-1])/41)
        error_cr.append([time, RMSE_pre_u1, RMSE_pre_v1])

    error = np.array(error)[1:,]
    error_cr = np.array(error_cr)[1:,]

    plt.figure()
    #plt.plot(error[:,0], error[:,1], label='RMSE_u')
    #plt.plot(error[:,0], error[:,2], label='RMSE_v')
    plt.plot(error[:,0], (error[:,1]+error[:,2])/2, label='PCNN')
    plt.plot(error[:,0], (error_cr[:,1]+error_cr[:,2])/2, label='PhyCRNet')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.xlabel('t(s)')
    plt.ylabel('a-RMSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./Diffusion_reaction/log_error_propogation.png')
    plt.show()
    plt.close()


"this part is for heat equation and allen-cahn equation"

def pcnn_heat(time):
    L_xsample = 33
    L_ysample = 33
    my_data = genfromtxt('./Data/Allencahn_update3_100%.csv', delimiter=',', skip_header=1)
    X = my_data[:, 0]
    Y = my_data[:, 1]
    # Convert numpy arrays to torch tensors
    # X= to_tensor(np.reshape(X,(X.size,1)))
    # Y= to_tensor(np.reshape(Y,(Y.size,1)))
    # T = to_tensor(torch.ones(L_xsample * L_ysample, 1)*time)
    X = torch.tensor(np.reshape(X,(X.size,1)), dtype=torch.float32).to(device)
    Y = torch.tensor(np.reshape(Y,(Y.size,1)), dtype=torch.float32).to(device)
    T = torch.tensor(torch.ones(L_xsample * L_ysample, 1)*time, dtype=torch.float32).to(device)
    # load data for evluation
    model_heat.load_state_dict(torch.load('./Allen_Cahn_equation/GDA_1e-3.pkl'))
    Z = model_heat(torch.cat((T, X, Y),1))#.cpu().detach().numpy().squeeze()
    #if time == 0:
    #    Z[:,0] = (0.5 * (torch.sin(4.0 * np.pi * X) + torch.sin(4.0 * np.pi * Y))).squeeze()
    timestep = int(time/0.05) 
    Z = Z.cpu().detach().numpy().squeeze()
    Z_truth = my_data[:, timestep+2]
    RMSE_u = np.sqrt(np.mean((Z - Z_truth) ** 2))
    #print('PCNN rmse at {:.4f} s:', time, RMSE_u)

    return RMSE_u, Z, Z_truth


def phycrnet_heat(time, output):
    timestep = int(time/0.05)
    data = pd.read_csv('./Data/Allencahn_update3_100%.csv', skiprows=0)  # read comsol generated data
    data_timestep = np.array(data)[:, [0,1,timestep+2]].T 

    Z_truth = data_to_img_heat(data_timestep)
    Z = output[timestep,0].squeeze().cpu().detach().numpy()

    u_RMSE = np.sqrt(np.mean((Z-Z_truth)**2))
    #print('PhyCRNet rmse at {:.4f} s:', timestep, u_RMSE)

    return u_RMSE, Z, Z_truth

def error_propagation_heat(output):
    pre_PCNN = []
    tru_PCNN = []
    pre_Phycrnet = []
    tru_Phycrnet = []
    error = []
    error_cr = []
    pcnn_rmse_list = []
    phycrnet_rmse_list = []

    for i in range(1,42):

        time = (i-1)*0.05
        # curve for pcnn
        pcnn_rmse, pcnn_u, pcnn_u_truth = pcnn_heat(time) 
        pre_PCNN.append([ pcnn_u])
        tru_PCNN.append([ pcnn_u_truth])
        arr_pre_PCNN = np.array(pre_PCNN)
        arr_tru_PCNN = np.array(tru_PCNN)
        #RMSE_pre_u, RMSE_pre_v = np.sqrt(np.sum((arr_pre_PCNN[:,0] - arr_tru_PCNN[:,0]) ** 2/arr_tru_PCNN.shape[-1])/41)                       
        RMSE_pre_u = np.sqrt(np.sum(np.sum((arr_pre_PCNN[:,0,:,] - arr_tru_PCNN[:,0,:,]) ** 2,axis=1)/pcnn_u.shape[0])/41)
        #norm_pre_u, norm_pre_v = np.linalg.norm((arr_pre_PCNN[:,0,:,] - arr_tru_PCNN[:,0,:,]),ord=2), np.linalg.norm((arr_pre_PCNN[:,1,:,] - arr_tru_PCNN[:,1,:,]),ord=2)
        #RMSE_pre_u, RMSE_pre_v = np.sqrt(norm_pre_u**2/(arr_tru_PCNN.shape[-1]*arr_tru_PCNN.shape[-2])/41), np.sqrt(norm_pre_v**2/(arr_tru_PCNN.shape[-1]*arr_tru_PCNN.shape[-2])/41)
        error.append([time, RMSE_pre_u])
        pcnn_rmse_list.append([time, pcnn_rmse])

        # curve for phycrnet
        phycrnet_rmse, phycrnet_u, phycrnet_u_truth, = phycrnet_heat(time, output)
        pre_Phycrnet.append([ phycrnet_u.flatten()])
        tru_Phycrnet.append([ phycrnet_u_truth.flatten()])
        arr_pre_Phycrnet = np.array(pre_Phycrnet)
        arr_tru_Phycrnet = np.array(tru_Phycrnet)
        #RMSE_pre_u1, RMSE_pre_v1 = np.sqrt(np.sum((arr_pre_Phycrnet[:,0] - arr_tru_Phycrnet[:,0]) ** 2/(arr_tru_Phycrnet.shape[-1]*arr_tru_Phycrnet.shape[-2]))/41),\
        #                            np.sqrt(np.sum((arr_pre_Phycrnet[:,1] - arr_tru_Phycrnet[:,1]) ** 2/(arr_tru_Phycrnet.shape[-1]*arr_tru_Phycrnet.shape[-2]))/41)
        RMSE_pre_u1 = np.sqrt(np.sum(np.sum((arr_pre_Phycrnet[:,0,:,] - arr_tru_Phycrnet[:,0,:,]) ** 2,axis=(1))/(arr_tru_Phycrnet.shape[-1]))/41)
                                   
        #norm_pre_u1, norm_pre_v1 = np.linalg.norm((arr_pre_Phycrnet[:,0,:,].squeeze() - arr_tru_Phycrnet[:,0,:,].squeeze()),ord=2),\
        #                           np.linalg.norm((arr_pre_Phycrnet[:,1,:,].squeeze() - arr_tru_Phycrnet[:,1,:,].squeeze()),ord=2)
        #RMSE_pre_u1, RMSE_pre_v1 = np.sqrt(norm_pre_u1**2/(arr_tru_Phycrnet.shape[-1])/41), np.sqrt(norm_pre_v1**2/(arr_tru_Phycrnet.shape[-1])/41)
        error_cr.append([time, RMSE_pre_u1])
        phycrnet_rmse_list.append([time, phycrnet_rmse])

    error = np.array(error)#[1:,]
    error_cr = np.array(error_cr)[1:,]
    pcnn_rmse_list = np.array(pcnn_rmse_list)
    phycrnet_rmse_list = np.array(phycrnet_rmse_list)[1:,]
    print('a_RMSE for PCNN (0-2):', error[-1])
    print('a_RMSE for PhyCRNet (0-2):', error_cr[-1])
    

    plt.figure()
    #plt.plot(error[:,0], error[:,1], label='RMSE_u')
    #plt.plot(error[:,0], error[:,2], label='RMSE_v')
    plt.plot(error[:,0], error[:,1], label='PCNN')
    plt.plot(error_cr[:,0], error_cr[:,1], label='PhyCRNet')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.xlabel('t(s)')
    plt.ylabel('a-RMSE')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('./Heat_equation/log_error_propogation.png')
    plt.savefig('./Allen_Cahn_equation/log_error_propogation.png')
    plt.show()
    plt.close()

    plt.figure()
    #plt.plot(error[:,0], error[:,1], label='RMSE_u')
    #plt.plot(error[:,0], error[:,2], label='RMSE_v')
    plt.plot(pcnn_rmse_list[:,0], pcnn_rmse_list[:,1], label='PCNN')
    plt.plot(phycrnet_rmse_list[:,0], phycrnet_rmse_list[:,1], label='PhyCRNet')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.xlabel('t(s)')
    plt.ylabel('RMSE')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('./Heat_equation/log_error_propogation.png')
    plt.savefig('./Allen_Cahn_equation/log_error.png')
    plt.show()
    plt.close()





if __name__ == '__main__':
    #error_propogation('PCNN_Heat')
    #pcnn_rmse(1.0)
    #phycrnet_rmse(1.0)
    #error_propogation()
    pcnn_heat(1.5)