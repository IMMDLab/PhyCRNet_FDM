'''PhyCRNet for solving Diffusion_reaction PDEs-06/03/2024'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
import cv2
import pandas as pd
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import matplotlib
import sys
from numpy import genfromtxt
sys.path.append('/home/gli12/project/JCISE2024')
from plot_figure import plotfigure_uv
from torchinfo import summary
from error_propogation_update import error_propogation
# import sys
# sys.path.append('/home/gli12/project/PhyCRNet/Heat')  # Add the path to the parent folder containing example_folder
# from GDA.adamgda import AdamGDA

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)


# define the finite difference kernels
# center difference
lapl_ops = [[[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]]]

# partial difference with forward and backward method
partial_l = [[[[-3/2, 2,-1/2],
               [0, 0, 0],
               [0, 0, 0]]]]
partial_u = [[[[-3/2, 0, 0],
               [2, 0, 0],
               [-1/2, 0, 0]]]]
partial_r = [[[[1/2, -2, 3/2],
               [0, 0, 0],
               [0, 0, 0]]]]
partial_b = [[[[1/2, 0, 0],
               [-2, 0, 0],
               [3/2, 0, 0]]]]
partial_x = [[[[0, 0, 0],
               [-1/2, 0, 1/2],
               [0, 0, 0]]]] 

partial_y = [[[[0, -1/2, 0],
               [0, 0, 0],
               [0, 1/2, 0]]]] 
#weigths of different losses
w = torch.randn(4, 1, requires_grad=True).to(device)
nn.init.xavier_normal_(w)

# specific parameters for Diffusion_reaction equation 
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1 #0.5
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
     
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):

        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='zeros')

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='zeros')

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='zeros')

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='zeros')

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')       

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):
        return (Variable(prev_state[0]).to(device), Variable(prev_state[1]).to(device))


class encoder_block(nn.Module):
    ''' encoder with CNN '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):
        
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels, 
            self.hidden_channels, self.input_kernel_size, self.input_stride, 
            self.input_padding, bias=True, padding_mode='zeros')) # accerate training stability and speed

        self.act = nn.ReLU()

        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


class PhyCRNet(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''
    def __init__(self, input_channels, hidden_channels, 
        input_kernel_size, input_stride, input_padding, dt, 
        num_layers, upscale_factor, step=1, effective_step=[1]):

        super(PhyCRNet, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        # encoder - downsampling  
        for i in range(self.num_encoder):
            name = 'encoder{}'.format(i)
            cell = encoder_block(
                input_channels = self.input_channels[i], 
                hidden_channels = self.hidden_channels[i], 
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])

            setattr(self, name, cell) 
            self._all_layers.append(cell)            
            
        # ConvLSTM
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                input_channels = self.input_channels[i],
                hidden_channels = self.hidden_channels[i],
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])
        
            setattr(self, name, cell)
            self._all_layers.append(cell)  

        # output layer
        self.output_layer = nn.Conv2d(2, 2, kernel_size = 5, stride = 1, 
                                      padding=2, padding_mode='zeros')

        # pixelshuffle - upscale
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)   

        # initialize weights
        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, initial_state, x):
        
        self.initial_state = initial_state
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            xt = x

            # encoder
            for i in range(self.num_encoder):
                name = 'encoder{}'.format(i)
                x = getattr(self, name)(x)
                
            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state = self.initial_state[i - self.num_encoder])  
                    internal_state.append((h,c))
                
                # one-step forward
                (h, c) = internal_state[i - self.num_encoder]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)                               

            # output
            x = self.pixelshuffle(x)
            x = self.output_layer(x)

            # residual connection
            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()
                
            if step in self.effective_step:
                outputs.append(x)                

        return outputs, second_last_state

class weightloss(nn.Module):
    def __init__(self, num):
        super(weightloss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)   # 1 is strides

        # Fixed gradient operator, so that parameter in this layer will not be updated
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (1.0/70), dx = (2.0/40)):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()
        
        # spatial derivative operator with 3*3 kernel
        self.laplaces = Conv2dDerivative(
            DerFilter = lapl_ops,
            resol = (dx**2),
            kernel_size = 3,
            name = 'laplace_operator').to(device)

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            kernel_size = 3,
            name = 'partial_t').to(device)
        
        # left BCs derivative operator
        self.dl = Conv2dDerivative(
            DerFilter = partial_l,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').to(device)
        
        # right BCs derivative operator
        self.dr = Conv2dDerivative(
            DerFilter = partial_r,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').to(device)
        
        # up BCs derivative operator
        self.du = Conv2dDerivative(
            DerFilter = partial_u,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').to(device)
        
        # left BCs derivative operator
        self.db = Conv2dDerivative(
            DerFilter = partial_b,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').to(device)
        
        self.dx = Conv2dDerivative(
            DerFilter = partial_x,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dx_operator').to(device)

        self.dy = Conv2dDerivative(
            DerFilter = partial_y,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').to(device)
    
    def get_heat_loss(self, output, u0, all_img, type):

        mse_loss = nn.MSELoss()

        # temporal derivative - u
        u = output[:21, 0:1, :, :,]  #  training time is 0-1 seconds
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny,1,lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # laplace
        laplace_u = self.laplaces(u[1:-1, :, :, :])   #    (output[1:-1, :, :, :])

        # temporal derivative - v
        v = output[:21, 1:2, :, :,]  #  training time is 0-1 seconds
        lent = v.shape[0]
        lenx = v.shape[3]
        leny = v.shape[2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny,1,lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent-2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # laplace
        laplace_v = self.laplaces(v[1:-1, :, :, :])   #    (output[1:-1, :, :, :])

        u_x = self.dx(u[:, 0:1, :, :])
        u_y = self.dy(u[:, 0:1, :, :])
        v_x = self.dx(v[:, 0:1, :, :])
        v_y = self.dy(v[:, 0:1, :, :])

        # x_arr, y_arr --- convolutional layer
        padding_lu = (0, 2, 0, 2)
        output_pad_lu = F.pad(u, padding_lu, value=0)
        padding_ru = (0, 2, 0, 0)
        output_pad_ru = F.pad(u, padding_ru, value=0)
        padding_bu = (0, 0, 0, 2)
        output_pad_bu = F.pad(u, padding_bu, value=0)
        u_l = self.dl(output_pad_lu)[:,:,:,0]
        u_r = self.dr(output_pad_bu)[:,:,:,-1]
        u_u = self.du(output_pad_lu)[:,:,0,:,]
        u_b = self.db(output_pad_ru)[:,:,-1,:,]

        # x_arr, y_arr --- convolutional layer
        padding_lv = (0, 2, 0, 2)
        output_pad_lv = F.pad(v, padding_lv, value=0)
        padding_rv = (0, 2, 0, 0)
        output_pad_rv = F.pad(v, padding_rv, value=0)
        padding_bv = (0, 0, 0, 2)
        output_pad_bv = F.pad(v, padding_bv, value=0)
        v_l = self.dl(output_pad_lv)[:,:,:,0]
        v_r = self.dr(output_pad_bv)[:,:,:,-1]
        v_u = self.du(output_pad_lv)[:,:,0,:,]
        v_b = self.db(output_pad_rv)[:,:,-1,:,]

        if type == 'phy':

            fu = u_t[:, :, 1:-1, 1:-1] - 0.001*laplace_u - (u - u*u*u - v)[1:-1, :, 1:-1, 1:-1] + 0.005
            fv = v_t[:, :, 1:-1, 1:-1] - 0.005*laplace_v - (u-v)[1:-1, :, 1:-1, 1:-1]
            f = mse_loss(fu, torch.zeros_like(fu).to(device)) + mse_loss(fv, torch.zeros_like(fv).to(device))    

        elif type == 'BC':
            x = torch.linspace(0, 1, 32)
            y = torch.linspace(0, 1, 32)
            " mixed BCs"
            fu_x1 = u[1:-1,:,:,0] # Left BC u_l[1:-1]#
            fu_x2 = u[1:-1,:,:,-1] # Right BC u_r[1:-1]#
            fu_y1 = u_b[1:-1]
            fu_y2 = u_u[1:-1]

            fv_x1 = v_l[1:-1] 
            fv_x2 = v_r[1:-1]
            fv_y1 = v[1:-1,:,0,:] # Bottom BC v_u[1:-1]#
            fv_y2 = v[1:-1,:,-1,:] # Top BC v_b[1:-1]#

            x1 = 0.5*torch.cos(8*torch.pi*y).cuda()
            x2 = 0.5*torch.cos(8*torch.pi*y).cuda()
            y1 = 0.5*torch.cos(8*torch.pi*x).cuda()
            y2 = 0.5*torch.cos(8*torch.pi*x).cuda()
            x1_d = x1.unsqueeze(0).unsqueeze(0).repeat(fu_x1.shape[0], 1, 1)
            x2_d = x2.unsqueeze(0).unsqueeze(0).repeat(fu_x2.shape[0], 1, 1)
            y1_d = y1.unsqueeze(0).unsqueeze(0).repeat(fv_y1.shape[0], 1, 1)
            y2_d = y2.unsqueeze(0).unsqueeze(0).repeat(fv_y1.shape[0], 1, 1)

            f1 = ((fu_x1-x1_d)**2 + (fu_x2-x2_d)**2 + (fu_y1)**2 + (fu_y2)**2).mean()/(fu_x1.shape[-1])
            f2 = ((fv_x1)**2 + (fv_x2)**2 + (fv_y1-y1_d)**2 + (fv_y2-y2_d)**2).mean()/(fv_x1.shape[-1])
            f = f1 + f2 #torch.tensor(0)#
            # f = torch.mean((torch.square(fu_x1-x1_d) + torch.square(fu_x2-x2_d) \
            #     + torch.square(fu_y1) + torch.square(fu_y2) + torch.square(fv_x1) + torch.square(fv_x2) \
            #     + torch.square(fv_y1-y1_d) + torch.square(fv_y2-y2_d)))/(fu_x1.shape[-1])  


        elif type == 'IC': # something wrong, is this hard constrin of IC?
            x = torch.linspace(0, 1, 32)
            y = torch.linspace(0, 1, 32)
            x, y = torch.meshgrid(x, y)
            # f = output[1]-0.5*(torch.sin(4*torch.pi*x).to(device)+torch.sin(4*torch.pi*y).to(device))
            fu = (u[0].squeeze()-(0.5*(torch.sin(8*torch.pi*x).to(device)+torch.cos(8*torch.pi*y).to(device))).T) # Add .T because meshgrid default left top as 0,0
            fv = (v[0].squeeze()-(0.5*(torch.cos(8*torch.pi*x).to(device)+torch.sin(8*torch.pi*y).to(device))).T) # but the image is left bottom as 0,0
            f = nn.MSELoss()(fu, torch.zeros_like(fu).cuda()) + nn.MSELoss()(fv, torch.zeros_like(fv).cuda())

        elif type == 'data':
            output_s1 = output[:21].flatten()
            all_img_s1 = all_img.squeeze()[:21].flatten()

            # sampling 20% of the data
            torch.manual_seed(42)
            randomlist = torch.randperm(output_s1.size()[0]).tolist()[:int(0.2*torch.numel(output_s1))]
            # randomlist = torch.randperm(10).tolist()[:7]
            # print('randomlist----------:',randomlist)
            output_s1 = output_s1[randomlist]#[torch.randperm(output_s1.size()[0])][:int(0.05*torch.numel(output_s1))]
            all_img_s1 = all_img_s1[randomlist]#[torch.randperm(all_img_s1.size()[0])][:int(0.05*torch.numel(all_img_s1))]
            f = mse_loss(all_img_s1, output_s1)

            #f = mse_loss(all_img_s1[:,0,:,:,], output_s1[:,0,:,:,]) + mse_loss(all_img_s1[:,1,:,:,], output_s1[:,1,:,:,])
        return f


def compute_loss(output, u0, all_img, loss_func):
    ''' calculate the phycis loss '''
    
    # get physics loss
    mse_loss = nn.MSELoss()
    f_phy = loss_func.get_heat_loss(output, u0, all_img, type = 'phy')
    f_bc = loss_func.get_heat_loss(output, u0, all_img, type = 'BC')
    f_ic = loss_func.get_heat_loss(output, u0, all_img, type = 'IC')
    if all_img is not None:
        f_data = loss_func.get_heat_loss(output, u0, all_img, type = 'data')
    else:
        f_data = 0
        f_data = torch.tensor(float(f_data)).to(device)
    #f_phy = mse_loss(f_phy, torch.zeros_like(f_phy).to(device))
    f = f_phy+f_bc+f_data+f_ic

    return f, f_phy/20, f_bc/50, f_ic, f_data

def train(model, weight_model, input, all_img, initial_state, n_iters, time_batch_size, learning_rate, 
          dt, dx, save_path, pre_model_save_path, num_time_batch):

    train_loss_list = []
    phy_loss_list = []
    bc_loss_list = []
    ic_loss_list = []
    data_loss_list = []
    second_last_state = []
    prev_output = []

    batch_loss = 0.0
    phy_loss = 0.0
    bc_loss = 0.0
    ic_loss = 0.0
    best_loss = 1e4
    loss = 0.0

    # load previous model
    optimizer = optim.Adam([{'params':model.parameters()},
                            {'params':weight_model.parameters()}], lr=learning_rate) 
    #optimizer = AdamGDA(list(model.parameters()) + list([w]), dim_max=4, lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)

    if pre_model_save_path is not None:  
        model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, 
        pre_model_save_path)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    loss_func = loss_generator(dt, dx)
        
    for epoch in range(n_iters):
        # input: [t,b,c,h,w]
        optimizer.zero_grad()
        batch_loss = 0 
        phy_loss = 0.0
        bc_loss = 0.0
        ic_loss = 0.0
        data_loss = 0.0
        
        for time_batch_id in range(num_time_batch):  # num_time_batch = 1
            # update the first input for each time batch
            if time_batch_id == 0:
                hidden_state = initial_state
                u0 = input # (1, 2, 128, 128)
            else:
                hidden_state = state_detached
                u0 = prev_output[-2:-1].detach() # second last output

            # output is a list
            output, second_last_state = model(hidden_state, u0)
            # [t, c, height (Y), width (X)]
            output = torch.cat(tuple(output), dim=0) # (1001, 2, 128, 128)
            # concatenate the initial state to the output for central diff
            output = torch.cat((u0.to(device), output), dim=0) # (1002, 2, 128, 128)
            # get loss
            f, f_phy, f_bc, f_ic, f_data = compute_loss(output, u0, all_img, loss_func)  # output - (40, 56)
            # loss = weight_model(f_phy, f_bc, f_ic, f_data)
            
            #loss =  (torch.square(f_phy)+ torch.square(f_bc)+ torch.square(f_ic)+ torch.square(f_data))/(f_phy+ f_bc+ f_ic+ f_data)
            loss =  (f_data**2 + f_phy**2  + f_bc**2 )/( f_data + f_phy + f_bc )
            loss.backward(retain_graph=True)
            batch_loss += loss.item() # add loss of every batch training
            phy_loss += f_phy.item()
            bc_loss += f_bc.item()
            ic_loss += f_ic.item()
            if type(f_data) is int:
                data = 0 
            else:
                data_loss += f_data.item()
            # update the state and output for next batch
            prev_output = output
            state_detached = []
            for i in range(len(second_last_state)):
                (h, c) = second_last_state[i]
                state_detached.append((h.detach(), c.detach())) # hidden state

        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        if epoch%100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
            epoch, n_iters, loss, f_phy.item(), f_bc.item(), f_ic.item(), f_data.item()))
        train_loss_list.append(batch_loss)
        phy_loss_list.append(phy_loss)
        bc_loss_list.append(bc_loss)
        ic_loss_list.append(ic_loss)
        data_loss_list.append(data_loss)
        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss
    return train_loss_list, phy_loss_list, bc_loss_list, ic_loss_list, data_loss_list

def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)

def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    if save_dir is None:# if torch.load(save_dir) is None:
        return
    else:
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('Pretrained model loaded!')
        return model, optimizer, scheduler

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

if __name__ == '__main__':

    # grid parameters
    time_steps = 41 
    dt = 0.05
    dx = 1/32 

    fig_save_path = './Diffusion_reaction/'
    data = pd.read_csv('./Data/Diffusion_reaction_update_100%.csv', skiprows=0)  # read comsol generated data
    data_first = np.array(data)[:,0:4].T
    imageu0, imagev0 = data_to_img_new(data_first)
    imageuv0 = np.array([imageu0, imagev0])[np.newaxis, :]

    # get all image
    all_imgs = []
    data_sample = pd.read_csv('./Data/Diffusion_reaction_update_100%.csv', skiprows=0)  # read comsol generated data
    for i in range(1, time_steps+1):  # so that there is 22 time steps
        all_img = np.array(data_sample)[:,[0,1,2*i, 2*i+1]]
        all_img = all_img.T 
        imgu, imgv =  data_to_img_new(all_img)
        all_imgs.append([imgu, imgv])
    all_imgs = np.array(all_imgs)[np.newaxis, :]
    all_imgs = torch.tensor(all_imgs, dtype=torch.float32).to(device)   # None
    #all_imgs = None
    
    input = torch.tensor(imageuv0, dtype=torch.float32).to(device)

    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (torch.randn(1, 128, 4, 4), torch.randn(1, 128, 4, 4))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))


    ################# build the model #####################
    time_batch_size = 40 # control the time step of output
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int((time_batch_size+1) / time_batch_size)
    n_iters_adam =20000
    lr_adam = 5e-4
    pre_model_save_path =None #'/home/gli12/project/PhyCRNet/phyCRNet_AM/checkpoint1000-1.pt'
    model_save_path =  './Diffusion_reaction/phycrnet_diffusion_reaction.pt' # '/home/gli12/project/PhyCRNet/phyCRNet_AM/checkpoint_heat.pt'

    model = PhyCRNet(
        input_channels = 2, 
        hidden_channels = [8, 32, 128, 128], 
        input_kernel_size = [4, 4, 4, 3], 
        input_stride = [2, 2, 2, 1], 
        input_padding = [1, 1, 1, 1],  
        dt = dt,
        num_layers = [3, 1],
        upscale_factor = 8, 
        step = steps,     
        effective_step = effective_step ).to(device) 
    weight_model = weightloss(num=4)
    #summary(model)
    start = time.time()

    #train_loss, phy_loss, bc_loss, ic_loss, data_loss = train(model, weight_model, input, all_imgs, initial_state, n_iters_adam, time_batch_size, 
     #lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch)
    end = time.time()
    #np.save('train_loss', train_loss)  
    print('The training time is: ', (end-start))

    ########### model inference (testing the trained model) ##################
    time_batch_size_load = 40 # control the time step of output
    steps_load = time_batch_size_load + 1
    num_time_batch = int(time_steps / time_batch_size_load)
    effective_step = list(range(0, steps_load))  
    
    model = PhyCRNet(
        input_channels = 2, 
        hidden_channels = [8, 32, 128, 128],
        input_kernel_size = [4, 4, 4, 3], 
        input_stride = [2, 2, 2, 1], 
        input_padding = [1, 1, 1, 1],  
        dt = dt,
        num_layers = [3, 1],
        upscale_factor = 8,
        step = steps_load, 
        effective_step = effective_step).to(device)
    
    # plot train loss
    # plt.figure()
    # plt.plot(phy_loss, label = 'phy_loss')
    # plt.plot(train_loss, label = 'train loss')
    # plt.plot(bc_loss, label = 'bc_loss')
    # plt.plot(ic_loss, label = 'ic loss')
    # plt.plot(data_loss, label = 'data loss')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig(fig_save_path + 'train loss.png', dpi = 300)
    # plt.close()

    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path) 
    output, _ = model(initial_state, input)  # shape of putput,  len - 21

    # shape: [t, c, h, w] 
    output = torch.cat(tuple(output), dim=0)  # make the output list as a tu]ensor
    output = torch.cat((input.to(device), output), dim=0)  # insert the input to output, input will be in the first of dim 0.
    error_propogation(output)

    fig_save_path = './Diffusion_reaction/' #'/home/gli12/project/PhyCRNet/phyCRNet_AM/result_figure/'
    data1 = np.array(data)[:, [0,1,42,43]].T   # t=1.0 seconds
    data0_5 = np.array(data)[:, [0,1,22,23]].T  # t=0.5 seconds
    data1_5 = np.array(data)[:, [0,1,62,63]].T  # t=1.5 seconds
    data2 = np.array(data)[:, [0,1,82,83]].T  # t=2.0 seconds
    data0 = np.array(data)[:, [0,1,2,3]].T 
    ZU_truth0, ZV_truth0 = data_to_img_new(data0)
    ZU_truth1, ZV_truth1 = data_to_img_new(data1)
    ZU_truth0_5, ZV_truth0_5 = data_to_img_new(data0_5)
    ZU_truth1_5, ZV_truth1_5 = data_to_img_new(data1_5)
    ZU_truth2, ZV_truth2 = data_to_img_new(data2)

    output = output.cpu().detach().numpy()

    u_test_err1, v_test_err1 = np.mean((output[21,0].squeeze()-ZU_truth1)**2), np.mean((output[21,1].squeeze()-ZV_truth1)**2) # t=1.0 seconds
    u_test_err0, v_test_err0 = np.mean((output[0,0].squeeze()-ZU_truth0)**2), np.mean((output[0,1].squeeze()-ZV_truth0)**2) # t=0.05 seconds
    u_test_err0_5, v_test_err0_5 = np.mean((output[11,0].squeeze()-ZU_truth0_5)**2), np.mean((output[11,1].squeeze()-ZV_truth0_5)**2) # t=0.5 seconds
    u_test_err1_5, v_test_err1_5 = np.mean((output[31,0].squeeze()-ZU_truth1_5)**2), np.mean((output[31,1].squeeze()-ZV_truth1_5)**2) # t=1.5 seconds
    u_test_err2, v_test_err2 = np.mean((output[-1,0].squeeze()-ZU_truth2)**2), np.mean((output[-1,1].squeeze()-ZV_truth2)**2) # t=2.0 seconds
    
    print('U ,V test_error t=0: ', np.sqrt(u_test_err0), np.sqrt(v_test_err0))
    print('U ,V test_error t=0.5:', np.sqrt(u_test_err0_5), np.sqrt(v_test_err0_5))
    print('U ,V test_error t=1: ', np.sqrt(u_test_err1), np.sqrt(v_test_err1))

    # from extropolation
    # input_extro = truth1[np.newaxis, np.newaxis,:]
    # input_extro = torch.tensor(input_extro, dtype=torch.float32).to(device)
    # output_extro, _ = model(initial_state, input_extro)
    # output_extro = torch.cat(tuple(output_extro), dim=0)  # make the output list as a tu]ensor
    # output_extro = torch.cat((input_extro.to(device), output_extro), dim=0)
    # output_extro = output_extro.cpu().detach().numpy()
    # test_err1_5 = np.mean((output_extro[11].squeeze()-truth1_5)**2)
    # test_err2 = np.mean((output_extro[-1].squeeze()-truth2)**2) 
    #test_err_interior = np.mean((output[-1][:,1:-1,1:-1].squeeze()-truth[1:-1,1:-1])**2) # MSE
    print('U ,V test_error t=1.5:', np.sqrt(u_test_err1_5), np.sqrt(v_test_err1_5))
    print('U ,V test_error t=2: ', np.sqrt(u_test_err2), np.sqrt(v_test_err2))


    matplotlib.rcParams.update({'font.size': 20})
    # plot image at time t
    x = np.linspace(0, 31, 32)
    y = np.linspace(0, 31, 32)
    L_xsample = 31
    X, Y = np.meshgrid(x, y)
    X = X.flatten()/L_xsample
    Y = Y.flatten()/L_xsample
    ZU1, ZV1 = output[21][0].flatten(), output[21][1].flatten()
    ZU0, ZV0 = output[0][0].flatten(), output[0][1].flatten()
    ZU0_5, ZV0_5 = output[11][0].flatten(), output[11][1].flatten()
    ZU1_5, ZV1_5 = output[31][0].flatten(), output[31][1].flatten()
    ZU2, ZV2 = output[-1][0].flatten(), output[-1][1].flatten()
    Mymodel = 'PhyCRNet_Diffusion_reaction'
 
    matplotlib.rcParams.update({'font.size': 16})
    plotfigure_uv(X, Y, ZU0, ZV0, Mymodel, 0.0, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU0_5, ZV0_5, Mymodel, 0.5, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU1, ZV1, Mymodel, 1.0, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU1_5, ZV1_5, Mymodel, 1.5, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU2, ZV2, Mymodel, 2.0, './Diffusion_reaction/Results/')

    plotfigure_uv(X, Y, ZU_truth0.flatten(), ZV_truth0.flatten(), 'Truth-Diffusion_reaction', 0.0, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU_truth0_5.flatten(), ZV_truth0_5.flatten(), 'Truth-Diffusion_reaction', 0.5, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU_truth1.flatten(), ZV_truth1.flatten(), 'Truth-Diffusion_reaction', 1.0, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU_truth1_5.flatten(), ZV_truth1_5.flatten(), 'Truth-Diffusion_reaction', 1.5, './Diffusion_reaction/Results/')
    plotfigure_uv(X, Y, ZU_truth2.flatten(), ZV_truth2.flatten(), 'Truth-Diffusion_reaction', 2.0, './Diffusion_reaction/Results/')

    print('done')




    