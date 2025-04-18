'''PhyCRNet for solving Allen-Cahn PDEs'''

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
sys.path.append('/home/gli12/project/PhyCRNet/Heat')  # Add the path to the parent folder containing example_folder
from GDA.adamgda import AdamGDA

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

#weigths of different losses
w = torch.randn(4, 1, requires_grad=True).cuda()
nn.init.xavier_normal_(w)

# specific parameters for burgers equation 
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
        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())


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
        self.output_layer = nn.Conv2d(2, 1, kernel_size = 5, stride = 1, 
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
            name = 'laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            kernel_size = 3,
            name = 'partial_t').cuda()
        
        # left BCs derivative operator
        self.dl = Conv2dDerivative(
            DerFilter = partial_l,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').cuda()
        
        # right BCs derivative operator
        self.dr = Conv2dDerivative(
            DerFilter = partial_r,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').cuda()
        
        # up BCs derivative operator
        self.du = Conv2dDerivative(
            DerFilter = partial_u,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').cuda()
        
        # left BCs derivative operator
        self.db = Conv2dDerivative(
            DerFilter = partial_b,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').cuda()
    
    def get_heat_loss(self, output, u0, all_img, type):

        # temporal derivative - u
        u = output  #[:, 0:1, 1:-1, 1:-1]   
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny,1,lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # laplace
        laplace_u = self.laplaces(output[1:-1, :, :, :])   #    (output[1:-1, :, :, :])

        # x_arr, y_arr --- convolutional layer
        padding_lu = (0, 2, 0, 2)
        output_pad_lu = F.pad(output, padding_lu, value=0)
        padding_r = (0, 2, 0, 0)
        output_pad_r = F.pad(output, padding_r, value=0)
        padding_b = (0, 0, 0, 2)
        output_pad_b = F.pad(output, padding_b, value=0)
        u_l = self.dl(output_pad_lu)[:,:,:,0]
        u_r = self.dr(output_pad_b)[:,:,:,-1]
        u_u = self.du(output_pad_lu)[:,:,0,:,]
        u_b = self.db(output_pad_r)[:,:,-1,:,]

        # x_arr, y_arr ---  direct finite difference, [t, c, h, w] 
        # u_l = -1/2*(output[:,:,:,0:3][:,:,:,-1])+2*(output[:,:,:,0:3][:,:,:,1])-(3/2)*(output[:,:,:,0:3][:,:,:,0])
        # u_r = 3/2*(output[:,:,:,-4:-1][:,:,:,-1])-2*(output[:,:,:,-4:-1][:,:,:,1])+(1/2)*(output[:,:,:,-4:-1][:,:,:,0])
        # u_b = 3/2*(output[:,:,-4:-1,:][:,:,-1,:])-2*(output[:,:,-4:-1,:][:,:,1,:])+(1/2)*(output[:,:,-4:-1,:][:,:,0,:])
        # u_u = 3/2*(output[:,:,0:3,:][:,:,-1,:])-2*(output[:,:,0:3,:][:,:,1,:])+(1/2)*(output[:,:,0:3,:][:,:,0,:])

        # second derivative on boundary
        u_ll = ((output[:,:,:,0:3][:,:,:,-1])-2*(output[:,:,:,0:3][:,:,:,1])-(output[:,:,:,0:3][:,:,:,0]))*1024
        u_rr = (output[:,:,:,-4:-1][:,:,:,-1])-2*(output[:,:,:,-4:-1][:,:,:,1])+(output[:,:,:,-4:-1][:,:,:,0])*1024
        u_bb = ((output[:,:,-4:-1,:][:,:,-1,:])-2*(output[:,:,-4:-1,:][:,:,1,:])+(output[:,:,-4:-1,:][:,:,0,:]))*1024
        u_uu = ((output[:,:,0:3,:][:,:,-1,:])-2*(output[:,:,0:3,:][:,:,1,:])+(output[:,:,0:3,:][:,:,0,:]))*1024

        if type == 'phy':
            # laplace_u = F.pad(laplace_u, (1, 1, 1, 1), value=0)  # padding with 0 first
            # laplace_u[:, :, 0, :] = u_uu[1:-1]
            # laplace_u[:, :, -1, :] = u_bb[1:-1]
            # laplace_u[:, :, :, 0] = u_ll[1:-1]
            # laplace_u[:, :, :, -1] = u_rr[1:-1]

            f = u_t[:, :, 1:-1, 1:-1] - 0.001*laplace_u - (u - u*u*u)[1:-1, :, 1:-1, 1:-1]
        elif type == 'BC':
            f_x1 = u[1:-1,:,:,0] - u[1:-1,:,:,-1]
            f_x2 = u_l[1:-1] - u_r[1:-1]
            f_y1 = u[1:-1,:,0,] - u[1:-1,:,-1,]
            f_y2 = u_b[1:-1] - u_u[1:-1]
            f = (torch.square(f_x1) + torch.square(f_x2) \
                + torch.square(f_y1) + torch.square(f_y2))/f_x1.shape[-1]#*f_x1.shape[-1]
        elif type == 'IC':
            x = torch.linspace(0, 1, 32)
            y = torch.linspace(0, 1, 32)
            x, y = torch.meshgrid(x, y)
            f = output[1]-0.5*(torch.sin(4*torch.pi*x).cuda()+torch.sin(4*torch.pi*y).cuda())
            f = nn.MSELoss()(f, torch.zeros_like(f).cuda())
        elif type == 'data':
            sample_index = [0, 6, 13, 19, 26, -1]
            #output = output[:, :, ::6, ::6]
            output_s = output[:, :, sample_index, :,]
            output_s1 = output_s[:, :, :, sample_index]
            #output_s1 = output
            f = nn.MSELoss()(all_img, output_s1[0:-1])
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
        f_data = torch.tensor(float(f_data)).cuda()
    f_phy = mse_loss(f_phy, torch.zeros_like(f_phy).cuda())
    f_bc = torch.mean(f_bc)
    f = f_phy+f_bc+f_data+f_ic*100
    # loss = mse_loss(f, torch.zeros_like(f).cuda())

    return f, f_phy, f_bc, f_ic, f_data
    # return loss, f_phy, f_bc, f_ic, f_data


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
            output = torch.cat((u0.cuda(), output), dim=0) # (1002, 2, 128, 128)
            # get loss
            f, f_phy, f_bc, f_ic, f_data = compute_loss(output, u0, all_img, loss_func)  # output - (40, 56)
            # loss = weight_model(f_phy, f_bc, f_ic, f_data)
            
            #loss =  (torch.square(f_phy)+ torch.square(f_bc)+ torch.square(f_ic)+ torch.square(f_data))/(f_phy+ f_bc+ f_ic+ f_data)
            loss =  (f_phy**2+ f_bc**2+ f_ic**2)/(f_phy+ f_bc+ f_ic)
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
        if epoch%10 == 0:
            print('[%d/%d %d%%] loss: %.10f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0), batch_loss))
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

def data_to_image(data, length, fig_save_path, resize=True):
    x_coords = data[0]
    y_coords = data[1]
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    grid = np.zeros((length, length))
    for i in range(len(data[2])):
        x_index = np.where(unique_x == x_coords[i])[0][0]
        y_index = np.where(unique_y == y_coords[i])[0][0]
        grid[y_index, x_index] = data[2][i]

    # save the figure
    if resize:
        grid = cv2.resize(grid, (32, 32), interpolation=cv2.INTER_AREA)
        # grid = cv2.resize(grid, (26, 26), interpolation=cv2.INTER_AREA)
    plt.figure()
    plt.imshow(grid, cmap='viridis') # [::6, ::6]
    plt.colorbar()
    plt.savefig(fig_save_path + 'truth_allen.png', dpi = 300)
    plt.close()
    return grid

def output_to_img(img_output, name):

    # save the figure
    for i in range(img_output.shape[0]):
        if i%10==0:
            plt.figure()
            plt.imshow(img_output[i,:,:,].squeeze(), cmap='viridis')
            plt.colorbar()
            plt.savefig(fig_save_path + str(i)+'_'+name+'.png', dpi = 300)
            plt.close()
    return 

def full_datato_image(data, length, fig_save_path, resize=True):
    x_coords = [i/25 for i in range(26)]
    y_coords = [i/25 for i in range(26)]
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    grid = np.zeros((length, length))
    data = data.astype(float)
    for i in range(length):
        for j in range(length):
            x_index = np.where(unique_x == x_coords[i])[0][0]
            y_index = np.where(unique_y == y_coords[i])[0][0]
            T = data[(data[:, 1] == x_coords[i]) & (data[:, 2] == y_coords[j]), 0]
            grid[y_index, x_index] = T[0]

    # save the figure
    if resize:
        grid = cv2.resize(grid, (32, 32), interpolation=cv2.INTER_AREA)
        # grid = cv2.resize(grid, (26, 26), interpolation=cv2.INTER_AREA)
    plt.figure()
    plt.imshow(grid[::6, ::6], cmap='viridis')
    plt.colorbar()
    plt.savefig(fig_save_path + 'heat_First_full.png', dpi = 300)
    plt.close()
    return grid

if __name__ == '__main__':
    # path = '/home/gli12/project/PhyCRNet/CSV/Heat/heat_Eva_FEM_100.csv'
    # desired_columns = [col for col in pd.read_csv(path, nrows=0).columns if 't=0' in col]
    #data_t0 = pd.read_csv(path, header=None)

    ######### download/read the ground truth/training data ############
    fig_save_path = '/home/gli12/project/PhyCRNet/phyCRNet_AM/result_figure/'

    #data = np.array(data_t0.iloc[1:,2:6])
    #full_datato_image(data, 26, fig_save_path, resize=True)

    # data_first = pd.read_csv('/home/gli12/project/PhyCRNet/CSV/Heat/Heat_Eva_FEM_First.csv', header=None)
    all_img1 = pd.read_csv('/home/gli12/project/PhyCRNet/CSV/Training_AllenCahn_LF_FEM.csv', header=None)
    # generate image0
    n_points = 32
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    x, y = np.meshgrid(x, y)
    image0 = (0.5 * (np.sin(4 * np.pi * x) + np.sin(4 * np.pi * y)))[np.newaxis, np.newaxis,:]

    #image0 = data_to_image(data_first, 26, fig_save_path, resize=True)[np.newaxis, np.newaxis,:]
    # all_imgs = []
    # for i in range(21):
    #     all_img = np.array(all_img1)[:,1:4]
    #     all_img = all_img.T #reshape(all_img.shape[1],all_img.shape[0])
    #     img =  data_to_image(all_img[:,36*i:36*(i+1)], 6, fig_save_path, resize=False)[np.newaxis, :]
    #     all_imgs.append(img)
    # all_imgs = np.array(all_imgs)
    # all_imgs = torch.tensor(all_imgs, dtype=torch.float32).cuda()   # None
    all_imgs = None
    
    input = torch.tensor(image0, dtype=torch.float32).cuda()

    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (torch.randn(1, 128, 4, 4), torch.randn(1, 128, 4, 4))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))
    
    # grid parameters
    time_steps = 21 #
    dt = 0.05 # 
    dx = 1/32 # 1/32

    ################# build the model #####################
    time_batch_size = 20 # 1000
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters_adam =20000
    lr_adam = 1e-3 #1e-3 
    pre_model_save_path =None #'/home/gli12/project/PhyCRNet/phyCRNet_AM/checkpoint1000-1.pt'
    model_save_path =  '/home/gli12/project/PhyCRNet/phyCRNet_AM/checkpoint1000-1_allen.pt' # '/home/gli12/project/PhyCRNet/phyCRNet_AM/checkpoint_heat.pt'
    fig_save_path = '/home/gli12/project/PhyCRNet/phyCRNet_AM/result_figure/'  

    model = PhyCRNet(
        input_channels = 1, 
        hidden_channels = [8, 32, 128, 128], 
        input_kernel_size = [4, 4, 4, 3], #[4, 4, 4, 3], 
        input_stride = [2, 2, 2, 1], # [2, 2, 2, 1]
        input_padding = [1, 1, 1, 1],  
        dt = dt,
        num_layers = [3, 1],
        upscale_factor = 8, #8
        step = steps, 
        effective_step = effective_step).cuda()
    weight_model = weightloss(num=4)
    start = time.time()

    train_loss, phy_loss, bc_loss, ic_loss, data_loss = train(model, weight_model, input, all_imgs, initial_state, n_iters_adam, time_batch_size, 
         lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch)
    end = time.time()
    # np.save('train_loss', train_loss)  
    print('The training time is: ', (end-start))

    ########### model inference (testing the trained model) ##################
    time_batch_size_load = 20
    steps_load = time_batch_size_load + 1
    num_time_batch = int(time_steps / time_batch_size_load)
    effective_step = list(range(0, steps_load))  
    
    model = PhyCRNet(
        input_channels = 1, 
        hidden_channels = [8, 32, 128, 128],
        input_kernel_size = [4, 4, 4, 3], 
        input_stride = [2, 2, 2, 1], 
        input_padding = [1, 1, 1, 1],  
        dt = dt,
        num_layers = [3, 1],
        upscale_factor = 8,
        step = steps_load, 
        effective_step = effective_step).cuda()
    
    # plot train loss
    # plt.figure()
    # plt.plot(phy_loss, label = 'phy_loss')
    # plt.plot(train_loss, label = 'train loss')
    # plt.plot(bc_loss, label = 'bc_loss')
    # plt.plot(ic_loss, label = 'ic loss')
    # plt.plot(data_loss, label = 'data loss')
    # plt.yscale('log') 
    # plt.legend()
    # plt.savefig(fig_save_path + 'train loss allen.png', dpi = 300)

    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path) 
    output, _ = model(initial_state, input)  # shape of putput,  len - 21

    # shape: [t, c, h, w] 
    output = torch.cat(tuple(output), dim=0)  # make the output list as a tu]ensor
    output = torch.cat((input.cuda(), output), dim=0)  # insert the input to output, input will be in the first of dim 0.

    fig_save_path = '/home/gli12/project/PhyCRNet/phycrnet_allen/'
    data_last = pd.read_csv('/home/gli12/project/PhyCRNet/CSV/Training_AllenCahn_Eva_FEM.csv', header=None)

    data_last = data_last.iloc[-676:,1:] # using iloc to read the excel file
    data_last.columns = [0,1,2] # change the columns from [1,2,3] to [0,1,2]
    data_last.index = [i for i in range(676)] # change the rows from [27700:] to [0,1,2,...]
    
    image_last = data_to_image(data_last, 26, fig_save_path)
    truth = image_last
    truth0 = image0

    output = output.cpu().detach().numpy()
    image_out = output   # *std0 + mean0 
    output_to_img(image_out, name='output_heat_allen')

    test_err1 = np.mean((image_out[-1].squeeze()-truth)**2) 
    test_err0 = np.mean((image_out[1].squeeze()-truth0)**2)
    test_err_interior = np.mean((image_out[-1][:,1:-1,1:-1].squeeze()-truth[1:-1,1:-1])**2) # MSE
    test_err_1_2 = np.mean((image_out[10].squeeze()-truth)**2)
    print('test_error t=1:', np.sqrt(test_err1), test_err1)
    print('test_error t=0:', np.sqrt(test_err0), test_err0)
    #print('test_interior_error:', np.sqrt(test_err_interior), test_err_interior)

    # plot output last
    plt.figure()
    plt.imshow((image_out[-1].squeeze()), cmap='viridis')
    plt.colorbar()
    plt.savefig(fig_save_path + 'ouput_last_allen.png', dpi = 300)
    plt.close()

    # plot error
    plt.figure()
    plt.imshow((image_out[-1].squeeze()-truth), cmap='viridis')
    plt.colorbar()
    plt.savefig(fig_save_path + 'error_map_allen.png', dpi = 300)
    plt.close()

    matplotlib.rcParams.update({'font.size': 20})
    # plot image at time t
    x = np.linspace(0, 31, 32)
    y = np.linspace(0, 31, 32)
    L_xsample = 31
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = output[-1][-1].flatten()#.squeeze().cpu().detach().numpy().flatten()
    Z0 = output[1][-1].flatten()
    Z_truth = truth.flatten()
    image0 = image0.flatten()
    Mymodel = 'PhyCRNet'

    # plot u predict
    plt.tricontour(X/L_xsample, Y/L_xsample, Z, 25, cmap='jet')
    plt.tricontourf(X/L_xsample, Y/L_xsample, Z, 25, cmap='jet')
    cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    plt.plot(X/L_xsample, Y/L_xsample, 'ko', ms=3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.savefig('/home/gli12/project/PhyCRNet/phycrnet_allen/' + Mymodel + '-allen' + str(1) + '.png')
    plt.show()
    plt.close()

    # plot u truth
    plt.tricontour(X/L_xsample, Y/L_xsample, Z0, 15, cmap='jet')
    plt.tricontourf(X/L_xsample, Y/L_xsample, Z0, 15, cmap='jet')
    cbar=plt.colorbar()
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    plt.plot(X/L_xsample, Y/L_xsample, 'ko', ms=3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.savefig('/home/gli12/project/PhyCRNet/phycrnet_allen/' + Mymodel + '-allen' + str(0) + '.png')
    plt.show()
    plt.close()

    