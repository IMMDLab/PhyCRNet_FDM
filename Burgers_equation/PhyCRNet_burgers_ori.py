'''PhyCRNet for solving spatiotemporal PDEs'''

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
from torch.nn.utils import weight_norm
from torchinfo import summary

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
# define the high-order finite difference kernels
# lapl_op = [[[[    0,   0, -1/12,   0,     0],
#              [    0,   0,   4/3,   0,     0],
#              [-1/12, 4/3,    -5, 4/3, -1/12],
#              [    0,   0,   4/3,   0,     0],
#              [    0,   0, -1/12,   0,     0]]]]   # O(h^4)
lapl_op = [[[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]]]  # O(h^2)

# partial_y = [[[[0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0],
#                [1/12, -8/12, 0, 8/12, -1/12],
#                [0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0]]]] # O(h^4)

# partial_x = [[[[0, 0, 1/12, 0, 0],
#                [0, 0, -8/12, 0, 0],
#                [0, 0, 0, 0, 0],
#                [0, 0, 8/12, 0, 0],
#                [0, 0, -1/12, 0, 0]]]] # O(h^4)

partial_y = [[[[0, 0, 0],
               [-1/2, 0, 1/2],
               [0, 0, 0]]]] 

partial_x = [[[[0, -1/2, 0],
               [0, 0, 0],
               [0, 1/2, 0]]]] 

# generalized version
# def initialize_weights(module):
#     ''' starting from small initialized parameters '''
#     if isinstance(module, nn.Conv2d):
#         c = 0.1
#         module.weight.data.uniform_(-c*np.sqrt(1 / np.prod(module.weight.shape[:-1])),
#                                      c*np.sqrt(1 / np.prod(module.weight.shape[:-1])))
     
#     elif isinstance(module, nn.Linear):
#         module.bias.data.zero_()

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
            bias=True, padding_mode='circular')

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')       

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
        #return (Variable(prev_state[0]), Variable(prev_state[1]))


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
            self.input_padding, bias=True, padding_mode='circular')) # accerate training stability and speed

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

            setattr(self, name, cell) # 将指定对象的属性值设置为新值
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
                                      padding=2, padding_mode='circular')

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
                x = getattr(self, name)(x)   # 获取对象的属性值
                
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
            1, padding=0, bias=False)

        # Fixed gradient operator
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

    def __init__(self, dt = (10.0/200), dx = (20.0/128)):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        self.dx = Conv2dDerivative(
            DerFilter = partial_x,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dx_operator').cuda()

        self.dy = Conv2dDerivative(
            DerFilter = partial_y,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dy_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            kernel_size = 3,
            name = 'partial_t').cuda()

    def get_phy_Loss(self, output):

        output = output[:501]

        # spatial derivatives
        laplace_u = self.laplace(output[1:-1, 0:1, :, :])  # [t,c,h,w]
        laplace_v = self.laplace(output[1:-1, 1:2, :, :])

        u_x = self.dx(output[1:-1, 0:1, :, :])
        u_y = self.dy(output[1:-1, 0:1, :, :])
        v_x = self.dx(output[1:-1, 1:2, :, :])
        v_y = self.dy(output[1:-1, 1:2, :, :])

        # temporal derivative - u
        u = output[:, 0:1, 1:-1, 1:-1] #output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny,1,lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivative - v
        v = output[:, 1:2, 1:-1, 1:-1]#output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny,1,lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent-2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[1:-1, 0:1, 1:-1, 1:-1]#output[1:-1, 0:1, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]
        v = output[1:-1, 1:2, 1:-1, 1:-1]#output[1:-1, 1:2, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]

        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        R = 200.0

        # 2D burgers eqn
        f_u = u_t + u * u_x + v * u_y - (1/R) * laplace_u
        f_v = v_t + u * v_x + v * v_y - (1/R) * laplace_v

        return f_u, f_v


def compute_loss(output, loss_func):
    ''' calculate the phycis loss '''
    
    # Padding x axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
    #print('------------output-----------0:', output.shape)

    # Padding y axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)
    #print('------------output-----------1:', output.shape)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    #print('------------f_u, f_v-----------:', f_u.shape, f_v.shape)
    loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda()) 

    return loss


def train(model, input, initial_state, n_iters, time_batch_size, learning_rate, 
          dt, dx, save_path, pre_model_save_path, num_time_batch):

    train_loss_list = []
    second_last_state = []
    prev_output = []

    batch_loss = 0.0
    best_loss = 1e4

    # load previous model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
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
            #print('+++++++++++++++++++0:',time_batch_id, len(output), output[0].shape, u0.shape)
            # [t, c, height (Y), width (X)]
            output = torch.cat(tuple(output), dim=0) # (1001, 2, 128, 128)
            #print('+++++++++++++++++++1:',output.shape)  

            # concatenate the initial state to the output for central diff
            output = torch.cat((u0.cuda(), output), dim=0) # (1002, 2, 128, 128)
            #print('+++++++++++++++++++2:',output.shape)

            # get loss
            loss = compute_loss(output, loss_func)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()

            # update the state and output for next batch
            prev_output = output
            state_detached = []
            for i in range(len(second_last_state)):
                (h, c) = second_last_state[i]
                state_detached.append((h.detach(), c.detach())) # hidden state


        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.10f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0), 
            batch_loss))
        train_loss_list.append(batch_loss)

        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss

        if loss.item() < 1e-3:
            break
    
    return train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    ''' 
    axis_lim: [xmin, xmax, ymin, ymax]
    uv_lim: [u_min, u_max, v_min, v_max]
    num: Number of time step
    '''

    # get the limit 
    xmin, xmax, ymin, ymax = axis_lim
    u_min, u_max, v_min, v_max = uv_lim

    # grid
    x = np.linspace(xmin, xmax, 64+1)
    x = x[:-1]
    x_star, y_star = np.meshgrid(x, x)
    
    u_star = true[num, 0, 1:-2, 1:-2]
    u_pred = output[num, 0, 1:-2, 1:-2].detach().cpu().numpy()

    v_star = true[num, 1, 1:-2, 1:-2]
    v_pred = output[num, 1, 1:-2, 1:-2].detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    u_min, u_max, v_min, v_max = np.min(u_pred), np.max(u_pred), \
                                np.min(v_pred), np.max(v_pred)
    #print('****************************1:', u_min, u_max, v_min, v_max)
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_title('u-RCNN')
    fig.colorbar(cf, ax=ax[0, 0])

    u_min, u_max, v_min, v_max = float(np.min(u_star)), float(np.max(u_star)), \
                                float(np.min(v_star)), float(np.max(v_star))
    #print('****************************2:', float(u_min),  u_max, v_min, v_max)
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_title('u-Ref.')
    fig.colorbar(cf, ax=ax[0, 1])

    u_min, u_max, v_min, v_max = np.min(u_pred), np.max(u_pred), \
                                np.min(v_pred), np.max(v_pred)
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 0].set_title('v-RCNN')
    fig.colorbar(cf, ax=ax[1, 0])

    u_min, u_max, v_min, v_max = float(np.min(u_star)), float(np.max(u_star)), \
                            float(np.min(v_star)), float(np.max(v_star))
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 1].set_title('v-Ref.')
    fig.colorbar(cf, ax=ax[1, 1])

    # plt.draw()
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

    return u_star, u_pred, v_star, v_pred


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    if torch.load(save_dir) is None:
        return
    else:
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Pretrained model loaded!')

        return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))


if __name__ == '__main__':

    ######### download the ground truth data ############
    data_dir = '/home/gli12/project/PhyCRNet/data/2dBurgers/burgers_ic_1501x2x64x64.mat'
    M = 64
    S = int(M/8)
    # data_dir = './Datasets/ICs/IC_Burgers.mat'    
    data = scio.loadmat(data_dir)
    uv = data['uv'] # [t,c,h,w]  

    # initial conidtion
    uv0 = uv[0:1,...] 
    input = torch.tensor(uv0, dtype=torch.float32).cuda() 

    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (torch.randn(1, 128, S, S), torch.randn(1, 128, S, S))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))
    
    # grid parameters
    time_steps = 1001
    dt = 0.002
    dx = 1.0 / M

    ################# build the model #####################
    time_batch_size = 1000
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters_adam = 1000
    lr_adam = 1e-3 #1e-3 
    pre_model_save_path = None #'/home/gli12/project/PhyCRNet/model/burger64.pt' #None #
    model_save_path = '/home/gli12/project/PhyCRNet/model/burger64.pt'
    fig_save_path = '/home/gli12/project/PhyCRNet/figures/'  
 
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
        effective_step = effective_step).cuda()
    summary(model)

    start = time.time()
    train_loss = train(model, input, initial_state, n_iters_adam, time_batch_size, 
      lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch)
    end = time.time()
    
    #np.save('/home/gli12/project/PhyCRNet/model/train_loss', train_loss)  
    print('The training time is: ', (end-start))

    ########### model inference ##################
    time_batch_size_load = 1000
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
        effective_step = effective_step).cuda()

    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path) 
    start_predict = time.time()
    output, _ = model(initial_state, input)
    end_predict = time.time()
    print('The prediction time is: ', (end_predict-start_predict))

    # shape: [t, c, h, w] 
    output = torch.cat(tuple(output), dim=0)  
    output = torch.cat((input.cuda(), output), dim=0)
  
    # Padding x and y axis due to periodic boundary condition
    output = torch.cat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), dim=3)
    output = torch.cat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), dim=2)

    # [t, c, h, w]
    truth = uv[0:1001,:,:,:]

    # [101, 2, 131, 131]
    truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)

    # get propogation curve

    # post-process
    ten_true = []
    ten_pred = []
    #for i in range(0, 50):
    # for i in range(0, 50): 
    #     u_star, u_pred, v_star, v_pred = post_process(output, truth, [0,1,0,1], 
    #         [-0.7,0.7,-1.0,1.0], num=20*i, fig_save_path=fig_save_path)

    #     ten_true.append([u_star, v_star])
    #     ten_pred.append([u_pred, v_pred])

    # compute the error
    error = frobenius_norm(np.array(ten_pred)-np.array(ten_true)) / frobenius_norm(
        np.array(ten_true))

    print('The predicted error is: ', error)
    output = output.detach().cpu().numpy()

    u_test_err1, v_test_err1 = np.mean((output[501,0].squeeze()-truth[501,0].squeeze())**2), np.mean((output[501,1].squeeze()-truth[501,1].squeeze())**2) # t=1.0 seconds
    u_test_err0, v_test_err0 = np.mean((output[1,0].squeeze()-truth[1,0])**2), np.mean((output[1,1].squeeze()-truth[1,1])**2) # t=0.05 seconds
    u_test_err0_5, v_test_err0_5 = np.mean((output[251,0].squeeze()-truth[501,0])**2), np.mean((output[251,1].squeeze()-truth[501,1])**2) # t=0.5 seconds
    u_test_err1_5, v_test_err1_5 = np.mean((output[751,0].squeeze()-truth[751,0])**2), np.mean((output[751,1].squeeze()-truth[751,1])**2) # t=1.5 seconds
    u_test_err2, v_test_err2 = np.mean((output[-1,0].squeeze()-truth[-1,0])**2), np.mean((output[-1,1].squeeze()-truth[-1,1])**2) # t=2.0 seconds
    
    print('U ,V test_error t=0: ', np.sqrt(u_test_err0), np.sqrt(v_test_err0))
    print('U ,V test_error t=0.5:', np.sqrt(u_test_err1), np.sqrt(v_test_err1))
    print('U ,V test_error t=1: ', np.sqrt(u_test_err1), np.sqrt(v_test_err1))
    print('U ,V test_error t=1.5:', np.sqrt(u_test_err1_5), np.sqrt(v_test_err1_5))
    print('U ,V test_error t=2: ', np.sqrt(u_test_err2), np.sqrt(v_test_err2))

    u_pred = output[:-1, 0, :, :]#.detach().cpu().numpy()
    u_pred = np.swapaxes(u_pred, 1, 2) # [h,w] = [y,x]
    u_true = truth[:, 0, :, :]

    t_true = np.linspace(0, 2, 1001)
    t_pred = np.linspace(0, 2, time_steps)

    plt.plot(t_pred, u_pred[:, 32, 32], label='x=32, y=32, CRL')
    plt.plot(t_true, u_true[:, 32, 32], '--', label='x=32, y=32, Ref.')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.xlim(0, 2)
    plt.legend()
    plt.savefig(fig_save_path + "x=32,y=32.png")
    plt.close("all")
    # plt.show()

    # plot figure
    plt.figure()
    plt.imshow(output[-1, 0, :, :,].detach().cpu().numpy().squeeze(), cmap='viridis')
    plt.colorbar()
    plt.savefig(fig_save_path + 'burger0.png', dpi = 300)
    plt.close()

    plt.figure()
    plt.imshow(output[-1, 1, :, :,].detach().cpu().numpy().squeeze(), cmap='viridis')
    plt.colorbar()
    plt.savefig(fig_save_path + 'burger1.png', dpi = 300)
    plt.close()

    # plot train loss
    plt.figure()
    plt.plot(train_loss, label = 'train loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'train loss.png', dpi = 300)





















