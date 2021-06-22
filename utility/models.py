import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
    
def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should 
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(MultiRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout, 
                                         batch_first=True, bidirectional=bidirectional)
        
        

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_())
        
        
class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.
    
    args:
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should 
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """
    
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(FCLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None
            
        self.init_hidden()
    
    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)
              
    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)
            
            
class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()
        
        self.causal = causal
        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual
        
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, 
                 causal=False, dilated=True):
        super(TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)
        
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal)) 
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))   
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
                    
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
        # output layer
        
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )
        
        self.skip = skip
        
    def forward(self, input):
        
        # input shape: (B, N, L)
        
        # normalization
        output = self.BN(self.LN(input))
        
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual
            
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        
        return output


# base-CNN for D
class base_CNN(nn.Module):
    def __init__(self,sdr=False,wgan=False):
        super(base_CNN,self).__init__()

        self.sdr = sdr
        self.wgan = wgan
        self.causal = False

        if not self.sdr:
            # channels: mixture and estimation -> (B,2,16000)
            self.channels = 2
        else:
            # channels: mixture and estimation to SI-SNR -> (B,1,16000)
            self.channels = 1
        self.in_channel = self.channels

        if not self.causal:
            self.LN = nn.GroupNorm(self.in_channel, self.channels, eps=1e-8)
        else:
            self.LN = cLN(self.channels, eps=1e-8)

        # CNN 参数
        self.kernel = 3
        self.stride = 2
        self.padding = 1
        self.layer = 7

        self.conv = nn.ModuleList([])
        self.active = nn.ModuleList([])
        self.normal = nn.ModuleList([])
        for i in range(self.layer):
            self.conv.append(nn.Conv1d(self.channels,self.channels*2,self.kernel,self.stride,self.padding))
            self.channels *= 2

            if not self.causal:
                self.normal.append(nn.GroupNorm(self.in_channel, self.channels, eps=1e-8))
            else:
                self.normal.append(cLN(self.channels, eps=1e-8))

            self.active.append(nn.PReLU())

        if self.wgan:
            self.linear = nn.Sequential(
                # nn.Linear(self.in_channel * 128 * 125 * 3, 1),
                nn.Linear(self.in_channel * 128 * 125, self.in_channel * 32 * 5),
                nn.LayerNorm(normalized_shape=self.in_channel * 32 * 5, eps=1e-8),
                nn.PReLU(),
                nn.Linear(self.in_channel * 32 * 5, 1),
                # nn.PReLU(),
                # nn.Linear(self.in_channel * 3, 1),
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.in_channel * 128 * 125, self.in_channel * 32 * 5),
                nn.LayerNorm(normalized_shape=self.in_channel * 32 * 5, eps=1e-8),
                nn.PReLU(),
                nn.Linear(self.in_channel * 32 * 5, 1),

                # nn.Sigmoid(),
                nn.Tanh(),
            )

    def forward(self, x):

        x = self.LN(x)

        for i in range(self.layer):
            x = self.conv[i](x)
            x = self.active[i](self.normal[i](x))

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

# 深度可分离卷积 for D
class dilate_conv(nn.Module):
    def __init__(self,in_channel=1,channels=1,kernel=3,stride=2,dilation = 2,padding = 2,layer = 3):
        super(dilate_conv, self).__init__()
        self.causal = False
        self.in_channel = in_channel
        self.channels = channels
        # CNN 参数
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.layer = layer

        if not self.causal:
            self.LN = nn.GroupNorm(self.in_channel, self.channels, eps=1e-8)
        else:
            self.LN = cLN(self.channels, eps=1e-8)

        self.conv = nn.ModuleList([])
        self.active = nn.ModuleList([])
        self.normal = nn.ModuleList([])
        for i in range(self.layer):
            self.conv.append(nn.Conv1d(self.channels, self.channels, self.kernel, self.stride, self.padding, dilation=self.dilation))

            if not self.causal:
                self.normal.append(nn.GroupNorm(self.in_channel, self.channels, eps=1e-8))
            else:
                self.normal.append(cLN(self.channels, eps=1e-8))
            self.active.append(nn.PReLU())

    def forward(self, x):

        x = self.LN(x)

        for i in range(self.layer):
            x = self.conv[i](x)
            x = self.active[i](self.normal[i](x))

        return x

class dilate_CNN(nn.Module):
    def __init__(self, win=32):
        super(dilate_CNN, self).__init__()
        # window = 2ms
        self.win = win
        self.stride = self.win // 2

        self.encoder1 = nn.Conv1d(1, 128, self.win, bias=False, stride=self.stride) # for mix to shape [B,128,1000]
        self.encoder2 = nn.Conv1d(1, 128, self.win, bias=False, stride=self.stride) # for estimation to shape [B,128,1000]

        self.dilate_conv1 = dilate_conv(channels=128)  # for mix to shape [B,128,125]
        self.dilate_conv2 = dilate_conv(channels=128)  # for estimation to shape [B,128,125]

        # input shape [B,2,2000]
        self.conv = dilate_conv(2, 256, layer=4)  # out [B,256,8]

        # self.linear = nn.Linear(256 * 125,1)
        self.linear = nn.Sequential(
            nn.Linear(256 * 8, 64 * 4),
            nn.LayerNorm(normalized_shape=64 * 4, eps=1e-8),
            nn.PReLU(),
            nn.Linear(64 * 4, 1),
        )

    def forward(self, mix, estimation):
        x_mix = self.dilate_conv1(self.encoder1(mix))
        x_est = self.dilate_conv2(self.encoder2(estimation))

        x = torch.cat((x_mix, x_est), 1)
        x = self.conv(x)
        # print(x_mix.shape,x.shape)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)

        return x

# 8.20 spectral normao base-CNN for D
class sn_base_CNN(nn.Module):
    def __init__(self,wgan=False):
        super(sn_base_CNN,self).__init__()


        self.wgan = wgan
        self.causal = False

        self.channels = 2
        self.in_channel = self.channels

        # if not self.causal:
        #     self.LN = nn.GroupNorm(self.in_channel, self.channels, eps=1e-8)
        # else:
        #     self.LN = cLN(self.channels, eps=1e-8)

        # CNN 参数
        self.kernel = 3
        self.stride = 2
        self.padding = 1
        self.layer = 7

        self.conv = nn.ModuleList([])
        self.active = nn.ModuleList([])
        # self.normal = nn.ModuleList([])
        for i in range(self.layer):
            self.conv.append(spectral_norm(nn.Conv1d(self.channels,self.channels*2,self.kernel,self.stride,self.padding)))
            self.channels *= 2

            # if not self.causal:
            #     self.normal.append(nn.GroupNorm(self.in_channel, self.channels, eps=1e-8))
            # else:
            #     self.normal.append(cLN(self.channels, eps=1e-8))

            self.active.append(nn.PReLU())

        if self.wgan:
            self.linear = nn.Sequential(
                # nn.Linear(self.in_channel * 128 * 125 * 3, 1),
                spectral_norm(nn.Linear(self.in_channel * 128 * 125, self.in_channel * 32 * 5)),
                # nn.LayerNorm(normalized_shape=self.in_channel * 32 * 5, eps=1e-8),
                nn.LeakyReLU(),
                spectral_norm(nn.Linear(self.in_channel * 32 * 5, 1)),
                # nn.PReLU(),
                # nn.Linear(self.in_channel * 3, 1),
            )
        else:
            self.linear = nn.Sequential(
                spectral_norm(nn.Linear(self.in_channel * 128 * 125, self.in_channel * 32 * 5)),
                nn.PReLU(),
                spectral_norm(nn.Linear(self.in_channel * 32 * 5, 1)),
                # nn.PReLU(),

                # nn.Tanh(),
            )

    def forward(self, x):

        # x = self.LN(x)

        for i in range(self.layer):
            x = self.conv[i](x)
            x = self.active[i]((x))

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

class SNConv(nn.Module):
    def __init__(self,encoder=None):
        super(SNConv,self).__init__()

        if encoder is None:
            self.is_encoder_valable = False
        else:
            self.is_encoder_valable = True
            self.encoder = encoder

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 15, 5, 2)),  # <-- (3,2,512,999)
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(15, 25, 7, 2)),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(25, 40, 9, 2)),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(40, 50, 11, 2)),  # --> (3,50,24,55)
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),  # --> (3,50,1,1)
        )
        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(50, 50)), # <-- (3,50)
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(50, 10)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(10, 1)), # --> (3,1)
            # nn.Tanh() #3.30
        )
        # self.tanh = nn.Tanh()

    # def pad_signal(self, input):
    #
    #     # input is the waveforms: (B, T) or (B, 1, T)
    #     # reshape and padding
    #     if input.dim() not in [2, 3]:
    #         raise RuntimeError("Input can only be 2 or 3 dimensional.")
    #
    #     if input.dim() == 2:
    #         input = input.unsqueeze(1)
    #
    #     return input

    def forward(self, x, est=None):
        # ref, est = self.pad_signal(ref), self.pad_signal(est)
        # ref, est = encoder(ref).detach(), encoder(est).detach()
        # x = torch.cat((ref.unsqueeze(1),est.unsqueeze(1)),1)
        if self.is_encoder_valable:
            x = self.encoder(x,est)

        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        # x = self.tanh(x)

        return x

class Encoder(nn.Module):
    def __init__(self,encoder):
        super(Encoder,self).__init__()

        self.encoder = encoder
        self.win = 32
        self.stride = 16

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        device = input.device
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        # 计算填充0数据点的数量rest
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type()).to(device)
            input = torch.cat([input, pad], 2)

        # print('input:', input.shape)
        # 多补一个self.stride长，相当于STFT多加一帧，在逆变换时保持原长度
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type()).to(device)
        input = torch.cat([pad_aux, input, pad_aux], 2)

        # print('input:',input.shape)
        return input

    def forward(self,ref, est):
        ref, est = self.pad_signal(ref), self.pad_signal(est)
        ref, est = self.encoder(ref), self.encoder(est)
        x = torch.cat((ref.unsqueeze(1), est.unsqueeze(1)), 1)
        return x

if __name__ == "__main__":
    a = np.ones((3,1,16000)).astype(np.float32)
    a = torch.from_numpy(a)
    # b = base_CNN()
    # c = b(a)
    # print(c)
    b = np.ones((3, 2, 512, 999)).astype(np.float32)
    b = torch.from_numpy(b)
    # c = nn.Conv1d(1, 512, 32, bias=False, stride=16)
    # print(SNConv()(a,a,c).shape)
    c = nn.Sequential(
        nn.Conv2d(2,15,5,2), # <-- (3,2,512,999)
        nn.Conv2d(15,25,7,2),
        nn.Conv2d(25,40,9,2),
        nn.Conv2d(40,50,11,2), # --> (3,50,24,55)
        # nn.Conv2d(50,50,13),
        # nn.Conv2d(50,50,15), # --> (3,50,458,945)
        # nn.AdaptiveAvgPool2d(1), # --> (3,50,1,1)
    )
    print(c(b).shape)
    # d = nn.Sequential(
    #     nn.Linear(50,50),
    #     nn.Linear(50,10),
    #     nn.Linear(10,1),
    # )
    # print(d(c(b).view(c(b).shape[0],-1)).shape)