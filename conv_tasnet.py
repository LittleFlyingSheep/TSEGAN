import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utility import models, sdr


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3,
                 kernel=3, num_spk=1, causal=False):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        self.TCN = models.TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
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
            pad = torch.zeros(batch_size, 1, rest).type(input.type()).to(device)
            input = torch.cat([input, pad], 2)

        # print('input:', input.shape)
        # 多补一个self.stride长，相当于STFT多加一帧，在逆变换时保持原长度
        pad_aux = torch.zeros(batch_size, 1, self.stride).type(input.type()).to(device)
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input):
        
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)

        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L
        # print('enc_output:', enc_output.shape)

        # # random vector z ---- 2021.3.19
        # z = torch.randn(enc_output.shape).to(enc_output.device)
        # enc_output += z

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        # print('mask_output:', masked_output.shape)

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        # print('output:', output.shape)
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        # print('output:', output.shape)
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output

# SI-SNR D
'''
    input: mixture, estimation
        mixture: (B,16000), mixture audio as reference
        estimation: (B,16000), clean or result from G
    return: 
        output: a mark estimated by D in [0,1] or Wassertain or [-1,1]
'''
class D(nn.Module):
    def __init__(self,sdr=False,wgan=False,model_type='base'):
        super(D,self).__init__()

        self.sdr = sdr

        self.wgan = wgan

        self.model_type = model_type

        if self.model_type == 'base':
            self.base_cnn = models.base_CNN(sdr=self.sdr,wgan=self.wgan)
        elif self.model_type == 'latent':
            self.dilate_cnn = models.dilate_CNN()
        elif self.model_type == 'metric':
            self.cnn = models.sn_base_CNN(wgan=self.wgan)
        else:
            raise Exception("Invalid model type!")

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        return input

    def forward(self,mixture,estimation):

        if self.model_type == 'base':
            if not self.sdr:
                # without SI-SNR
                # reshape -> (B,2,16000)
                mixture, estimation = self.pad_signal(mixture), self.pad_signal(estimation)
                x = torch.cat((mixture, estimation), 1)
                # print(mixture.shape,estimation.shape)
            else:
                # with SI-SNR
                # reshape -> (B,1,16000)
                x = sdr.calc_sdr_D(estimation, mixture, zero_mean=True)

            x = self.pad_signal(x)

            x = self.base_cnn(x)

        elif self.model_type == 'latent':
            mixture, estimation = self.pad_signal(mixture), self.pad_signal(estimation)
            x = self.dilate_cnn(mixture, estimation)

        elif self.model_type == 'metric':
            mixture, estimation = self.pad_signal(mixture), self.pad_signal(estimation)
            x = torch.cat((mixture, estimation), 1)
            x = self.cnn(x)

        return x

def test_conv_tasnet():
    x = torch.rand(2, 48000)
    nnet = TasNet()
    z = nnet(x)
    print('z:',z.shape)
    s1 = z[0]
    print(s1.shape)

def test_D():
    x = torch.rand(2, 16000)
    y = torch.rand(2, 16000)
    nnet = D()
    x = nnet(x,y)
    print('x:', x.shape)
    s1 = x[0]
    print(s1.shape)

if __name__ == "__main__":
    test_conv_tasnet()
    # test_D()
