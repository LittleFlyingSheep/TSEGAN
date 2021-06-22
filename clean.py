import numpy as np
import librosa
import soundfile as sf
import glob
import os
from time import *
import torch
import torch.nn as nn
from tqdm import tqdm

from utility.utils import get_model, get_last_model
from train import device_ids

device = torch.device('cuda:1')
# 增强测试

def test_GAN_clean(test_mix_path,aim_path):
    # 如果操作路径不存在，则创建它
    if not os.path.exists(aim_path):
        os.mkdir(aim_path)
        print('Creat {}'.format(aim_path))

    mix_names = glob.glob(os.path.join(test_mix_path,'*.wav'))

    # G = get_last_model(r'./save_models/G',index=-1).cuda()
    G = get_model(r'./save_models/G_useful', 'W-TSEGAN-L1.pkl').cuda(device)
    # G = get_model('./nets/G_useful/', 'G_val-batch45-epoch-96-step-592-SISNR.pkl')0-sdr-
    # G = get_model('./nets/tasnet_100/', 'Conv-TasNet-MSE.pkl')

    pbar = tqdm(mix_names, desc='Clean: ')
    for mix_name in pbar:
        start_time = time()

        mix,sr = librosa.load(mix_name,sr=None)
        mix = mix[np.newaxis,:]
        # print(mix.shape)
        mix_torch = torch.from_numpy(mix.astype(np.float32)).cuda(device)

        # if torch.cuda.is_available():
        #     mix_torch = mix_torch.cuda()

        estimation_torch = G(mix_torch)
        estimation_torch = estimation_torch.cpu()
        estimation = estimation_torch.data.numpy()
        estimation = np.squeeze(estimation)

        aim = os.path.join(aim_path,os.path.basename(mix_name))
        sf.write(aim,estimation,sr)

        end_time = time()
        # print('Clean as {} , with {}s.'.format(aim,end_time-start_time))

if __name__ == "__main__":
    #test_mix_path = r'./data/test/16k_mixture'
    test_mix_path = r'./data/test/16k_mixture_all'
    #test_mix_path = r'./data/test/real_test'
    aim_path = r'./result/win_2ms_TSEGAN_epoch100_batch70_16k'
    #aim_path = r'./result/real_test'

    test_GAN_clean(test_mix_path,aim_path)
