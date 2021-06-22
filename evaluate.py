import numpy as np
import soundfile
import librosa
from pesq import pesq
import pysepm
from pystoi import stoi
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt

import os

from utility.sdr import calc_sdr

# 评测单组音频
def evaluate_one(clean_name,estimation_name,sample_rate=16000):

    # print(clean_name,estimation_name)
    clean,sr = librosa.load(clean_name,sr=sample_rate)
    estimation,sr = librosa.load(estimation_name,sr=sample_rate)

    # sr,clean = wavfile.read(clean_name)
    # sr,estimation = wavfile.read(estimation_name)

    # print('pesq:', pesq(sr, clean, estimation, 'wb'))
    # pesq_value = pesq(sr, clean, estimation, 'wb')
    _, pesq_value = pysepm.pesq(clean, estimation, sr)

    # stoi_value = stoi(clean, estimation, sr, extended=False)

    clean, estimation = clean[np.newaxis,:], estimation[np.newaxis,:]

    # print(calc_sdr(estimation,clean)[0])
    sdr = calc_sdr(estimation,clean)[0]

    return pesq_value,sdr

def batch_evaluate(clean_path,estimation_path,evaluate_path='./evaluate/pesq_and_sisnr/',model=''):

    clean_names = glob.glob(os.path.join(clean_path,'*.wav'))
    estimation_names = glob.glob(os.path.join(estimation_path,'*.wav'))

    # 记录评价值
    all_pesq_value, all_sdr, all_stoi = [], [],[]

    for clean_name,estimation_name in zip(clean_names,estimation_names):
        # assert os.path.basename(clean_name) == os.path.basename(estimation_name),'{} is not match {} !'.format(clean_name,estimation_name)

        pesq_value, sdr = evaluate_one(clean_name, estimation_name)

        all_pesq_value.append(pesq_value)
        all_sdr.append(sdr)


    all_pesq_value, all_sdr = np.array(all_pesq_value),np.array(all_sdr)

    # 如果操作路径不存在，则创建它
    if not os.path.exists(evaluate_path):
        os.mkdir(evaluate_path)

    # 保存数据
    np.save(os.path.join(evaluate_path,'all_pesq_value_{}.npy'.format(model)),all_pesq_value)
    np.save(os.path.join(evaluate_path,'all_sdr_{}.npy'.format(model)),all_sdr)
    # np.save(os.path.join(evaluate_path,'all_stoi.npy'),all_stoi)

    # 返回均值
    return np.mean(all_pesq_value),np.mean(all_sdr)

def evaluate_one_composite(clean_name,estimation_name,sample_rate=16000):

    # print(clean_name,estimation_name)
    clean,sr = librosa.load(clean_name,sr=sample_rate)
    estimation,sr = librosa.load(estimation_name,sr=sample_rate)

    Csig, Cbak, Covl = pysepm.composite(clean, estimation, sr)

    return Csig,Cbak,Covl

def batch_evaluate_composite(clean_path,estimation_path,evaluate_path='./evaluate/composite/',model=''):

    clean_names = glob.glob(os.path.join(clean_path,'*.wav'))
    estimation_names = glob.glob(os.path.join(estimation_path,'*.wav'))

    # 记录评价值
    all_csig, all_cbak, all_covl = [], [], []

    for clean_name,estimation_name in zip(clean_names,estimation_names):
        # assert os.path.basename(clean_name) == os.path.basename(estimation_name),'{} is not match {} !'.format(clean_name,estimation_name)

        Csig,Cbak,Covl = evaluate_one_composite(clean_name, estimation_name)

        all_csig.append(Csig)
        all_cbak.append(Cbak)
        all_covl.append(Covl)

    all_csig, all_cbak, all_covl = np.array(all_csig), np.array(all_cbak), np.array(all_covl)

    # 如果操作路径不存在，则创建它
    if not os.path.exists(evaluate_path):
        os.mkdir(evaluate_path)

    # 保存数据
    np.save(os.path.join(evaluate_path,'all_csig_{}.npy'.format(model)),all_csig)
    np.save(os.path.join(evaluate_path,'all_cbak_{}.npy'.format(model)),all_cbak)
    np.save(os.path.join(evaluate_path,'all_covl_{}.npy'.format(model)),all_covl)

    # 返回均值
    return np.mean(all_csig),np.mean(all_cbak), np.mean(all_covl)

def evaluate_one_segSNR(clean_name,estimation_name,sample_rate=16000):

    # print(clean_name,estimation_name)
    clean,sr = librosa.load(clean_name,sr=sample_rate)
    estimation,sr = librosa.load(estimation_name,sr=sample_rate)

    segsnr = pysepm.SNRseg(clean, estimation, sr)

    return segsnr

def batch_evaluate_segSNR(clean_path,estimation_path,evaluate_path='./evaluate/segSNR/',model=''):

    clean_names = glob.glob(os.path.join(clean_path,'*.wav'))
    estimation_names = glob.glob(os.path.join(estimation_path,'*.wav'))

    # 记录评价值
    all_segSNR = []

    for clean_name,estimation_name in zip(clean_names,estimation_names):
        # assert os.path.basename(clean_name) == os.path.basename(estimation_name),'{} is not match {} !'.format(clean_name,estimation_name)

        segsnr = evaluate_one_segSNR(clean_name, estimation_name)

        all_segSNR.append(segsnr)

    all_segSNR = np.array(all_segSNR)

    # 如果操作路径不存在，则创建它
    if not os.path.exists(evaluate_path):
        os.mkdir(evaluate_path)

    # 保存数据
    np.save(os.path.join(evaluate_path,'all_segSNR_{}.npy'.format(model)),all_segSNR)

    # 返回均值
    return np.mean(all_segSNR)

def evaluate_one_STOI(clean_name, estimation_name, sample_rate=16000):
    clean, sr = librosa.load(clean_name, sr=sample_rate)
    estimation, sr = librosa.load(estimation_name, sr=sample_rate)

    stoi_value = stoi(clean, estimation, sr, extended=False)

    return stoi_value

def batch_evaluate_STOI(clean_path,estimation_path,evaluate_path='./evaluate/STOI/',model=''):
    clean_names = glob.glob(os.path.join(clean_path, '*.wav'))
    estimation_names = glob.glob(os.path.join(estimation_path, '*.wav'))

    # report STOI value
    all_stoi = []
    for clean_name, estimation_name in zip(clean_names, estimation_names):
        stoi_value = evaluate_one_STOI(clean_name, estimation_name)
        all_stoi.append(stoi_value)

    # 如果操作路径不存在，则创建它
    if not os.path.exists(evaluate_path):
        os.mkdir(evaluate_path)

    # 保存数据
    np.save(os.path.join(evaluate_path, 'all_STOI_{}.npy'.format(model)), all_stoi)

    # 返回均值
    return np.mean(all_stoi)

if __name__ == "__main__":


    # clean_path = r'./data/test/16k_clean_all'
    # gan_path = r'./evaluate/base_gan'
    # wgan_path = r'./result/WGAN_epoch100_batch160_16k'
    # wgan2_path = r'./result/win_2ms_WGAN_epoch100_batch70_16k_sdr'
    # # tas_path = r'./evaluate/noisy_test'
    # tas_path = r'./result/win_2ms_Tasnet_epoch100_batch70_16k'
    #
    # print(batch_evaluate(clean_path,wgan2_path,model='WGAN-sdr'))
    # print(batch_evaluate_composite(clean_path,wgan2_path,model='WGAN-sdr'))
    # print(batch_evaluate_segSNR(clean_path, wgan2_path, model='WGAN-sdr'))

    clean_path = r'./data/test/16k_clean_all'
    gan_path = r'./evaluate/base_gan'
    wgan_path = r'./result/WGAN_epoch100_batch160_16k'
    wgan2_path = r'./result/win_2ms_WGAN_epoch100_batch70_16k_sdr'
    # tas_path = r'./evaluate/noisy_test'
    bi_tasnet_path = r'./result/win_2ms_TSEGAN_epoch100_batch70_16k'
    # bi_tasnet_path = r'./result/Conv-TasNet-MSE'

    print(batch_evaluate(clean_path, bi_tasnet_path, model='Bi_Tasnet'))
    print(batch_evaluate_composite(clean_path, bi_tasnet_path, model='Bi_Tasnet'))
    print(batch_evaluate_segSNR(clean_path, bi_tasnet_path, model='Bi_Tasnet'))
    print(batch_evaluate_STOI(clean_path, bi_tasnet_path, model='Bi_Tasnet'))

    # print(batch_evaluate_composite(clean_path,gan_path,model='Gan'))
    # print(batch_evaluate_composite(clean_path,wgan_path,model='WGAN'))

    # print(batch_evaluate(clean_path, tas_path, model='Tas'))
    # print(batch_evaluate_composite(clean_path, tas_path, model='Tas'))
    # print(batch_evaluate_segSNR(clean_path, tas_path, model='Tas'))

    # print(batch_evaluate_segSNR(clean_path, gan_path, model='Gan'))
    # print(batch_evaluate_segSNR(clean_path, wgan_path, model='WGAN'))


