import numpy as np
import librosa
import soundfile as sf
import os
import glob

## 语音预处理，将数据划分为1s的块（sr），overlap = 50%

# 分割重写单个音频
def process_one(wav,aim_path):
    '''
    input:
        wav: audio file for process
        aim_path: where the result is writed
    example:
        wav = r'E:\DeepStudy\segan\data\noisy_trainset\p226_002.wav'
        aim_path = r'./data'
    '''
    y,sr = librosa.load(wav,sr=None)
    wav_name = os.path.basename(wav)[:-4]

    len_y = len(y)
    # 少于1s剔除
    if len_y < sr:
        pass
    # 超过1s但不被1s整除的，舍去多余部分进行分割
    elif len_y % sr != 0:
        len_y = len_y // sr * sr + 1
        for i in range(sr,len_y,sr):
            aim_wav = os.path.join(aim_path,(wav_name + '_{}.wav'.format(i//(sr)-1)))
            sf.write(aim_wav,y[i-sr:i],sr)
    # 整除1s的，直接分割
    else:
        for i in range(sr, len_y, sr):
            aim_wav = os.path.join(aim_path, (wav_name + '_{}.wav'.format(i//(sr)-1)))
            sf.write(aim_wav, y[i - sr:i], sr)

# 批处理多个音频文件进行预处理
def batch_process(wav_path,aim_path):
    '''
        input:
            wav_path: audio files path for process
            aim_path: where the result is writed
        example:
            wav_path = r'E:\DeepStudy\segan\data\noisy_trainset'
            aim_path = r'./data/train/mixture'
    '''
    # 如果操作路径不存在，则创建它
    if not os.path.exists(aim_path):
        os.mkdir(aim_path)

    wav_files = glob.glob(os.path.join(wav_path,'*.wav'))

    for wav in wav_files:
        process_one(wav,aim_path)
        print('Finished {} at {}'.format(wav,aim_path))

# 重采样48000kHz -> 16000kHz
def re_sample_one(wav,aim_path):
    '''
        input:
            wav: audio file for process
            aim_path: where the result is writed
        example:
            wav = r'E:\DeepStudy\segan\data\noisy_trainset\p226_002.wav'
            aim_path = r'./data'
    '''

    y,sr = librosa.load(wav,sr=16000)
    name = os.path.basename(wav)
    name = os.path.join(aim_path,name)
    sf.write(name,y,sr)

# 重采样48000kHz -> 16000kHz
def batch_re_sample(wav_path,aim_path):
    '''
        input:
            wav_path: audio files path for process
            aim_path: where the result is writed
        example:
            wav_path = r'E:\DeepStudy\segan\data\noisy_trainset'
            aim_path = r'./data/train/mixture'
    '''

    # 如果操作路径不存在，则创建它
    if not os.path.exists(aim_path):
        os.mkdir(aim_path)

    wav_files = glob.glob(os.path.join(wav_path, '*.wav'))

    for wav in wav_files:
        re_sample_one(wav,aim_path)
        print('Resampled {} at {}'.format(wav, aim_path))

if __name__ == "__main__":

    mix_path = r'./data/noisy_trainset_56spk_wav'
    mix_aim_path = r'./data/train/mixture_56spk'
    clean_path = r'./data/clean_trainset_56spk_wav'
    clean_aim_path = r'./data/train/clean_56spk'

    mix_aim_path_16k = r'./data/train/16k_mixture_56spk_all'
    clean_aim_path_16k = r'./data/train/16k_clean_56spk_all'

    # process_one(wav,aim_path=aim_path)
    batch_process(mix_path,mix_aim_path)
    batch_process(clean_path,clean_aim_path)

    batch_re_sample(mix_aim_path,mix_aim_path_16k)
    batch_re_sample(clean_aim_path,clean_aim_path_16k)