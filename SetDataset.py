import numpy as np
import librosa
import torch
import glob
import os
from torch.utils.data import DataLoader,Dataset

# 自定义Dataset，保证mix与clean对应导入

class AudioDataset(Dataset):
    def __init__(self,mix_path,clean_path):
        super(AudioDataset,self).__init__()

        mix_wavs = glob.glob(os.path.join(mix_path, '*.wav'))
        clean_wavs = glob.glob(os.path.join(clean_path,'*.wav'))

        assert len(mix_wavs) == len(clean_wavs), "Mixture and clean audios should have same size."

        self.mix_wavs = mix_wavs
        self.clean_wavs = clean_wavs

    def __getitem__(self, item):
        mix_name = self.mix_wavs[item]
        clean_name = self.clean_wavs[item]

        mix,sr = librosa.load(mix_name,sr=None)
        clean,sr = librosa.load(clean_name,sr=None)

        return torch.from_numpy(mix),torch.from_numpy(clean),os.path.basename(mix_name)

    def __len__(self):
        return len(self.mix_wavs)

# class AudioDataset(torch.utils.data.Dataset):
#     def __init__(self,mix_path_1,clean_path_1,mix_path_2,clean_path_2):
#         super(AudioDataset,self).__init__()
#
#         mix_wavs_1 = glob.glob(os.path.join(mix_path_1, '*.wav'))
#         clean_wavs_1 = glob.glob(os.path.join(clean_path_1,'*.wav'))
#         mix_wavs_2 = glob.glob(os.path.join(mix_path_2, '*.wav'))
#         clean_wavs_2 = glob.glob(os.path.join(clean_path_2, '*.wav'))
#
#         mix_wavs = mix_wavs_1 + mix_wavs_2
#         clean_wavs = clean_wavs_1 + clean_wavs_2
#
#         assert len(mix_wavs) == len(clean_wavs), "Mixture and clean audios should have same size."
#
#         self.mix_wavs = mix_wavs
#         self.clean_wavs = clean_wavs
#
#     def __getitem__(self, item):
#         mix_name = self.mix_wavs[item]
#         clean_name = self.clean_wavs[item]
#
#         mix,sr = librosa.load(mix_name,sr=None)
#         clean,sr = librosa.load(clean_name,sr=None)
#
#         return torch.from_numpy(mix),torch.from_numpy(clean)
#
#     def __len__(self):
#         return len(self.mix_wavs)

if __name__ == "__main__":
    mix_path = r'./data/train/mixture'
    clean_path = r'./data/train/clean'

    train_data = AudioDataset(mix_path=mix_path,clean_path=clean_path)
    train_loader = DataLoader(dataset=train_data,batch_size=3)

    for step,(mix,clean) in enumerate(train_loader):
        if step == 0:
            print(mix.shape,clean.shape)
            print(mix)
            print(clean)
        else:
            break