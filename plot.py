import torch
import librosa.display
import numpy as np
import matplotlib.pyplot as plt



y, sr = librosa.load(r'./data/clean_testset_wav/p232_199.wav',sr=None)
y = torch.from_numpy(y).float().unsqueeze(0).cuda()

G = torch.load(r'./nets/G_useful/G_good-batch70-epoch-84-step-400-sdr-18.71.pkl')

# models = G.modules()
# for name, model in models:
#     print(name)

if isinstance(G, torch.nn.DataParallel):
    G = G.module

# padding
y_t, rest = G.pad_signal(y)
print(y_t.shape)
batch_size = y_t.size(0)
# print('output:',output.shape)

# waveform encoder
y_t = G.encoder(y_t).squeeze()  # B, N, L

# encoder = G.encoder
# y_t = encoder(y)
y_t = y_t.cpu().data.numpy().astype(np.float32) * 1e5
print(y_t.shape)

plt.figure()

#librosa.display.specshow(y_t,sr=sr,y_axis='linear')
plt.imshow(y_t)

plt.title('Time-domain Feature')

plt.show()