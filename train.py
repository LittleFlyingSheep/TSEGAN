import numpy as np
import librosa
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.autograd as autograd

from utility import models, sdr, utils
from SetDataset_2 import AudioDataset, split_dataloader
#import conv_tasnet
# import Bi_conv_tasnet as conv_tasnet
import conv_tasnet as conv_tasnet
from utility.utils import get_last_model
from tqdm import tqdm
import time

# 超参数
BATCH_SIZE = 50
LR = 1e-3
EPOCH = 200
Lamda = 200
Beta = 1
patient = 10
NETS_PATH = r'./save_models'
device_ids = [0,1]
# 标签类型
REAL_LABEL = 1
FAKE_LABEL = 0

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

def train_TasNet(train_loader, test_loader, batch=BATCH_SIZE):
    # 网络初始化
    # i = 100
    # tasnet = utils.get_last_model('./nets/bi_tasnet_100')

    tasnet = conv_tasnet.TasNet()
    if torch.cuda.is_available():
        print('GPU ok')
        # tasnet = torch.nn.DataParallel(tasnet, device_ids=device_ids)  # 声明所有可用设备
        tasnet.cuda()
    else:
        print('GPU lost, using CPU')

    # 优化器
    opt = torch.optim.Adam(tasnet.parameters(), lr=LR, weight_decay=0)
    # 动态调整学习率
    scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3, verbose=True)

    mse_loss = nn.MSELoss()

    best_val_sdr = 0
    patience = 10
    patience_count = 0
    for epoch in range(EPOCH):

        sl, mse = 0, 0
        d_len = len(train_loader)
        pbar = tqdm(train_loader, desc='Epoch: {}'.format(epoch + 1), ncols=100)
        for step, (mix, clean) in enumerate(pbar):

            if torch.cuda.is_available():
                mix,clean = mix.cuda(),clean.cuda()

            # tasnet输出
            estimation = tasnet(mix)

            clean = torch.unsqueeze(clean,dim=1)

            opt.zero_grad()
            sdr_l = -sdr.batch_SDR_torch(estimation,clean)
            loss = mse_loss(estimation,clean)
            loss.backward()
            opt.step()
            sl += sdr_l.item()
            mse += loss.item()

        scheduler.step(mse / d_len)  # tasnet针对SI-SNR调整lr
        time.sleep(2)

        val_sdr = validition(tasnet, test_loader)

        print('Epoch: ', epoch,
              '| sdr: %.8f' % (sl / d_len),
              '| mse: %.8f' % (mse / d_len),
              '| val_sdr: %.8f' % (val_sdr),
              '| mse: %.8f' % (best_val_sdr),
              )

        if best_val_sdr - val_sdr > 1e-4:
            best_val_sdr = val_sdr
            patience_count = 0
            utils.save_net(NETS_PATH, net=tasnet, net_type='TasNet', batch=batch, epoch=epoch, step=step, num=100,
                           sdr_loss=best_val_sdr)
        else:
            patience_count += 1

        if patience_count >= patience:
            print('Never improving for {} epochs, so early stopping.'.format(patience))

def calculate_gradient_penatly(netD, real_wavs, fake_wavs, k=10, p=2, n=1):
    """Calculates the gradient penalty loss for WGAN GP"""
    eta = torch.FloatTensor(real_wavs.size(0),1,1).uniform_(0,1)
    eta = eta.expand(real_wavs.size(0),1,real_wavs.size(1))
    real_wavs = utils.pad_signal(real_wavs)

    # 中间变量
    interpolated = eta * real_wavs.cpu() + ((1 - eta) * fake_wavs.cpu())
    interpolated.requires_grad_(True)

    if torch.cuda.is_available():
        interpolated = interpolated.cuda()

    # 中间变量的判别器D评价
    prob_interpolated = netD(real_wavs,interpolated)
    # 对应的梯度维度
    grad_outputs = torch.ones(prob_interpolated.size())

    if torch.cuda.is_available():
        grad_outputs = grad_outputs.cuda()

    # 计算对应梯度
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_penalty = ((gradients.norm(2, dim=1) - n) ** p).mean() * k

    return gradients_penalty

def train_WGAN_GP(data_loader,D_sdr=False,G_loss_type='mse',wgan=True,lamda=Lamda,beta=Beta,batch=BATCH_SIZE):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # 网络初始化
    G = conv_tasnet.TasNet(win=2)
    D = models.SNConv()

    if torch.cuda.is_available():
        D = torch.nn.DataParallel(D, device_ids=device_ids)  # 声明所有可用设备
        G = torch.nn.DataParallel(G, device_ids=device_ids)  # 声明所有可用设备
        D.cuda()
        G.cuda()

    # D = get_model(r'./nets/D', r'D-batch70-epoch-23-step-400-sdr-0.pkl')
    # G = get_model(r'./nets/G', r'G-batch70-epoch-23-step-400-sdr-0.pkl')

    # 优化器
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR)
    # 动态调整学习率
    scheduler_D = ReduceLROnPlateau(opt_D, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    scheduler_G = ReduceLROnPlateau(opt_G, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    # scheduler_D_multi = MultiStepLR(opt_D, milestones=[30], gamma=0.1)
    # scheduler_G_multi = MultiStepLR(opt_G, milestones=[30], gamma=0.1)
    # scheduler_D = MultiStepLR(opt_D, milestones=[200,400],gamma=0.1)
    # scheduler_G = MultiStepLR(opt_G, milestones=[200,400],gamma=0.1)

    mse_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()

    train_loader, test_loader = split_dataloader(batch_size=BATCH_SIZE, train_ratio=0.95, dataset=data_loader)

    i = 0
    best_loss = 0
    patient = 0
    best_epoch = 0
    best_val_sdr = 0
    val_sdr = 0
    for epoch in range(i, EPOCH + i):
        dl, gl, mr, mf, sl, mse_l = 0, 0, 0, 0, 0, 0
        q_sl = 0
        d_len = len(train_loader)
        g_len = 0
        pbar = tqdm(train_loader, desc='Epoch: {}'.format(epoch + 1), ncols=80)

        if epoch < 0:
            for step, (mix, clean) in enumerate(pbar):
                if torch.cuda.is_available():
                    mix, clean = mix.cuda(), clean.cuda()
                opt_G.zero_grad()
                # G 的输出
                G_estimation = G(mix)
                clean = torch.unsqueeze(clean, dim=1)

                mse = mse_loss(G_estimation, clean)
                g_loss = mse
                g_loss.backward()
                gl += g_loss.item()
                mse_l += mse.item()
                opt_G.step()

                sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                sl += sdr_loss.item()
            train_loss = sl / d_len
        else:

            for step, (mix, clean) in enumerate(pbar):

                # target = torch.ones(mix.shape[0], 1)
                if torch.cuda.is_available():
                    mix, clean = mix.cuda(), clean.cuda()
                    # target = target.cuda()

                # print(step)
                # G 的输出
                G_estimation = G(mix).detach()

                Encoder = models.Encoder(G.module.encoder)
                utils.set_requires_grad(Encoder)
                Encoder = torch.nn.DataParallel(Encoder, device_ids=device_ids)  # 声明所有可用设备
                Encoder.cuda()

                # train D
                opt_D.zero_grad()
                # print('d:',clean.shape)
                mark_real = D(Encoder(clean, clean)).mean()
                # mark_real = D(clean,clean)

                # D_loss_real = mse_loss(mark_real, target)
                # D_loss_real = mse_loss(mark_real, sdr.Q_calc_sdr_torch(clean, clean))

                # mark_fake = D(mix,G_estimation.detach())
                mark_fake = D(Encoder(clean, G_estimation)).mean()
                # mark_fake = D(clean,G_estimation)

                # D_loss_fake = mse_loss(mark_fake, sdr.Q_calc_sdr_torch(G_estimation, clean))
                # q_sl += sdr.Q_calc_sdr_torch(G_estimation, clean).mean().item()
                # pesq, pesq_mean = sdr.Q_calc_pesq(G_estimation, clean)
                # D_loss_fake = mse_loss(mark_fake, pesq)
                # q_sl += pesq_mean.item()

                D_loss = -mark_real + mark_fake

                D_loss.backward()
                dl += D_loss.item()
                mr += mark_real.mean().item()
                mf += mark_fake.mean().item()
                # print(dl,mr,mf, '%.8f' % (dl/d_len))

                opt_D.step()

                # train G
                if (step + 1) % 1 == 0:

                    g_len += 1
                    clean = torch.unsqueeze(clean, dim=1)
                    # sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)

                    if G_loss_type == 'sdr':
                        opt_G.zero_grad()
                        # G 的输出
                        G_estimation = G(mix)
                        g_mark_fake = D(clean, G_estimation)
                        G_loss = -g_mark_fake.mean()
                        # G_loss = beta * mse_loss(g_mark_fake, target)
                        # si_mse = utils.si_mse(G_estimation,clean)
                        si_mse = -sdr.batch_SDR_torch(G_estimation, clean)
                        g_loss = G_loss - lamda * si_mse
                        g_loss.backward()
                        gl += G_loss.item()
                        mse_l += si_mse.item()
                        opt_G.step()
                        # sdr_loss = - sdr.batch_SDR_torch(G_estimation, clean)
                        sdr_loss = -si_mse
                    elif G_loss_type == 'mse':
                        opt_G.zero_grad()
                        # G 的输出
                        G_estimation = G(mix)
                        # g_mark_fake = D(clean, G_estimation)
                        g_mark_fake = D(Encoder(clean, G_estimation)).mean()
                        # G_loss = beta * mse_loss(g_mark_fake, target)
                        G_loss = -1 * beta * g_mark_fake
                        mse = mse_loss(G_estimation, clean)
                        g_loss = G_loss + lamda * mse
                        g_loss.backward()
                        gl += g_loss.item()
                        mse_l += mse.item()
                        opt_G.step()

                        sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                    elif G_loss_type == 'L1':
                        opt_G.zero_grad()
                        G_estimation = G(mix)
                        # g_mark_fake = D(clean, G_estimation)
                        g_mark_fake = D(Encoder(clean, G_estimation)).mean()
                        # G_loss = beta * mse_loss(g_mark_fake, target)
                        G_loss = -1*beta * g_mark_fake
                        mse = L1_loss(G_estimation, clean)
                        g_loss = G_loss + lamda * mse
                        g_loss.backward()
                        gl += g_loss.item()
                        mse_l += mse.item()
                        opt_G.step()

                        sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                    else:
                        opt_G.zero_grad()
                        G_estimation = G(mix)
                        g_mark_fake = D(Encoder(clean, G_estimation)).mean()
                        # g_mark_fake = D(clean, G_estimation)
                        # G_loss = beta * mse_loss(g_mark_fake, target)
                        G_loss = -1 * beta * g_mark_fake

                        g_loss = G_loss
                        g_loss.backward()
                        gl += g_loss.item()

                        opt_G.step()

                        # sdr_loss = -sdr.calc_sdr_torch(G_estimation, clean).mean()
                        sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)

                    sl += sdr_loss.item()
            train_loss = sl / g_len

            val_sdr = validition(G, test_loader)

            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch + 1
                patient = 0
                utils.save_net(NETS_PATH, net=D, net_type='D', batch=batch, epoch=epoch, step=step)
                # if (epoch + 1) % 10 == 0:
                #     utils.save_net(NETS_PATH, net=G, net_type='G_10s', batch=batch, epoch=epoch, step=step, sdr_loss=sl/g_len, num=20)
                #
                utils.save_net(NETS_PATH, net=G, net_type='G', batch=batch, epoch=epoch, step=step)
            else:
                patient += 1

            if val_sdr < best_val_sdr:
                best_val_sdr = val_sdr
                utils.save_net(NETS_PATH, net=G, net_type='G_val', batch=batch, epoch=epoch, step=step,
                               sdr_loss=val_sdr,
                               num=20)

            if (epoch + 1) % 10 == 0:
                utils.save_net(NETS_PATH, net=G, net_type='G_10s', batch=batch, epoch=epoch, step=step,
                               sdr_loss=train_loss,
                               num=20)

            assert patient < 16, 'Early finished because of no improved! Best epoch {}'.format(best_epoch)

            # time.sleep(2)
            scheduler_D.step(train_loss)  # D针对real组的判断调整lr
            scheduler_G.step(train_loss)  # G针对fake组的判断调整lr

        time.sleep(2)
        # # 定期保存网络
        # 8.22 get the mean for a epoch
        print(
            # 'Epoch: ', epoch,
            '| D_loss: %.8f' % (dl / d_len),
            '| G_loss: %.8f' % (gl / g_len),
            '| real: %.8f' % (mr / d_len),
            '| fake: %.8f' % (mf / d_len),
            '| Q: %.8f' % (q_sl / d_len),
            '| sdr: %.8f' % (train_loss),
            '| val: %.8f' % val_sdr,
            '| mse: %.8f' % (mse_l / d_len),
        )
        time.sleep(2)
        pbar.close()
        # scheduler_D.step(train_loss)  # D针对real组的判断调整lr
        # scheduler_G.step(train_loss)  # G针对fake组的判断调整lr
        # scheduler_D_multi.step()  # D针对real组的判断调整lr
        # scheduler_G_multi.step()  # G针对fake组的判断调整lr

def train_WGAN(data_loader,patient=patient,G_loss_type='mse',lamda=Lamda,beta=Beta,batch=BATCH_SIZE):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # init G and D models
    G = conv_tasnet.TasNet(win=2)
    D = models.SNConv()

    # if cuda is available, use it.
    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    # optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR)
    # learning rate schedule
    scheduler_D = ReduceLROnPlateau(opt_D, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    scheduler_G = ReduceLROnPlateau(opt_G, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    # L1 loss function for generator as the norm constraint.
    L1_loss = nn.L1Loss()
    # load train and test data loader
    train_loader, test_loader = split_dataloader(batch_size=BATCH_SIZE, train_ratio=0.95,
                                                 dataset=data_loader)

    best_loss = 0
    patient_count = 0
    patient = 10
    best_epoch = 0
    best_val_sdr = 0
    val_sdr = 0
    for epoch in range(EPOCH):
        dl, gl, mr, mf, sl, mse_l = 0, 0, 0, 0, 0, 0
        q_sl = 0
        d_len = len(train_loader)
        g_len = 0
        pbar = tqdm(train_loader, desc='Epoch: {}'.format(epoch + 1), ncols=80)

        for step, (mix, clean) in enumerate(pbar):

            # if cuda is available, get the data in cuda.
            if torch.cuda.is_available():
                mix, clean = mix.cuda(), clean.cuda()

            # G estimation results
            G_estimation = G(mix).detach()

            # Get the encoder of the generator, and fix it.
            Encoder = models.Encoder(G.module.encoder)
            utils.set_requires_grad(Encoder)
            if torch.cuda.is_available(): Encoder.cuda()

            # train D model
            opt_D.zero_grad()
            mark_real = D(Encoder(clean, clean)).mean()
            mark_fake = D(Encoder(clean, G_estimation)).mean()
            D_loss = -mark_real + mark_fake
            D_loss.backward()
            dl += D_loss.item()
            mr += mark_real.mean().item()
            mf += mark_fake.mean().item()
            opt_D.step()

            # train G
            if (step + 1) % 1 == 0:

                g_len += 1
                clean = torch.unsqueeze(clean, dim=1)
                # sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)

                if G_loss_type == 'sdr':
                    opt_G.zero_grad()
                    # G 的输出
                    G_estimation = G(mix)
                    g_mark_fake = D(clean, G_estimation)
                    G_loss = -g_mark_fake.mean()
                    # G_loss = beta * mse_loss(g_mark_fake, target)
                    # si_mse = utils.si_mse(G_estimation,clean)
                    si_mse = -sdr.batch_SDR_torch(G_estimation, clean)
                    g_loss = G_loss - lamda * si_mse
                    g_loss.backward()
                    gl += G_loss.item()
                    mse_l += si_mse.item()
                    opt_G.step()
                    # sdr_loss = - sdr.batch_SDR_torch(G_estimation, clean)
                    sdr_loss = -si_mse
                elif G_loss_type == 'mse':
                    opt_G.zero_grad()
                    # G 的输出
                    G_estimation = G(mix)
                    # g_mark_fake = D(clean, G_estimation)
                    g_mark_fake = D(Encoder(clean, G_estimation)).mean()
                    # G_loss = beta * mse_loss(g_mark_fake, target)
                    G_loss = -1 * beta * g_mark_fake
                    mse = mse_loss(G_estimation, clean)
                    g_loss = G_loss + lamda * mse
                    g_loss.backward()
                    gl += g_loss.item()
                    mse_l += mse.item()
                    opt_G.step()

                    sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                elif G_loss_type == 'L1':
                    opt_G.zero_grad()
                    G_estimation = G(mix)
                    # g_mark_fake = D(clean, G_estimation)
                    g_mark_fake = D(Encoder(clean, G_estimation)).mean()
                    # G_loss = beta * mse_loss(g_mark_fake, target)
                    G_loss = -1 * beta * g_mark_fake
                    mse = L1_loss(G_estimation, clean)
                    g_loss = G_loss + lamda * mse
                    g_loss.backward()
                    gl += g_loss.item()
                    mse_l += mse.item()
                    opt_G.step()

                    sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                else:
                    opt_G.zero_grad()
                    G_estimation = G(mix)
                    g_mark_fake = D(Encoder(clean, G_estimation)).mean()
                    # g_mark_fake = D(clean, G_estimation)
                    # G_loss = beta * mse_loss(g_mark_fake, target)
                    G_loss = -1 * beta * g_mark_fake

                    g_loss = G_loss
                    g_loss.backward()
                    gl += g_loss.item()

                    opt_G.step()

                    # sdr_loss = -sdr.calc_sdr_torch(G_estimation, clean).mean()
                    sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)

                sl += sdr_loss.item()
        train_loss = sl / g_len

        val_sdr = validition(G, test_loader)

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1
            patient = 0
            utils.save_net(NETS_PATH, net=D, net_type='D', batch=batch, epoch=epoch, step=step)
            # if (epoch + 1) % 10 == 0:
            #     utils.save_net(NETS_PATH, net=G, net_type='G_10s', batch=batch, epoch=epoch, step=step, sdr_loss=sl/g_len, num=20)
            #
            utils.save_net(NETS_PATH, net=G, net_type='G', batch=batch, epoch=epoch, step=step)
        else:
            patient += 1

        if val_sdr < best_val_sdr:
            best_val_sdr = val_sdr
            utils.save_net(NETS_PATH, net=G, net_type='G_val', batch=batch, epoch=epoch, step=step,
                           sdr_loss=val_sdr,
                           num=20)

        if (epoch + 1) % 10 == 0:
            utils.save_net(NETS_PATH, net=G, net_type='G_10s', batch=batch, epoch=epoch, step=step,
                           sdr_loss=train_loss,
                           num=20)

        assert patient < 16, 'Early finished because of no improved! Best epoch {}'.format(best_epoch)

        # time.sleep(2)
        scheduler_D.step(train_loss)  # D针对real组的判断调整lr
        scheduler_G.step(train_loss)  # G针对fake组的判断调整lr

        time.sleep(2)
        # # 定期保存网络
        # 8.22 get the mean for a epoch
        print(
            # 'Epoch: ', epoch,
            '| D_loss: %.8f' % (dl / d_len),
            '| G_loss: %.8f' % (gl / g_len),
            '| real: %.8f' % (mr / d_len),
            '| fake: %.8f' % (mf / d_len),
            '| Q: %.8f' % (q_sl / d_len),
            '| sdr: %.8f' % (train_loss),
            '| val: %.8f' % val_sdr,
            '| mse: %.8f' % (mse_l / d_len),
        )
        time.sleep(2)
        pbar.close()
        # scheduler_D.step(train_loss)  # D针对real组的判断调整lr
        # scheduler_G.step(train_loss)  # G针对fake组的判断调整lr
        # scheduler_D_multi.step()  # D针对real组的判断调整lr
        # scheduler_G_multi.step()  # G针对fake组的判断调整lr

def train_Metric_GAN(data_loader,D_sdr=False,G_loss_type='',wgan=False,lamda=Lamda,beta=Beta,batch=BATCH_SIZE):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # 网络初始化
    # D = conv_tasnet.D(sdr=D_sdr,wgan=wgan,model_type='metric')

    G = conv_tasnet.TasNet(win=2)
    # Encoder = models.Encoder(G.encoder)
    # D = models.SNConv(Encoder)
    D = models.SNConv()

    if torch.cuda.is_available():
        D = torch.nn.DataParallel(D, device_ids=device_ids)  # 声明所有可用设备
        G = torch.nn.DataParallel(G, device_ids=device_ids)  # 声明所有可用设备
        D.cuda()
        G.cuda()

    # D = get_model(r'./nets/D', r'D-batch70-epoch-23-step-400-sdr-0.pkl')
    # G = get_model(r'./nets/G', r'G-batch70-epoch-23-step-400-sdr-0.pkl')

    # 优化器
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR)
    # 动态调整学习率
    scheduler_D = ReduceLROnPlateau(opt_D, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    scheduler_G = ReduceLROnPlateau(opt_G, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    scheduler_D_multi = MultiStepLR(opt_D, milestones=[30], gamma=0.1)
    scheduler_G_multi = MultiStepLR(opt_G, milestones=[30], gamma=0.1)
    # scheduler_D = MultiStepLR(opt_D, milestones=[200,400],gamma=0.1)
    # scheduler_G = MultiStepLR(opt_G, milestones=[200,400],gamma=0.1)

    mse_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()

    train_loader, test_loader = split_dataloader(batch_size=BATCH_SIZE, train_ratio=0.95, dataset=data_loader)

    i = 0
    best_loss = 0
    patient = 0
    best_epoch = 0
    best_val_sdr = 0
    val_sdr = 0
    for epoch in range(i,EPOCH+i):
        dl, gl, mr, mf, sl,mse_l = 0,0,0,0,0,0
        q_sl = 0
        d_len = len(train_loader)
        g_len = 0
        pbar = tqdm(train_loader, desc='Epoch: {}'.format(epoch + 1), ncols=80)

        if epoch < 0:
            for step, (mix, clean) in enumerate(pbar):
                if torch.cuda.is_available():
                    mix, clean = mix.cuda(), clean.cuda()
                opt_G.zero_grad()
                # G 的输出
                G_estimation = G(mix)
                clean = torch.unsqueeze(clean, dim=1)

                mse = mse_loss(G_estimation, clean)
                g_loss = mse
                g_loss.backward()
                gl += g_loss.item()
                mse_l += mse.item()
                opt_G.step()

                sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                sl += sdr_loss.item()
            train_loss = sl / d_len
        else:

            for step, (mix, clean) in enumerate(pbar):

                target = torch.ones(mix.shape[0],1)
                if torch.cuda.is_available():
                    mix,clean = mix.cuda(),clean.cuda()
                    target = target.cuda()

                # print(step)
                # G 的输出
                G_estimation = G(mix).detach()

                Encoder = models.Encoder(G.module.encoder)
                utils.set_requires_grad(Encoder)
                Encoder = torch.nn.DataParallel(Encoder, device_ids=device_ids)  # 声明所有可用设备
                Encoder.cuda()

                # train D
                opt_D.zero_grad()
                # print('d:',clean.shape)
                mark_real = D(Encoder(clean,clean))
                # mark_real = D(clean,clean)


                D_loss_real = mse_loss(mark_real, target)
                # D_loss_real = mse_loss(mark_real, sdr.Q_calc_sdr_torch(clean, clean))

                # mark_fake = D(mix,G_estimation.detach())
                mark_fake = D(Encoder(clean,G_estimation))
                # mark_fake = D(clean,G_estimation)

                D_loss_fake = mse_loss(mark_fake, sdr.Q_calc_sdr_torch(G_estimation, clean))
                q_sl += sdr.Q_calc_sdr_torch(G_estimation, clean).mean().item()
                # pesq, pesq_mean = sdr.Q_calc_pesq(G_estimation, clean)
                # D_loss_fake = mse_loss(mark_fake, pesq)
                # q_sl += pesq_mean.item()

                D_loss = (D_loss_real + D_loss_fake)

                D_loss.backward()
                dl += D_loss.item()
                mr += mark_real.mean().item()
                mf += mark_fake.mean().item()
                # print(dl,mr,mf, '%.8f' % (dl/d_len))

                opt_D.step()

                # train G
                if (step + 1) % 1 == 0:

                    g_len += 1
                    clean = torch.unsqueeze(clean, dim=1)
                    # sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)

                    if G_loss_type == 'sdr':
                        opt_G.zero_grad()
                        # G 的输出
                        G_estimation = G(mix)
                        g_mark_fake = D(clean, G_estimation)
                        G_loss = -g_mark_fake.mean()
                        # G_loss = beta * mse_loss(g_mark_fake, target)
                        # si_mse = utils.si_mse(G_estimation,clean)
                        si_mse = -sdr.batch_SDR_torch(G_estimation, clean)
                        g_loss = G_loss - lamda * si_mse
                        g_loss.backward()
                        gl += G_loss.item()
                        mse_l += si_mse.item()
                        opt_G.step()
                        # sdr_loss = - sdr.batch_SDR_torch(G_estimation, clean)
                        sdr_loss = -si_mse
                    elif G_loss_type == 'mse':
                        opt_G.zero_grad()
                        # G 的输出
                        G_estimation = G(mix)
                        # g_mark_fake = D(clean, G_estimation)
                        g_mark_fake = D(Encoder(clean, G_estimation))
                        G_loss = beta * mse_loss(g_mark_fake, target)
                        mse = mse_loss(G_estimation, clean)
                        g_loss = G_loss + lamda * mse
                        g_loss.backward()
                        gl += g_loss.item()
                        mse_l += mse.item()
                        opt_G.step()

                        sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                    elif G_loss_type == 'L1':
                        opt_G.zero_grad()
                        G_estimation = G(mix)
                        # g_mark_fake = D(clean, G_estimation)
                        g_mark_fake = D(Encoder(clean, G_estimation))
                        G_loss = beta * mse_loss(g_mark_fake, target)
                        mse = L1_loss(G_estimation, clean)
                        g_loss = G_loss + lamda * mse
                        g_loss.backward()
                        gl += g_loss.item()
                        mse_l += mse.item()
                        opt_G.step()

                        sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)
                    else:
                        opt_G.zero_grad()
                        G_estimation = G(mix)
                        g_mark_fake = D(Encoder(clean,G_estimation))
                        # g_mark_fake = D(clean, G_estimation)
                        G_loss = beta * mse_loss(g_mark_fake, target)

                        g_loss = G_loss
                        g_loss.backward()
                        gl += g_loss.item()

                        opt_G.step()

                        # sdr_loss = -sdr.calc_sdr_torch(G_estimation, clean).mean()
                        sdr_loss = -sdr.batch_SDR_torch(G_estimation, clean)

                    sl += sdr_loss.item()
            train_loss = sl/g_len

            val_sdr = validition(G, test_loader)

            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch + 1
                patient = 0
                utils.save_net(NETS_PATH, net=D, net_type='D', batch=batch, epoch=epoch, step=step)
                # if (epoch + 1) % 10 == 0:
                #     utils.save_net(NETS_PATH, net=G, net_type='G_10s', batch=batch, epoch=epoch, step=step, sdr_loss=sl/g_len, num=20)
                #
                utils.save_net(NETS_PATH, net=G, net_type='G', batch=batch, epoch=epoch, step=step)
            else:
                patient += 1

            if val_sdr < best_val_sdr:
                best_val_sdr = val_sdr
                utils.save_net(NETS_PATH, net=G, net_type='G_val', batch=batch, epoch=epoch, step=step, sdr_loss=val_sdr,
                               num=20)

            if (epoch + 1) % 10 == 0:
                utils.save_net(NETS_PATH, net=G, net_type='G_10s', batch=batch, epoch=epoch, step=step, sdr_loss=train_loss,
                               num=20)

            assert patient < 16, 'Early finished because of no improved! Best epoch {}'.format(best_epoch)

            # time.sleep(2)
            scheduler_D.step(train_loss)  # D针对real组的判断调整lr
            scheduler_G.step(train_loss)  # G针对fake组的判断调整lr

        time.sleep(2)
        # # 定期保存网络
        # 8.22 get the mean for a epoch
        print(
            # 'Epoch: ', epoch,
              '| D_loss: %.8f' % (dl/d_len),
              '| G_loss: %.8f' % (gl/g_len),
              '| real: %.8f' % (mr/d_len),
              '| fake: %.8f' % (mf/d_len),
              '| Q: %.8f' % (q_sl/d_len),
              '| sdr: %.8f' % (train_loss),
              '| val: %.8f' % val_sdr,
              '| mse: %.8f' % (mse_l/d_len),
              )
        # scheduler_D.step(train_loss)  # D针对real组的判断调整lr
        # scheduler_G.step(train_loss)  # G针对fake组的判断调整lr
        # scheduler_D_multi.step()  # D针对real组的判断调整lr
        # scheduler_G_multi.step()  # G针对fake组的判断调整lr

        time.sleep(2)
        pbar.close()

def validition(model,test_loader,ratio=0.7):
    with torch.no_grad():
        ls = 0
        iters = len(test_loader)
        for step, (mix, clean) in enumerate(test_loader):

            if torch.cuda.is_available():
                mix, clean = mix.cuda(), clean.cuda()
            clean = torch.unsqueeze(clean, dim=1)
            G_estimation = model(mix)

            loss = -sdr.batch_SDR_torch(G_estimation, clean)
            # loss = -sdr.calc_sdr_torch(G_estimation, clean).mean()
            ls += loss.item()

        return ls/iters

if __name__ == "__main__":
    # 数据集
    # mix_path = r'./data/train/mixture'
    # clean_path = r'./data/train/clean'


    mix_path = r'./data/train/16k_mixture_28spk_all'
    # mix_path_2 = r'./data/train/16k_mixture_28spk_augmentation'
    clean_path = r'./data/train/16k_clean_28spk_all'

    train_data = AudioDataset(mix_path_1=mix_path,clean_path_1=clean_path)
    # train_data = AudioDataset(mix_path_1=mix_path,clean_path_1=clean_path,mix_path_2=mix_path_2,clean_path_2=clean_path)
    # data_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=8)

    # train_GAN(train_loader)
    # train_TasNet(train_loader)
    # train_WGAN_GP(train_data)
    train_Metric_GAN(train_data)