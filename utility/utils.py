import numpy as np
import glob
import os
import torch

def new_filedir(file_path):
    # 如果操作路径不存在，则创建它
    if not os.path.exists(file_path):
        # print(file_path)
        os.makedirs(file_path)

# 模型梯度设置
def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets(list): networks
        requires_grad(bool): True or False
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

# 目录下文件按时间排序
def get_file_list(file_path):
    # dir_list = os.listdir(file_path)
    dir_list = os.listdir(file_path)
    if not dir_list:
        return []
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list,key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list

# 加载最新模型
def get_last_model(nets_path, index=-1):
    model_path = os.path.join(nets_path,get_file_list(nets_path)[index])
    print('Loading {}'.format(model_path))
    model = torch.load(model_path)
    return model

def si_mse(estimation, target):
    """
    batch-wise SI-MSE on pytorch Variables.
    estimation: (batch, nspk, nsample)
    origin: (batch, nspk, nsample)
    mask: optional, (batch, nspk, nsample), binary
    """

    # origin_power = torch.pow(target, 2).sum(2, keepdim=True) # (batch, 1, 1)
    #
    # scale = origin_power / (torch.sum(target * estimation, 2, keepdim=True)  + 1e-8) # (batch, 1, 1)
    #
    # # est_true = scale * target  # (batch, nsample)
    # # est_res = estimation - est_true  # (batch, nsample)
    # est = target - scale * estimation  # (batch, nsample)
    #
    # # true_power = torch.pow(est_true, 2).sum(1, keepdim=True)
    # # res_power = torch.pow(est_res, 2).sum(1, keepdim=True)
    # res_power = torch.pow(est, 2).mean()
    #
    # return res_power  # a item

    # 9.2
    origin_power = torch.pow(target, 2).sum(2, keepdim=True) + 1e-8  # (batch, 1, 1)

    scale = torch.sum(target * estimation, 2, keepdim=True) / origin_power  # (batch, 1, 1)

    # est_true = scale * target  # (batch, 1, nsample)
    est_true = target  # (batch, 1, nsample) use target source as the aim signal. SNR loss
    est_res = estimation - est_true  # (batch, 1, nsample)

    true_power = torch.pow(est_true, 2).sum(2, keepdim=True) + 1e-8
    res_power = torch.pow(est_res, 2).sum(2, keepdim=True)

    return (res_power / true_power).mean()


# 保存网络模型，定期更新删去过旧的模型
def save_net(file_path,net=None,net_type='',epoch=0,step=0,num=10,sdr_loss=0,batch=0):
    # 确定操作路径对应D还是G
    file_path = os.path.join(file_path,net_type)

    # 如果操作路径不存在，则创建它
    new_filedir(file_path)

    # glob_path = os.path.join(file_path, '*')
    # 读取路径下所有文件
    # filenames = glob.glob(glob_path)
    filenames = get_file_list(file_path)  # 按文件时间顺序排序

    if len(filenames) >= num:
        os.remove(os.path.join(file_path,filenames[0]))

    save_file = '{}-batch{}-epoch-{}-step-{}-sdr-{}.pkl'.format(net_type,batch,epoch,step,sdr_loss)
    save_path = os.path.join(file_path,save_file)
    # 保存整个模型
    torch.save(net,save_path)

# 加载模型
def get_model(nets_path,model_name,map_location=False):
    model_path = os.path.join(nets_path,model_name)
    if map_location:
        model = torch.load(model_path,map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
    return model


def pad_signal(input):

    # input is the waveforms: (B, T) or (B, 1, T)
    # reshape and padding
    if input.dim() not in [2, 3]:
        raise RuntimeError("Input can only be 2 or 3 dimensional.")

    if input.dim() == 2:
        input = input.unsqueeze(1)

    return input