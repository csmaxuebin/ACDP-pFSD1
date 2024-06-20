import copy

import numpy as np
import torch

#得到模型更新量
def subtract(params_a, params_b):
    w = copy.deepcopy(params_a)
    for k in w.keys():
        w[k] -= params_b[k].to(w[k].dtype)
    return w

def add(params_a, params_b):
    w = copy.deepcopy(params_a)
    for k in w.keys():
        w[k] += params_b[k].to(w[k].dtype)
    return w

#计算整个模型字典的l2范数
def cal_clip(w):
    norm = 0.0
    for name in w.keys():
        norm += pow(w[name].float().norm(2), 2)
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
    return total_norm[0]

#计算l2范数和b-A类裁剪对比方法
def cal_percent_clip(w,clip):
    norm = 0.0
    for name in w.keys():
        norm += pow(w[name].float().norm(2), 2)
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
    w_delta_vector = torch.cat([param.view(-1).abs() for param in w.values()])
    count = torch.sum(w_delta_vector < clip).item()
    total = len(w_delta_vector)
    b = count / total  # 计算的是b
    return total_norm[0],b


#计算模型字典的百分位数
def cal_percentile(w,percent):
    per_norm=0.0
    norm=cal_clip(w)
    per_norm = norm * percent
    return per_norm

#裁剪并加噪
def clip_and_add_noise_our(w,c,noise_multiplier,args):
    l2_norm = cal_clip(w)
    with torch.no_grad():
        for name in w.keys():
            noise = torch.FloatTensor(w[name].shape).normal_(0, noise_multiplier * c /np.sqrt(args.num_users*args.frac))
            noise = noise.cpu().numpy()
            noise = torch.from_numpy(noise).type(torch.FloatTensor).to(w[name].device)
            w[name] = w[name].float() * min(1, c / l2_norm)
            w[name] = w[name].add_(noise)
    return w

#计算l2范数和b-A类裁剪对比方法
def clip_and_add_noise_per(w,c,noise_multiplier,args):
    l2_norm,b = cal_percent_clip(w,c)
    with torch.no_grad():
        for name in w.keys():
            std=(noise_multiplier * c /np.sqrt(args.num_users*args.frac)).item()
            noise = torch.FloatTensor(w[name].shape).normal_(0, std)
            noise = noise.cpu().numpy()
            noise = torch.from_numpy(noise).type(torch.FloatTensor).to(w[name].device)
            w[name] = w[name].float() * min(1, c / l2_norm)
            w[name] = w[name].add_(noise)
    return w,b



#计算得到自适应裁剪阈值
# def Adaclip(c,args):
#     max_clip_norm = c * args.decay_lamda_2
#     # print("c:" + str(max_clip_norm))
#     return max_clip_norm

#微调裁剪阈值
def Adaclip_tune1(c,args,iter):
    # 复制衰减得到的裁剪阈值
    max_clip_norm = c / pow(iter+1, args.decay_lamda_1)
    # 如果裁剪阈值小于某个百分位数对应的值，裁剪阈值更新，因为损失信息太多了
    #max_clip_norm = max(max_clip_norm1, max_clip_norm2)
    return max_clip_norm

def Adaclip_tune2(c,args,iter):
    # if iter == 0:
    #     max_clip_norm =args.c
    # 复制衰减得到的裁剪阈值
    max_clip_norm = c * args.decay_lamda_2
    return max_clip_norm