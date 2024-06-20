import matplotlib
import datetime

from models.Fed import FedAvg
#from DP1 import cal_sensitivity_client, Gaussian_client
from models.test import test_img
from utility.fodername import create_dynamic_folder_name, create_dynamic_folder, create_dynamic_norm_folder
from parameters import get_args
from models.Fed_sampleclient import FedAvg_sampleClient_delta

matplotlib.use('Agg')
import numpy as np
import torch
import time
import random
import logging
import copy
import os



# Import different files required for loading dataset, model, testing, training
from utility.LoadSplit import Load_Dataset, Load_Model
# from utility.options import args_parser
from models.Update import train_client, test_client, train_client_with_last_model_DKD, isBetter

torch.manual_seed(0)

#有w的是字典格式的，有net的是网络格式的。如：w_glob;net_glob

def subtract(params_a, params_b):
    w = copy.deepcopy(params_a)
    for k in w.keys():
        w[k] -= params_b[k]
    return w
def add(params_a, params_b):
    w = copy.deepcopy(params_a)
    for k in w.keys():
        w[k] += params_b[k].to(w[k].dtype)
    return w

def cal_clip(w):
    norm = 0.0
    for name in w.keys():
        norm += pow(w[name].float().norm(2), 2)
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
    return total_norm[0]


def clip_and_add_noise_sigma(w):
    l2_norm = cal_clip(w)
    clip=0.5 * l2_norm
    print("clip:" +str(clip))
    with torch.no_grad():
        for name in w.keys():
            noise = torch.FloatTensor(w[name].shape).normal_(0, 2 * 1 /np.sqrt(args.num_users*args.frac))
            noise = noise.cpu().numpy()
            noise = torch.from_numpy(noise).type(torch.FloatTensor).to(w[name].device)
            w[name] = w[name].float() * min(1, 1 / torch.norm(w[name].float(), 2))
            w[name] = w[name].add_(noise)
    return w

def clip_and_add_noise_gaussian(w):
    sensitivity = cal_sensitivity_client(0.2, args.num_users)
    dp_delta = 1/args.num_users
    with torch.no_grad():
        for name in w.keys():
            noise = Gaussian_client(epsilon = args.dp_epsilon/args.local_ep, delta =dp_delta,
                                    sensitivity = sensitivity, participated_client_size = args.num_users * args.frac, communication_round = args.epochs, size = w[name].shape)
            noise = torch.from_numpy(noise).type(torch.FloatTensor).to(w[name].device)
            w[name] = w[name].float() * min(1, 0.2 / torch.norm(w[name].float(), 2))
            w[name] += noise
    return w



if __name__ == '__main__':
    
    # Initialize argument dictionary
    # 创建动态文件夹名字
    start = time.time()
    args = get_args()
    file_name = create_dynamic_folder_name(args)
    file_path = create_dynamic_folder(args)
    file_norm = create_dynamic_norm_folder(args)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(file_norm):
        os.makedirs(file_norm)
    # 将文件路径和文件名合并起来
    file_path_name=os.path.join(file_path, file_name)
    file_norm_name=os.path.join(file_norm, file_name)

    a=0
    # Setting the device - GPU or CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # Load the training and testing datasets
    dataset_train, dataset_test, dict_users = Load_Dataset(args=args)
    #dataset_train.data.numel()=数据集的样本大小*像素*像素
    # Initialize Global Server Model
    net_glob = Load_Model(args=args)
    # print(net_glob)   
    net_glob.train()
    # Print name of the architecture - 'MobileNet or ResNet or NewNet'
    print(args.model)

    # copy weights
    w_glob = net_glob.state_dict()

    #splitting user data into training and testing parts
    train_data_users = {}
    test_data_users = {}
    for i in range(args.num_users):
        if args.iid == 'dir':
            dict_users[i] = list(dict_users[i])
        train_data_users[i] = list(random.sample(dict_users[i],int(args.split_ratio*len(dict_users[i]))))
        test_data_users[i] = list(set(dict_users[i])-set(train_data_users[i]))

    # exit()
    # local models for each client
    delta_locals_nets={}
    local_nets = {}
    #存放本地模型对应的上轮本地模型
    w_last_locals = {}
    last_local_nets = {}
    #定义选择过的客户端ID列表
    selected_clients = [0]*args.num_users
    norm=[0]*args.num_users
    # Start training
    logging.info("Training")
    start = time.time()
    #进入全局通信轮数
    for iter in range(args.epochs):
        print('Round {}'.format(iter))
        logging.info("---------Round {}---------".format(iter))
        loss_locals,w_select_locals = [],[]
        w_locals={}
        w_delta_locals = {}
        # 选择参与客户端
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()

        for i in idxs_users:
            local_nets[i] = Load_Model(args=args)
            local_nets[i].train()
            if iter == 0:
                local_nets[i].load_state_dict(w_glob)
            else:
                w_locals[i] = copy.deepcopy(w_glob)
                local_nets[i].load_state_dict(w_locals[i])

        for idx in idxs_users:
            #如果这轮被选择，但是之前没被选择
            if selected_clients[idx] == 0:
                w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
            else:
                w, loss = train_client_with_last_model_DKD(args,dataset_train,train_data_users[idx],net = local_nets[idx], last_net = last_local_nets[idx])

            print(loss)
            loss_locals.append(copy.deepcopy(loss))
            #将上轮的模型保存，当作之后的个性化模型
            w_last_locals[idx] = w
            #更新训练后的模型
            w_locals[idx] = w
            w_select_locals.append(w)
            #计算更新量
            w_delta_locals[idx] = subtract(w_locals[idx],w_glob)
            norm_test = cal_clip(w_delta_locals[idx])
            norm[idx]=norm_test
            # # 如果加DP，裁剪并加噪，然后计算准确率
            # if args.useDP:
            #     w_delta_locals[idx] = clip_and_add_noise_sigma(w_delta_locals[idx])

        #如果客户端被选择过
        #将上轮本地模型存下来，并加载到网络中，并将列表中对应值设为1，默认是0（没被选择参与过训练）
        for i in idxs_users:
            last_local_nets[i] = Load_Model(args=args)
            last_local_nets[i].train()
            last_local_nets[i].load_state_dict(w_last_locals[i])
            selected_clients[i] = 1


        # update global weights
        #上传的是更新量,用上一轮的全局模型加上更新量聚合之后的全局模型，得到新一轮的w_glob
        #w_delta_glob = FedAvg_sampleClient_delta(w_delta_locals, idxs_users)
        #w_glob = add(w_glob,w_delta_glob)
        w_glob=FedAvg(w_select_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        #测试全局模型精度
        # net_glob.eval()
        s_global = 0
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        s_global, loss_test = test_img(net_glob, dataset_test, args)
        # s_global = 0
        # for i in idxs_users:
        #     acc_train_glo, loss_train_glo = test_client(args, dataset_train, train_data_users[i], net_glob)
        #     acc_test_glo, loss_test_glo = test_client(args, dataset_train, test_data_users[i], net_glob)
        #     s_global += acc_test_glo
        # s_global /= m


        # store testing and training accuracies of the model before global aggregation
        #个性化联邦学习精度
        s0 = 0
        s = 0
        for i in idxs_users:
            logging.info("Client {}:".format(i))
            acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
            acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
            s0 += acc_train
            s += acc_test
        s0 /= m
        s /= m
        a = isBetter(s, a)
        iter_loss = sum(loss_locals) / len(loss_locals)

        with open(file_norm_name, 'a') as f:
            f.write("round " + str(iter) + "  ")
            for item in norm:
                # 直接检查是否item是单个数字（不可迭代），然后转换为字符串
                if isinstance(item, (int, float, np.float32, np.float64)):
                    f.write(str(item) + '\n')
                else:
                    # 如果item是可迭代的（如列表或数组），则正常处理
                    f.write(' '.join(map(str, item)) + '\n')


        with open(file_path_name, 'a') as f:
            f.write("round " + str(iter) + "  ")
            f.write('loss: {:.6f}  '.format(iter_loss))
            f.write('train_Accuracy: {:.4f} '.format(s0))
            f.write('test_Accuracy: {:.4f} '.format(s))
            f.write('best_Accuracy: {:.4f} '.format(a))
            if iter == 99:
                f.write('global_Accuracy: {:.4f} '.format(s_global))
            else:
                f.write('global_Accuracy: {:.4f} \n'.format(s_global))

    end = time.time()
    stime =end-start
    with open(file_path_name, 'a') as f:
        f.write("time " + str(stime) + "  ")







