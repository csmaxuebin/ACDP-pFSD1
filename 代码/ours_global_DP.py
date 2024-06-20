import matplotlib
import datetime
from collections import deque

#from DP.private_trainer import get_sigma_or_epsilon, clip_and_add_noise
from DP.DP_clip import subtract, clip_and_add_noise_our, add, Adaclip_tune1, Adaclip_tune2, cal_clip,clip_and_add_noise_per
from DP.private_trainer import get_sigma_or_epsilon
from ddpg import DDPG
from PPO import PPO
from models.test import test_img
from utility.fodername import create_dynamic_folder_name, create_dynamic_DP_folder, create_dynamic_norm_folder, create_dynamic_RL_folder
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

def middle_60_percent_average(arr):
    arr_sorted = sorted(arr)  # 对数组排序
    n = len(arr_sorted)  # 获取排序后数组的长度
    start = int(n * 0.2)  # 计算起始索引
    end = int(n * 0.8)  # 计算结束索引
    middle_60_percent = arr_sorted[start:end]  # 选取中间的60%
    return sum(middle_60_percent) / len(middle_60_percent)  # 计算并返回平均值


def cosine_similarity(params_a_dict, params_b_dict):
    # 将参数字典中的参数展平成向量
    params_a = torch.cat([p.view(-1) for p in params_a_dict.values()])
    params_b = torch.cat([p.view(-1) for p in params_b_dict.values()])

    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(params_a.unsqueeze(0), params_b.unsqueeze(0), dim=1)

    return similarity.item()

#有w的是字典格式的，有net的是网络格式的。如：w_glob;net_glob

if __name__ == '__main__':
    
    # Initialize argument dictionary
    # 创建动态文件夹名字
    args = get_args()
    file_name = create_dynamic_folder_name(args)
    file_path = create_dynamic_DP_folder(args)
    file_norm_path = create_dynamic_norm_folder(args)
    file_RL_path = create_dynamic_RL_folder(args)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(file_norm_path):
        os.makedirs(file_norm_path)
    if not os.path.exists(file_RL_path):
        os.makedirs(file_RL_path)
    file_path_name = os.path.join(file_path, file_name)
    #file_norm_name = os.path.join(file_norm_path, file_name)
    file_RL_name = os.path.join(file_RL_path, file_name)
    a=0

    # Setting the device - GPU or CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # Load the training and testing datasets
    dataset_train, dataset_test, dict_users = Load_Dataset(args=args)

    # Initialize Global Server Model
    net_glob = Load_Model(args=args)
    # print(net_glob)   
    net_glob.train()
    # Print name of the architecture - 'MobileNet or ResNet or NewNet'
    print(args.model)

    # copy weights
    w_glob = net_glob.state_dict()

    # 计算epsilon,noise_multiplier
    #epsilon, noise_multiplier = get_sigma_or_epsilon(iter=args.epochs, args=args, filename=file_path_name)
    #print("epsilon:"+str(epsilon))


    #splitting user data into training and testing parts
    train_data_users = {}
    test_data_users = {}
    for i in range(args.num_users):
        if args.iid=='dir':
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
    max_clip_norm=args.c


    if args.clipmethod=="ppo":
        ppo = PPO(2, 1, args.device)
        empty_recoder = []
        ddpg_local_loss, ddpg_local_acc, ddpg_local_b= copy.deepcopy(empty_recoder), copy.deepcopy(empty_recoder),copy.deepcopy(empty_recoder)
        local_action0, local_action1 = copy.deepcopy(empty_recoder), copy.deepcopy(empty_recoder)  # local action record
        reward, cost = copy.deepcopy(empty_recoder), copy.deepcopy(empty_recoder)  # local reward record and cost record
        learning_rate, dp_clip = copy.deepcopy(empty_recoder), copy.deepcopy(
            empty_recoder)  # local hypermater record, conputed from actions
        actor_loss, critic_loss, cost_loss = copy.deepcopy(empty_recoder), copy.deepcopy(empty_recoder), copy.deepcopy(
            empty_recoder)  # local ddpg train loss record
        xi1, xi2, xi3 = 1, 1, 1
        base_learning_rate = 10
        base_dp_clip = 2




    # Start training
    logging.info("Training")
    start = time.time()
    #进入全局通信轮数
    for iter in range(args.epochs):
        print('============ Round {} ============'.format(iter))
        with open(file_path_name, 'a') as f:
            f.write('Round {}'.format(iter))
        loss_locals = []
        b_locals = []
        global_acc_test=[]
        global_loss_test=[]
        w_locals={}
        w_delta_locals = {}
        # 选择参与客户端
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()

        #加载模型
        for i in idxs_users:
            local_nets[i] = Load_Model(args=args)
            local_nets[i].train()
            if iter == 0:
                local_nets[i].load_state_dict(w_glob)
            else:
                w_locals[i] = copy.deepcopy(w_glob)
                local_nets[i].load_state_dict(w_locals[i])


        #ppo算法全程
        if args.clipmethod == "ppo":
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
            }
            #初始化，如果是第一轮
            if iter == 0:
                max_clip_norm = args.c
                #选择客户端
                for idx in idxs_users:
                    # 如果这轮被选择，但是之前没被选择
                    if selected_clients[idx] == 0:
                        w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                    else:
                        w, loss = train_client_with_last_model_DKD(args, dataset_train, train_data_users[idx],
                                                                   net=local_nets[idx], last_net=last_local_nets[idx])
                    # 将上轮的模型保存，当作之后的个性化模型
                    w_last_locals[idx] = w
                    # 更新训练后的模型
                    w_locals[idx] = w
                    # 计算更新量
                    w_delta_locals[idx] = subtract(w_locals[idx], w_glob)
                    norm = cal_clip(w_delta_locals[idx])
                    print("norm:" + str(norm))
                    #加DP，裁剪并加噪，然后计算准确率
                    w_delta_locals[idx], b = clip_and_add_noise_per(w_delta_locals[idx], max_clip_norm,
                                                                    args.noise_multiplier,
                                                                    args)
                    b_locals.append(b)
                    print(loss)
                    loss_locals.append(copy.deepcopy(loss))
                b_var = np.var(b_locals)
                ddpg_local_b.append(b_var)
                # 如果客户端被选择过
                # 将上轮本地模型存下来，并加载到网络中，并将列表中对应值设为1，默认是0（没被选择参与过训练）
                for i in idxs_users:
                    last_local_nets[i] = Load_Model(args=args)
                    last_local_nets[i].train()
                    last_local_nets[i].load_state_dict(w_last_locals[i])
                    selected_clients[i] = 1

                    # update global weights
                    # 上传的是更新量,用上一轮的全局模型加上更新量聚合之后的全局模型，得到新一轮的w_glob
                w_delta_glob = FedAvg_sampleClient_delta(w_delta_locals, idxs_users)
                w_glob = add(w_glob, w_delta_glob)
                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)
                # 计算全局模型准确率和loss，下一轮状态
                gs = 0
                gl = 0
                for i in idxs_users:
                    acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
                    s_global, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    gs+=s_global
                    gl+=loss_test
                gs/=m
                #     global_acc_test.append(s_global)
                #     global_loss_test.append(loss_test)
                # gs=middle_60_percent_average(global_acc_test)
                # gl=middle_60_percent_average(global_loss_test)
                #计算本地模型精度
                la=ll=s0=s=0
                for i in idxs_users:
                    acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
                    # loss_test[i][0]
                    s0+=acc_train
                    s+=acc_test
                    global_acc_test.append(acc_test)
                    global_loss_test.append(loss_test)
                la = middle_60_percent_average(global_acc_test)
                ll = middle_60_percent_average(global_loss_test)
                # 将loss和acc保存下来
                ddpg_local_loss.append(la)
                ddpg_local_acc.append(ll)
                s0/=m
                s/=m
                a = isBetter(la, a)
                iter_loss = sum(loss_locals) / len(loss_locals)

                with open(file_path_name, 'a') as f:
                    f.write('loss: {:.6f}  '.format(iter_loss))
                    f.write('dp_train_Accuracy: {:.4f} '.format(s0))
                    f.write('dp_test_Accuracy: {:.4f} '.format(s))
                    f.write('best_Accuracy: {:.4f} '.format(a))
                    f.write('global_Accuracy: {:.4f} '.format(gs))

        else:
            for idx in idxs_users:
                #如果这轮被选择，但是之前没被选择
                if selected_clients[idx] == 0:
                    w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                else:
                    w, loss = train_client_with_last_model_DKD(args,dataset_train,train_data_users[idx],net = local_nets[idx], last_net = last_local_nets[idx])
                #将上轮的模型保存，当作之后的个性化模型
                w_last_locals[idx] = w
                #更新训练后的模型
                w_locals[idx] = w
                #计算更新量
                w_delta_locals[idx] = subtract(w_locals[idx],w_glob)
                norm=cal_clip(w_delta_locals[idx])
                print("norm:"+str(norm))
                # 如果加DP，裁剪并加噪，然后计算准确率
                if args.useDP:
                    if args.clipmethod=="per":
                        w_delta_locals[idx],b = clip_and_add_noise_per(w_delta_locals[idx], max_clip_norm,
                                                                     args.noise_multiplier,
                                                                     args)
                        b_locals .append(b)
                    else:
                        w_delta_locals[idx] = clip_and_add_noise_our(w_delta_locals[idx], max_clip_norm,
                                                                     args.noise_multiplier,
                                                                     args)
                    #计算每个客户端的l2范数
                    #w_delta_norms[idx] = cal_clip(w_delta_locals[idx])
                print(loss)
                loss_locals.append(copy.deepcopy(loss))

            #print("maxnorm:"+str(max(max_clip_norm_list)))

            #如果客户端被选择过
            #将上轮本地模型存下来，并加载到网络中，并将列表中对应值设为1，默认是0（没被选择参与过训练）
            for i in idxs_users:
                last_local_nets[i] = Load_Model(args=args)
                last_local_nets[i].train()
                last_local_nets[i].load_state_dict(w_last_locals[i])
                selected_clients[i] = 1

            # update global weights
            #上传的是更新量,用上一轮的全局模型加上更新量聚合之后的全局模型，得到新一轮的w_glob
            w_delta_glob = FedAvg_sampleClient_delta(w_delta_locals, idxs_users)

            w_glob = add(w_glob,w_delta_glob)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)


            gs = 0
            gl = 0
            for i in idxs_users:
                acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
                s_global, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                # s0 += acc_train
                gs += s_global
                gl += loss_test
            gs /= m
            gl /= m


            s0 = 0
            s = 0
            clip=0
            for i in idxs_users:
                acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
                acc_test, loss_test_temp = test_client(args, dataset_train, test_data_users[i], local_nets[i])
                #loss_test[i][0]
                s0 += acc_train
                s += acc_test
            s0 /= m
            s /= m
            a = isBetter(s, a)
            iter_loss = sum(loss_locals) / len(loss_locals)

            with open(file_path_name, 'a') as f:
                f.write('loss: {:.6f}  '.format(iter_loss))
                f.write('dp_train_Accuracy: {:.4f} '.format(s0))
                f.write('dp_test_Accuracy: {:.4f} '.format(s))
                f.write('best_Accuracy: {:.4f} '.format(a))
                f.write('global_Accuracy: {:.4f} '.format(gs))




            # 初始化裁剪阈值，固定的指定一个值就行，自适应的指定一个列表
            if args.clipmethod=='my2':
                max_clip_norm = Adaclip_tune2(max_clip_norm,args,iter)
            elif args.clipmethod=='my1':
                max_clip_norm = Adaclip_tune1(max_clip_norm, args, iter)
            elif args.clipmethod=='my':
                if iter <= 2:
                    max_clip_norm = Adaclip_tune1(max_clip_norm,args,iter)
                else:
                    max_clip_norm = Adaclip_tune2(max_clip_norm, args, iter)





