import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

# 定义之前已经提供的dirichlet_split_noniid函数，生成索引
def dirichlet_noniid(degree_noniid, dataset, num_users):
    # 将数据集的标签转换为numpy数组
    train_labels = np.array(dataset.targets)
    # 获取数据集中的类别数量
    num_classes = len(dataset.classes)
    # 使用狄利克雷分布生成每个客户端的标签分布
    label_distribution = np.random.dirichlet([degree_noniid] * num_users, num_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(num_classes)]

    client_idx = [[] for _ in range(num_users)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idx[i] += [idcs]

    # print(dict_users, np.shape(dict_users))

    client_idx = [set(np.concatenate(idcs)) for idcs in client_idx]

    return client_idx

def load_partition_data_emnist(dataset_train,dataset_test, args):

    dict_users={}
    dict_train = dirichlet_noniid(args.d_alpha, dataset_train, args.num_users) # non-iid
    dict_test = dirichlet_noniid(args.d_alpha, dataset_test, args.num_users) # non-iid\

    for client_idx in range(args.num_users):
        dict_users[client_idx] = dict_train[client_idx]
        dataset_train_final = list(dict_train[client_idx])
        dataset_test_final = list(dict_test[client_idx])
    return dict_users, dataset_train_final ,dataset_test_final
