import numpy as np
import torchvision
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import EMNIST
import torch
from models.Nets import MLP, CNNMnist, CNNCifar, ResNet, MobileNet,Net, NewNet, customResNet, customMobileNet162, customMobileNet138, customMobileNet150


#from fedlab.utils.dataset.partition import (CIFAR10Partitioner,
                                            #CIFAR100Partitioner)

# function to load predefined datasets; can make custom dataloader here as well
# also divide the data for all users by using sampling.py file present in utility folder
from models.Nets import CNNEmnist

from utility.sampling_dirichlet import load_partition_data_emnist


from utility.sampling import pathological_non_iid_split,dirichlet_split_noniid

from utility.sampling import mnist_iid, emnist_iid

from utility.sampling import fmnist_iid, svhn_iid, cifar_iid



def Load_Dataset(args):

    ''' 
    Function to load predefined datasets such as CIFAR-10, CIFAR-100 and MNIST via pytorch dataloader

    Declare Custom Dataloaders here if you want to change the dataset

    Also, the function to split training data among all the clients is called from here 
    
    '''
    
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid == 'iid':
            dict_users = mnist_iid(args,dataset_train, args.num_users)
        if args.iid == 'p':
            #dict_users = mnist_noniid(args,dataset_train, args.num_users)
            dict_users = pathological_non_iid_split(args, dataset_train, args.num_users)
        if args.iid == 'dir':
            dict_users = dirichlet_split_noniid(args, dataset_train, args.dirichlet_alpha)
        return dataset_train, dataset_test, dict_users
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.EMNIST('../data/emnist/', train=True, download=True, transform=trans_emnist,split='balanced')
        dataset_test = datasets.EMNIST('../data/emnist/', train=False, download=True, transform=trans_emnist,split='balanced')
        if args.iid== 'iid':
            dict_users = emnist_iid(args,dataset_train, args.num_users)
        if args.iid == 'p':
            #dict_users = emnist_noniid(args,dataset_train, args.num_users)
            dict_users = pathological_non_iid_split(args, dataset_train, args.num_users)
        if args.iid == 'dir':
            dict_users = dirichlet_split_noniid(args, dataset_train, args.dirichlet_alpha)
        return dataset_train, dataset_test, dict_users
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/FashionMNIST/', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../data/FashionMNIST/', train=False, download=True, transform=trans_fmnist)
        if args.iid== 'iid':
            dict_users = fmnist_iid(args,dataset_train, args.num_users)
        if args.iid == 'p':
            #dict_users = fmnist_noniid(args,dataset_train, args.num_users)
            dict_users = pathological_non_iid_split(args, dataset_train, args.num_users)
        if args.iid == 'dir':
            dict_users = dirichlet_split_noniid(args, dataset_train, args.dirichlet_alpha)
        return dataset_train, dataset_test, dict_users
    elif args.dataset == 'svhn':
        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                                                     (0.19803012, 0.20101562,
                                                                                      0.19703614))])
        dataset_train = torchvision.datasets.SVHN('../data/SVHN', split="train", download=True, transform=trans_svhn)
        dataset_test = torchvision.datasets.SVHN('../data/SVHN', split="test", download=True, transform=trans_svhn)
        if args.iid== 'iid':
            dict_users = svhn_iid(args,dataset_train, args.num_users)
        if args.iid == 'p':
            #dict_users = svhn_noniid(args,dataset_train, args.num_users)
            dict_users = pathological_non_iid_split(args, dataset_train, args.num_users)
        if args.iid == 'dir':
            dict_users = dirichlet_split_noniid(args, dataset_train, args.dirichlet_alpha)
        return dataset_train, dataset_test, dict_users
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dict_users={}
        if args.iid== 'iid':
            dict_users = cifar_iid(args,dataset_train, args.num_users)
        if args.iid == 'p':
            #dict_users = cifar_noniid(args,dataset_train, args.num_users)
            dict_users =pathological_non_iid_split(args, dataset_train, args.num_users)
        if args.iid =='dir':
            dict_users = dirichlet_split_noniid(args, dataset_train, args.dirichlet_alpha)
        return dataset_train, dataset_test, dict_users
    elif args.dataset =='cifar100':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(args,dataset_train, args.num_users)
        else:
            #dict_users = cifar100_noniid(args,dataset_train, args.num_users)
            dict_users = pathological_non_iid_split(args, dataset_train, args.num_users)
        return dataset_train, dataset_test, dict_users
    else:
        exit('Error: unrecognized dataset')


# function to use the model architectures present in Nets.py file present in models folder

def Load_Model(args):
    
    '''

    Function to load the required architecture (model) for federated learning

    '''

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'emnist':
        net_glob = CNNEmnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob = CNNEmnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'MobileNet':
        net_glob = MobileNet(args=args).to(args.device)
    elif args.model == 'resnet18':
        net_glob = ResNet.ResNet18(args=args).to(args.device)
    elif args.model == 'Tresnet18':
        net_glob = ResNet.TrueResNet18(args=args).to(args.device)
    elif args.model == 'ResNet':
        net_glob = ResNet.ResNet34(args=args).to(args.device)
    elif args.model == 'ResNet50':
        net_glob = ResNet.ResNet50(args=args).to(args.device)
    elif args.model == 'ResNet101':
        print('ResNet101')
        net_glob = ResNet.ResNet101(args=args).to(args.device)
    elif args.model == 'ResNet152':
        print('ResNet152')
        net_glob = ResNet.ResNet152(args=args).to(args.device)
    elif args.model =='NewNet':
        net_glob = NewNet(args=args).to(args.device)
    elif args.model =='customMobileNet162':
        net_glob = customMobileNet162(args=args).to(args.device)
    elif args.model =='customMobileNet150':
        net_glob = customMobileNet150(args=args).to(args.device)
    elif args.model =='customMobileNet138':
        net_glob = customMobileNet138(args=args).to(args.device)
    elif args.model =='customResNet204':
        net_glob = customResNet.customResNet204(args=args).to(args.device)
    elif args.model =='customResNet192':
        net_glob = customResNet.customResNet192(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    return net_glob
        
