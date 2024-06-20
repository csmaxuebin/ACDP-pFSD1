import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utility.Dittooptimizer import PerturbedGradientDescent


class DatasetSplit(Dataset):

    """
    Class DatasetSplit - To get datasamples corresponding to the indices of samples a particular client has from the actual complete dataset

    """

    def __init__(self, dataset, idxs):

        """

        Constructor Function

        Parameters:

            dataset: The complete dataset

            idxs : List of indices of complete dataset that is there in a particular client

        """
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):

        """

        returns length of local dataset

        """

        return len(self.idxs)

    def __getitem__(self, item):

        """
        Gets individual samples from complete dataset

        returns image and its label

        """
        image, label = self.dataset[self.idxs[item]]
        return image, label


def train_client_ini(args, dataset, train_idx, net):
    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    numbers=[1,2]

    for iter in range(2):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss = loss
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

# function to train a client
def train_client(args,dataset,train_idx,net):

    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    
    for iter in range(args.local_ep):   
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss=loss
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss) / len(epoch_loss)


def train_client_test(args, dataset, train_idx, net):
    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    for iter in range(args.local_ep):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss = loss
            loss.backward()
            #optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


#pFedSD对应的训练方法
def train_client_with_last_model(args, dataset, train_idx, net, last_net):
    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    last_net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            # 计算上轮本地模型的输出
            last_local_logit = last_net(images)
            #客户端这轮的logit
            output = net(images)
            #硬损失
            hard_loss = loss_func(output, labels)
            #蒸馏损失
            ditillation_loss = divergence(
                student_logits=output,
                teacher_logits=last_local_logit,
                KL_temperature=args.KL_T,
            )
            #计算总的损失
            loss = hard_loss + ditillation_loss
            #loss = args.alpha * hard_loss + (1-args.alpha) * ditillation_loss
            #反向传播
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

#pFedLSD对应的训练方法
def train_client_with_last_model_DKD(args, dataset, train_idx, net, last_net):
    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    last_net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    batch_diff = 0.0
    for iter in range(args.local_ep):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            # 计算上轮本地模型的输出
            last_local_logit = last_net(images)
            #客户端这轮的logit
            output = net(images)
            #硬损失
            hard_loss = loss_func(output, labels)
            #蒸馏损失
            student_logits=output
            teacher_logits=last_local_logit
            KL_temperature=args.KL_T
            gt_mask = _get_gt_mask(student_logits, labels)
            other_mask = _get_other_mask(student_logits,labels)
            pred_student = F.softmax(student_logits / KL_temperature, dim=1)
            pred_teacher = F.softmax(teacher_logits / KL_temperature, dim=1)
            pred_student = cat_mask(pred_student, gt_mask, other_mask)
            pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
            log_pred_student = torch.log(pred_student)
            pred_teacher_part2 = F.softmax(
                teacher_logits / KL_temperature - 1000.0 * gt_mask, dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                student_logits / KL_temperature - 1000.0 * gt_mask, dim=1
            )
            tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (KL_temperature ** 2)
                / labels.shape[0]
            )
            nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (KL_temperature ** 2)
                / labels.shape[0]
            )
            ditillation_loss = args.ALPHA * tckd_loss + args.BETA * nckd_loss

            target_class_idx = labels  # 目标类别索引
            max_other_class_logits, _ = torch.max(
                output.clone().scatter_(1, target_class_idx.view(-1, 1), float('-inf')), dim=1)
            class_diff = output[torch.arange(output.size(0)), target_class_idx] - max_other_class_logits

            batch_diff += class_diff.mean().item()

            #计算总的损失
            #loss =hard_loss + ditillation_loss
            loss = (1-args.alpha) * hard_loss + args.alpha * ditillation_loss
            #loss = hard_loss + ditillation_loss
            #反向传播
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    print("diff:"+str(batch_diff))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def ptrain_client_Ditto(args, dataset, train_idx, net,net_glob):
    '''
    Train individual client models
    Parameters:
        net (state_dict) : Client Model
        datatest (dataset) : Complete dataset loaded by the Dataloader
        args (dictionary) : The list of arguments defined by the user
        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client
    Returns:
        net.state_dict() (state_dict) : The updated weights of the client model
        train_loss (float) : Cumulative loss while training
    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()

    # train and update
    optimizer_per = PerturbedGradientDescent(net.parameters(), lr=args.lr, mu=args.mu)
    #optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    w_0 = copy.deepcopy(net.state_dict())
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer_per.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer_per.step(args,net,net_glob)

            # w_net = copy.deepcopy(net.state_dict())
            # w_ditto=net_glob.state_dict()
            # for key in w_net.keys():
            #     w_net[key] = w_net[key] - args.lr * args.mu * (w_0[key] - w_ditto[key])
            # net.load_state_dict(w_net)
            # optimizer.zero_grad()


            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def divergence(student_logits, teacher_logits, KL_temperature):
    divergence = F.kl_div(
        F.log_softmax(student_logits / KL_temperature, dim=1),
        F.softmax(teacher_logits / KL_temperature, dim=1),
        reduction="batchmean",
    )  # forward KL
    return KL_temperature * KL_temperature * divergence

def _get_gt_mask(logits, target):
    target =target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1).to(torch.int64), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1).to(torch.int64), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    # 对于t张量的每一行，将它和mask1的对应行相乘并对结果求和，得到一个新的张量t1
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def finetune_client(args,dataset,train_idx,net):

    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    
    for iter in range(1):   
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss) / len(epoch_loss)


# function to test a client
def test_client(args,dataset,test_idx,net):

    '''

    Test the performance of the client models on their datasets

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : The data on which we want the performance of the model to be evaluated

        args (dictionary) : The list of arguments defined by the user

        test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model

        test_loss (float) : Cumulative loss on the data

    '''
    
    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.local_bs)  
    net.eval()
    #print (test_data)
    test_loss = 0
    correct = 0
    
    l = len(data_loader)
    
    with torch.no_grad():
                
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss
def isBetter(accuracy,best_accuracy):
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    return best_accuracy



def train_client_with_last_model_DKD_test(args, dataset, train_idx, net, last_net):
    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    last_net.train()

    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    batch_diff = 0.0
    for iter in range(args.local_ep):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            # 计算上轮本地模型的输出
            last_local_logit = last_net(images)
            #客户端这轮的logit
            output = net(images)
            #硬损失
            hard_loss = loss_func(output, labels)
            #蒸馏损失
            student_logits=output
            teacher_logits=last_local_logit
            KL_temperature=args.KL_T
            gt_mask = _get_gt_mask(student_logits, labels)
            other_mask = _get_other_mask(student_logits,labels)
            pred_student = F.softmax(student_logits / KL_temperature, dim=1)
            pred_teacher = F.softmax(teacher_logits / KL_temperature, dim=1)
            pred_student = cat_mask(pred_student, gt_mask, other_mask)
            pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
            log_pred_student = torch.log(pred_student)
            pred_teacher_part2 = F.softmax(
                teacher_logits / KL_temperature - 1000.0 * gt_mask, dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                student_logits / KL_temperature - 1000.0 * gt_mask, dim=1
            )
            tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (KL_temperature ** 2)
                / labels.shape[0]
            )
            nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (KL_temperature ** 2)
                / labels.shape[0]
            )
            ditillation_loss = args.ALPHA * tckd_loss + args.BETA * nckd_loss

            target_class_idx = labels  # 目标类别索引
            max_other_class_logits, _ = torch.max(
                output.clone().scatter_(1, target_class_idx.view(-1, 1), float('-inf')), dim=1)
            class_diff = output[torch.arange(output.size(0)), target_class_idx] - max_other_class_logits

            batch_diff += class_diff.mean().item()

            #计算总的损失
            #loss =hard_loss + ditillation_loss
            loss = (1-args.alpha) * hard_loss + args.alpha * ditillation_loss
            #loss = hard_loss + ditillation_loss
            #反向传播
            #loss.backward()
            #optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    print("diff:"+str(batch_diff))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
