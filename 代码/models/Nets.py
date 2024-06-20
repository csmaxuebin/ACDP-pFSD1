#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


# A small multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


# A small cnn model for mnist dataset
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10
                               , kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# a small cnn model for eminist dataset
class CNNEmnist(nn.Module):
    def __init__(self,args):
        super(CNNEmnist, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10
                               , kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# a small cnn model for cifar10 dataset
# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   
# description of a block for a block in mobilenet     
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


# architecture of a mobilenet
class MobileNet(nn.Module):

    """ 
    Class for MobileNet architecture - Standard Architecture

    """
 

    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, args):

        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, args.num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)
    

# a block for resnet architecture
class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
class Bottleneck(nn.Module):


    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
  

# architecture for resnet  

class ResNet(nn.Module):

    """ 
    Class for ResNet architecture - Standard Architecture

    Resnet with 18, 34, 50, 101 and 152 layers can be declared. 

    We used ResNet 34 for our experiments. 

    """

    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)        
        return out

    def ResNet18(args):
        res18 = ResNet(BasicBlock, [2,2,2,2], args)

        res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

        res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
        res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

        res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
        res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
        res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
        res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
        res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

        res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
        res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
        res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
        res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
        res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

        res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
        res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
        res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
        res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
        res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

        assert len(dict(res18.named_parameters()).keys()) == len(
            res18.state_dict().keys()), 'More BN layers are there...'

        return res18

#122层的resnet18网络
    def TrueResNet18(args):
        res18 = ResNet(BasicBlock, [2,2,2,2], args)
        return res18

    def ResNet34(args):
        return ResNet(BasicBlock, [3,4,6,3], args)

    def ResNet50(args):
        return ResNet(Bottleneck, [3,4,6,3], args)

    def ResNet101(args):
        return ResNet(Bottleneck, [3,4,23,3], args)

    def ResNet152(args):
        return ResNet(Bottleneck, [3,8,36,3], args)
   
# a different way of implementing mobilenet architecture 

class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)


# changing the base layers with linear mapping in case of mobilenet
class NewNet(nn.Module):
    
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [1024]

    def __init__(self, args):
        super(NewNet, self).__init__()
        self.linear2 = nn.Linear(3072,100)
        self.linear3 = nn.Linear(100,4096)
        self.bn1 = nn.BatchNorm2d(1)
        self.layers = self._make_layers(in_planes=1024)
        self.linear = nn.Linear(1024, args.num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1,1,3072)
        out = self.linear2(x)
        out = self.linear3(out)
        out = out.view(-1,1024,2,2)
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        return x

class customMobileNet150(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
#     cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512]
    cfg = [1024]

    def __init__(self, args):
        super(customMobileNet150, self).__init__()
        self.linear1 = nn.Linear(3072,4096)
        
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.layers = self._make_layers(in_planes=1024)
        self.linear = nn.Linear(1024, args.num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1,3072)
        out = self.linear1(x)
        out = out.view(-1,1024,2,2)
        out = self.layers(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class customMobileNet138(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
#     cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512]
    cfg = [(1024,2), 1024]

    def __init__(self, args):
        super(customMobileNet138, self).__init__()
        self.linear1 = nn.Linear(3072,8192)
        
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.layers = self._make_layers(in_planes=512)
        self.linear = nn.Linear(1024, args.num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1,3072)
        out = self.linear1(x)
        out = out.view(-1,512,4,4)
        out = self.layers(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class customMobileNet162(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
#     cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512]
    cfg = [(1024,2), 1024]

    def __init__(self, args):
        super(customMobileNet162, self).__init__()
        self.linear1 = nn.Linear(3072,1024)
        
        self.linear = nn.Linear(1024, args.num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1,3072)
        # print(x.shape)
        out = self.linear1(x)
        out = self.linear(out)
        return out

class customResNet(nn.Module):

    """ 
    Class for ResNet architecture - Standard Architecture

    Resnet with 18, 34, 50, 101 and 152 layers can be declared. 

    We used ResNet 34 for our experiments. 

    """

    def __init__(self, block, num_blocks,args):
        super(customResNet, self).__init__()
        self.linear1 = nn.Linear(3072,8192)
        self.in_planes = 512
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            # print(self.in_planes)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1,3072)
        out = self.linear1(x)
        out = out.view(-1,512,4,4)
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)        
        return out

    def customResNet204(args):
        return customResNet(BasicBlock, [3,4,6,1], args)

    def customResNet192(args):
        return customResNet(BasicBlock, [3,4,6,2], args)