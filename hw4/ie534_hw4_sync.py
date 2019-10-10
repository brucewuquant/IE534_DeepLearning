# -*- coding: utf-8 -*-
"""ie534_hw4_sync.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uIA8arbFh3KNrB9RHy_k1rXx9vI5ZOhv
"""

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import os
import subprocess
from mpi4py import MPI

# Code for iniitialization pytorch distributed

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# For trainning data
trainset = torchvision.datasets.CIFAR100(root='./data',
train=True,download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset,
batch_size=100, shuffle=True, num_workers=0)
# For testing data
testset = torchvision.datasets.CIFAR100(root='./data',
train=False,download=True, transform=train_transform)
testloader = torch.utils.data.DataLoader(testset,
batch_size=100, shuffle=False, num_workers=0)

class BasicBlock(nn.Module):
    expansion =1 
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.downsample = downsample
    
    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

"""
2   32 32 Stride 1 padding 1
4   64 64 Stride 2 padding 1
4   128 128 stride 2 padding 1
2   256 256 stride 2 padding 1
"""

# Implement ResNets in Fig2 with basic block
class ResNet(nn.Module):
    def __init__(self, basic_block, num_blocks_list, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 32
        self.conv1 =nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.2)
        self.conv2 = self._make_layer(basic_block, 32, num_blocks_list[0])
        self.conv3 = self._make_layer(basic_block, 64, num_blocks_list[1],stride=2)
        self.conv4 = self._make_layer(basic_block, 128, num_blocks_list[2],stride=2)
        self.conv5 = self._make_layer(basic_block, 256, num_blocks_list[3],stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(256*2*2, num_classes)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, basic_block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels*basic_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*basic_block.expansion,
                            kernel_size=1, stride = stride, padding = 0),
                nn.BatchNorm2d(out_channels*basic_block.expansion))
        
        layers = []
        layers.append(basic_block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * basic_block.expansion
        for _ in range(1, num_blocks):
            layers.append(basic_block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def run(rank, size):

    test_accuracy_list = []

    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(1234)
    global trainloader
    model = ResNet(BasicBlock, [2,4,4,2], 100).cuda()

    optimizer = optim.Adam(model.parameters())

    for epoch in range(50):

        epoch_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            if (epoch >= 6):
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if 'state' in state.keys():
                            if (state['step'] >= 1024):
                                state['step'] = 1000
            optimizer.step()

        correct = 0
        total = 0

        model.eval()

        for i, data in enumerate(testloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = resnet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels).float().sum().item()
            total += len(labels)

        test_accuracy_list.append(correct/total)

        if correct/total > 0.62:
            with open('tiny_test_accuracy_'+str(correct/total)+'.txt','w') as f:
                for listitem in test_accuracy_list:
                    f.write("%s\n" % listitem)

if __name__ == '__main__':
    run(dist.get_rank(), num_nodes)
