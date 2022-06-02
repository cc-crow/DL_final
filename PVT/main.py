#!/usr/bin/env python
# coding: gbk


import matplotlib
matplotlib.use('SVG')

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from pvt import pvt_tiny
from trainer import trainer


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_class = 100
batch_size = 128
data_dir = '/home/leijingshi/cvpr/data/cifar100/'
param_dir = '/home/leijingshi/DL/final/pvt/pvt_tiny_cifar.pt'
figure_dir = '/home/leijingshi/DL/final/pvt/pvt_'
num_epochs_train = 250
#milestone = [100,200]
lr = 5e-04
pretrain = False

if pretrain:
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
else:
    mean = [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
    std = [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean, std)])

train_dataset = datasets.cifar.CIFAR100(root=data_dir, train=True, transform=transform_train, download=True)
test_dataset = datasets.cifar.CIFAR100(root=data_dir, train=False, transform=transform_test, download=True)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

net = pvt_tiny(pretrained=False, img_size=32, num_classes=num_class, drop_path_rate=0.1)
if pretrain:
    param = torch.load('pvt_tiny.pth')
    del param['head.weight']
    del param['head.bias']
    del param['pos_embed1']
    del param['pos_embed2']
    del param['pos_embed3']
    del param['pos_embed4']
    net.load_state_dict(param,strict=False)

optimizer = optim.AdamW(net.parameters(),lr = lr,weight_decay = 0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30, eta_min=1e-05)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestone,gamma=0.1)

trainer_ = trainer(train_iter,test_iter,net,optimizer,device,scheduler)
train_loss,test_loss,test_acc = trainer_.train(num_epochs_train,'PVT',param_dir)

num = list(range(1,1+len(train_loss)))
plt.figure()
plt.plot(num,train_loss,label='train_set')
plt.plot(num,test_loss,label='test_set')
plt.xlabel('iterations')
plt.ylabel('LabelSmoothingCrossEntropyLoss')
plt.legend()
plt.savefig(figure_dir+'loss.jpg')

plt.figure()
plt.plot(num,test_acc)
plt.xlabel('iterations')
plt.ylabel('Accuracy')
plt.savefig(figure_dir+'accuracy.jpg')
