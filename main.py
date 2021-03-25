import torch.nn as nn
from functionalities import dataloader as dl
from functionalities import evaluater as ev
from functionalities import filemanager as fm
from functionalities import trainer as tr
from functionalities import plot as p
from architecture import RotNet as RN

trainset, testset, classes = dl.load_cifar("./datasets")
trainloader, validloader, testloader = dl.make_dataloaders(trainset, testset, 128)

criterion = nn.CrossEntropyLoss()

# set rot classes
rot_classes = ['original', '90 rotation', '180 rotation', '270 rotation']

# initialize network
net_block3 = RN.RotNet(num_classes=4, num_conv_block=3, add_avg_pool=False)

# train network
rot_block3_loss_log, _, rot_block3_test_accuracy_log, _, _ = tr.adaptive_learning([0.1, 0.02, 0.004, 0.0008], 
    [60, 120, 160, 200], 0.9, 5e-4, net_block3, criterion, trainloader, None, testloader, rot=['90', '180', '270'])

# 5 ConvBlock RotNet model and Classifiers
ev.evaluate_all(5, testloader, classes)

