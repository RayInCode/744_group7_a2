import os
import time
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import argparse
import model as mdl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    model.train()   # set the model to trainning mode
    total_time = []
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()

        data, target = data.to(device), target.to(device)   # load data to device(cpu here)
        optimizer.zero_grad()   # clean all the gradient generated for last batch
        output = model(data)    # forward pass
        loss = criterion(output, target)    # calculate the loss
        loss.backward()     # backwaed pass to calculate the gradients
        optimizer.step()    # update the model with the 
        
        end_time = time.time()
        duration = end_time - start_time
        total_time.append(duration)

        if batch_idx % 20 == 0 or batch_idx == len(train_loader) - 1:
            print('Train Epoch {}: [{}/{} ({:.1f}%)]\tBatch {}\tLoss: {:.4f}'.format(
                epoch, batch_idx + 1,  len(train_loader), 100. * (batch_idx+1) / len(train_loader), batch_idx, loss.item()))
                
        if batch_idx >= 39:
            break

    print('Runtime: Total: {:.4f}, Each iteration: {:.4f}\n'.format(sum(total_time[1:]), sum(total_time[1:])/len(total_time[1:])))
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model.module(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    train_sampler = DistributedSampler(training_set, num_replicas=args.num_nodes, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=global_batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)

    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)

        dist.barrier()
        
        # only master need to test the accuracy of the final model
        if args.rank == 0:
            test_model(model, test_loader, training_criterion)

    dist.destroy_process_group()

if __name__ == "__main__":
    # Parse arguments to get master's IP address, the rank and the number of nodes
    parser = argparse.ArgumentParser(description='Parse DataParallel Traning Arguments')
    parser.add_argument('--master-ip', type=str, help='IP address of the master node')
    parser.add_argument('--num-nodes', type=int, default=4, help='Total number of nodes')
    parser.add_argument('--rank', type=int, help='Rank of the current node')
    args = parser.parse_args()

    # Set up
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.manual_seed(0)    # for checking the correctness 

    # Set up for communication group
    os.environ['MASTER_ADDR'] = args.master_ip
    os.environ['MASTER_PORT'] = '6585'
    dist.init_process_group(backend='gloo', rank=args.rank, world_size=args.num_nodes)

    global_batch_size = 256     # global  batch size for the whole cluster used in distriputed training
    batch_size = global_batch_size // args.num_nodes    # batch for one node  
    main()
