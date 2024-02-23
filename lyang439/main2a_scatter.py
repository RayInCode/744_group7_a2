import os
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
import model as mdl
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import time

device = "cpu"
torch.set_num_threads(4)

global_batch_size = 256 # batch for one node

def setup_seed(seed_value=42):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    # random.seed(seed_value)
    # torch.backends.cudnn.deterministic = True

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def train_model(model, train_loader, optimizer, criterion, epoch, rank, world_size):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    model.train()  # set the model to training mode
    total_time = []
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # zero the parameter gradients
        output = model(data)  # forward pass
        loss = criterion(output, target)  # calculate the loss
        loss.backward()  # backward pass

        # Gradient synchronization using gather and scatter
        for param in model.parameters():
            if param.requires_grad:
                # Initialize an empty tensor for scatter_grad on all ranks
                scatter_grad = torch.zeros_like(param.grad)

                if rank == 0:
                    # Only rank 0 prepares gathered_grads and performs the averaging
                    gathered_grads = [torch.zeros_like(param.grad) for _ in range(world_size)]
                    dist.gather(param.grad.data, gather_list=gathered_grads, dst=0)
                    mean_grad = torch.mean(torch.stack(gathered_grads), dim=0)
                    for i in range(world_size):
                        gathered_grads[i] = mean_grad
                    # Scatter averaged gradients from rank 0
                    dist.scatter(tensor=scatter_grad, scatter_list=gathered_grads, src=0)
                else:
                    # Non-source ranks call scatter with an empty list for scatter_list
                    dist.gather(param.grad.data, dst=0)
                    dist.scatter(tensor=scatter_grad, scatter_list=[], src=0)

                # Update gradients with the scattered values
                param.grad.data = scatter_grad.data

        optimizer.step()  # optimize the weights

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
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main(rank, world_size, master_addr, master_port):
    setup_seed()
    setup(rank, world_size, master_addr, master_port)
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
    train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=global_batch_size//world_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=global_batch_size//world_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    # model = DDP(model) 
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch, rank, world_size)
        dist.barrier()
        if rank == 0:  # Only perform testing on the master process
            test_model(model, test_loader, training_criterion)
            # test_model(model.module if world_size > 1 else model, test_loader, training_criterion)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed PyTorch Training')
    parser.add_argument('--rank', type=int, help='Rank of the current process')
    parser.add_argument('--world_size', type=int, default=4, help='Total number of processes')
    parser.add_argument('--master_addr', type=str, help='Master node IP address')
    parser.add_argument('--master_port', type=str, help='Master node port')
    args = parser.parse_args()

    main(args.rank, args.world_size, args.master_addr, args.master_port)
