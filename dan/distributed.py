import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging
import random
import argparse
import model as mdl


# Parse argument
parser = argparse.ArgumentParser(description='Parse DataParallel Traning Arguments')
parser.add_argument('--world-size', default=4, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

# Set up
device = "cpu"
torch.set_num_threads(4)
batch_size = 256 # batch for one node
torch.manual_seed(0) # start from same model on all the workers

def sync_gradient(model):
    # Gather gradients from all processes at rank 0
    all_gradients = [param.grad.clone() for param in model.parameters()]
    all_gradients = [grad.to(args.rank) for grad in all_gradients]
    gathered_gradients = [torch.zeros_like(grad) for grad in all_gradients]
    dist.gather(torch.stack(all_gradients), gathered_gradients, dst=0)
    
    if args.rank == 0:
      mean_gradients = torch.stack(gathered_gradients).mean(dim=0)
      mean_gradients = [grad.to(0) for grad in mean_gradients]
      dist.scatter(mean_gradients, torch.stack(all_gradients), scatter_list=all_gradients)

    # Synchronize model parameters across all processes
    dist.barrier()


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        sync_gradient(model)

        optimizer.step()
        
        # Print every 20 iterations
        running_loss += loss.item()
        if batch_idx % 20 == 19:
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

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
    # Use DistributedSampler to distribute the dataset among the workers
    train_sampler = DistributedSampler(training_set, num_replicas=args.world_size, rank=args.rank)
    
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # Initialize the process group
    dist.init_process_group(backend='gloo', init_method='tcp://172.18.0.2:6585', world_size=args.world_size, rank=args.rank)
    
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
