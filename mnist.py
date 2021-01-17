from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class Net_BN(nn.Module):
    def __init__(self):
        super(Net_BN, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(10)
        self.bn3 = nn.BatchNorm1d(10)
        self.bn4 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    t_loss = 0
    
    # 掃過整個 dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # data shape = (Batch_size, channel, width, height) = (32, 1, 28, 28)
        # target = 0~9的整數
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        t_loss += loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    # 回傳整個 dataset 平均的 loss
    return t_loss/ len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

class Args ():
    def __init__(self):
        self.batch_size = 30
        self.test_batch_size = 32
        self.lr = 0.001
        self.dry_run = False
        self.log_interval = 1000
        self.gamma = 0.7
        self.epochs = 10

def main():
    
    # 參數設定 請看72行
    args = Args()

    # 如果可以用gpu就用
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 將剛剛的參數分配給幾個變數
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 將資料轉成 tensor 跟 正規化 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # 設定training 跟 testing dataset
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    dataset1, val = random_split(dataset1,[50000,10000])
  
    
    # 設定data loader
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # 設定 Model 跟 optimizer
    model = Net().to(device)
    model_bn = Net_BN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    optimizer_bn = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler_bn = StepLR(optimizer_bn, step_size=1, gamma=args.gamma)
    
    # Training + Testing
    # 一次epoch 掃過整個 dataset
    y = []
    z = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        train_loss_bn = train(args, model_bn, device, train_loader, optimizer_bn, epoch)
        scheduler.step()
        scheduler_bn.step()
        y.append(train_loss)
        z.append(train_loss_bn)
    
    # 畫圖
    # x = [0, 1, 2, 3, 4,..., 10]
    # y = [loss_0, loss_1, ...,loss_10]
    x = [i for i in range(args.epochs)]
    plt.plot(x, y)
    plt.plot(x, z)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.show()
    # 儲存 Model 的 Weight
#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
