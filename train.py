from asyncore import write
import os
from random import shuffle # tensorboardの出力先作成
import matplotlib.pyplot as plt # 可視化
import numpy as np # 計算
import torch # 機械学習フレームワークとしてpytorchを使用
import torch.nn as nn # クラス内で利用するモジュールのため簡略化
import torch.nn.functional as F # クラス内で利用するモジュールのため簡略化
from torch import optim # 最適化アルゴリズム
from torch.utils.tensorboard import SummaryWriter # tensorboardの利用
from torchvision import datasets, transforms # データセットの準備

from model import VAE


if not os.path.exists('./logs'):
    os.makedirs("./logs")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))]
    )

dataset_train_valid = datasets.MNIST("./", train=True, download=True, transform=transform)
dataset_test = datasets.MNIST("./", train=False, download=True, transform=transform)

size_train_valid = len(dataset_train_valid)
size_train = int(size_train_valid * 0.8)
size_valid = size_train_valid - size_train
dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train_valid, [size_train, size_valid])

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 1000, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size = 1000, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(2).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
loss_valid = 10**7
loss_valid_min = 10**7
num_no_improved = 0
num_batch_train = 0
num_batch_valid = 0
writer = SummaryWriter(log_dir="./logs")

# learning
for num_iter in range(num_epochs):
    model.train()
    for x, t in dataloader_train:
        lower_bound, _, _ = model(x, device)
        loss = -sum(lower_bound)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss_train/KL", -lower_bound[0].cpu().detach().numpy(), num_iter + num_batch_train)
        writer.add_scalar("Loss_train/Reconst", -lower_bound[1].cpu().detach().numpy(), num_iter + num_batch_train)
        num_batch_train += 1
    num_batch_train -= 1


    model.eval()
    loss = []
    for x, t in dataloader_valid:
        lower_bound, _, _ = model(x, device)
        loss.append(-sum(lower_bound).cpu().detach().numpy())
        writer.add_scalar("Loss_valid/KL", -lower_bound[0].cpu().detach().numpy(), num_iter + num_batch_valid)
        writer.add_scalar("Loss_valid/Reconst", -lower_bound[1].cpu().detach().numpy(), num_iter + num_batch_valid)
        num_batch_valid += 1
    num_batch_valid -= 1
    loss_valid = np.mean(loss)
    loss_valid_min = np.minimum(loss_valid_min, loss_valid)
    print(f"[EPOCH{num_iter+1}] loss_valid: {int(loss_valid)} | Loss_valid_min: {int(loss_valid_min)}")

    if loss_valid_min < loss_valid:
        num_no_improved += 1
        print(f"{num_no_improved}回連続でValidationが悪化しました")
    else:
        num_no_improved = 0
        torch.save(model.state_dict(), f"./z_{model.z_dim}.pth")
    
    if (num_no_improved >= 10):
        print(f"{num_no_improved}回連続でValidationが悪化したため学習を止めます")
        break

writer.close()

