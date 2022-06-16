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
from torchvision import datasets, transforms

from model import VAE


if not os.path.exists('./logs'):
    os.makedirs("./logs")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))]
    )

dataset_test = datasets.MNIST("./", train=False, download=True, transform=transform)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1000, shuffle=False)

model = VAE(2)
model.load_state_dict(torch.load("./z_2.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for num_batch, data in enumerate(dataloader_test):
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
        
    for i, im in enumerate(data[0].view(-1, 28, 28)[:10]):
        axes[0][i].imshow(im, "gray")

    _, _, y = model(data[0], device)
    y = y.cpu().detach().numpy().reshape(-1, 28, 28)
    for i, im in enumerate(y[:10]):
        axes[1][i].imshow(im, "gray")
    
    fig.savefig(f"./results/reconstruction_{num_batch}.png")
    plt.close(fig)