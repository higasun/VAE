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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 2
model = VAE(z_dim)
model.load_state_dict(torch.load("./z_2.pth"))



cm = plt.get_cmap("tab10")
for num_batch, data in enumerate(dataloader_test):
    fig_plot, ax_plot = plt.subplots(figsize=(9, 9))

    _, z, _ = model(data[0], device)
    z = z.cpu().detach().numpy()

    for k in range(10):
        cluster_indexes = np.where(data[1].cpu().detach().numpy() == k)[0]
        ax_plot.plot(z[cluster_indexes, 0], z[cluster_indexes, 1], "o", ms=4, color=cm(k))

    fig_plot.savefig(f"./results/latent_space_{z_dim}.png")
    plt.close(fig_plot)