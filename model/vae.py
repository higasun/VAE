import numpy as np # 計算
import torch # 機械学習フレームワークとしてpytorchを使用
import torch.nn as nn # クラス内で利用するモジュールのため簡略化
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.x_dim = 28 * 28
        self.z_dim = z_dim
        self.enc_fc1 = nn.Linear(self.x_dim, 400)
        self.enc_fc2 = nn.Linear(400, 200)
        self.enc_fc3_mean = nn.Linear(200, z_dim)
        self.enc_fc3_logvar = nn.Linear(200, z_dim)
        self.dec_fc1 = nn.Linear(z_dim, 200)
        self.dec_fc2 = nn.Linear(200, 400)
        self.dec_drop = nn.Dropout(p=0.2)
        self.dec_fc3 = nn.Linear(400, self.x_dim)
        self.rec_loss = nn.BCELoss(reduction="sum")

    def encoder(self, x):
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def decoder(self, z):
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x, device):
        mean, log_var = self.encoder(x.to(device))
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        reconstruction = - self.rec_loss(y, x)
        return [KL, reconstruction], z, y

    