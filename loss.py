import torch
import torch.nn as nn
from torchvision import models
import os
os.environ['CUDA_VISIBLE_DEVICES']="1" # choose GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class TotalLoss:
    def __init__(self, weights,percepWei):
        assert len(percepWei) == 4
        self.pWei = percepWei
        self.vgg = Vgg16().to(device)
        self.vgg.eval()
        self.MSE = nn.MSELoss() 
        self.L1 = nn.L1Loss()
        
        ## distrubute weights
        self.contentW = weights[0]
        self.pixelW = weights[1]
        self.tvW = weights[2]

    def __call__(self, x, y_hat): #x = Hybridout , y_hat = aps
        x = x.to(device)
        y_hat = y_hat.to(device)
        b, c, h, w = x.shape
        ## gray to 3-dim image to fit vgg16
        if c == 1:
            x = x.repeat(1,3,1,1)
        if y_hat.shape[1] == 1:
            y_hat = y_hat.repeat(1,3,1,1)
            
        ## calculate perceptual loss
        y_content_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)
        L_tmp = []
        for i in range(4):
            recon = y_content_features[i]
            recon_hat = y_hat_features[i]
            L_tmp.append(self.MSE(recon,recon_hat))
        L_perceptual = self.pWei[0]*L_tmp[0] + self.pWei[1]*L_tmp[1] + self.pWei[2]*L_tmp[2] + self.pWei[3]*L_tmp[3]
      
        ## calculate pixel loss
        L_pixel = self.L1(y_hat, x)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        # 传统版本
        # diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        # diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        L_tv = (diff_i + diff_j) / float(c * h * w)
        ## total loss
        total_loss = self.contentW*L_perceptual + self.pixelW*L_pixel + self.tvW*L_tv
        
        return total_loss
