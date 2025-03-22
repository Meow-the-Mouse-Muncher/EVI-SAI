import torch
import torch.nn as nn
from torchvision import models
from torch import randn
from pytorch_msssim import SSIM
from torchmetrics.image import VisualInformationFidelity
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import logging_presets

import os
# os.environ['CUDA_VISIBLE_DEVICES']="1" # choose GPU
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
    



    

from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
CE = torch.nn.BCELoss(reduction='sum')
class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size = 4):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        # 降维 第一步
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # 继续缩小尺寸
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        
        # 添加自适应池化层确保输出尺寸为32x32
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

        self.channel = channels

        # 修改全连接层的输入维度为32*32
        self.fc1_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar): #VAE 
        std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        # 使用与输入相同的设备生成噪声
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        
        # 使用自适应池化确保特征图尺寸为32x32
        rgb_feat = self.adaptive_pool(rgb_feat)
        depth_feat = self.adaptive_pool(depth_feat)
        
        # 展平为32*32
        rgb_feat = rgb_feat.reshape(-1, self.channel * 1 * 32 * 32)
        depth_feat = depth_feat.reshape(-1, self.channel * 1 * 32 * 32)
        
        #  全连接层降维预测方差和均值
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)
        logvar_depth = self.fc2_depth3(depth_feat)

        # tanh激活函数
        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        
        return latent_loss




    
class MutualInformationLoss(nn.Module):
    def __init__(self, input_channel, channels):
        super(MutualInformationLoss, self).__init__()
        self.mi_frame_event = Mutual_info_reg(input_channel, channels)
        self.mi_frame_eframe = Mutual_info_reg(input_channel, channels)
        self.mi_event_eframe = Mutual_info_reg(input_channel, channels)
        
    def forward(self, event, frame, eframe):
        # 计算三对特征之间的互信息
        mi_fe = torch.clip(self.mi_frame_event(frame, event),-1,1)
        mi_ff = torch.clip(self.mi_frame_eframe(frame, eframe),-1,1)
        mi_ef = torch.clip(self.mi_event_eframe(event, eframe),-1,1)
        
        # 返回总的互信息损失
        return mi_fe + 0.1*mi_ff + 0.1*mi_ef

    
class SpatialFrequencyLoss(nn.Module):
    def __init__(self):
        super(SpatialFrequencyLoss, self).__init__()
        
    def forward(self, img):
        """
        计算图像的空间频率
        Args:
            img: 输入图像 tensor (B, C, H, W)
        Returns:
            空间频率值
        """
        rf = self.row_frequency(img)
        cf = self.column_frequency(img)
        sf = torch.sqrt(rf**2 + cf**2)
        return sf
    
    def row_frequency(self, img):
        """计算行频率"""
        row_diff = img[:,:,:-1,:] - img[:,:,1:,:]
        rf = torch.sqrt(torch.mean(row_diff**2))
        return rf
        
    def column_frequency(self, img):
        """计算列频率"""
        col_diff = img[:,:,:,:-1] - img[:,:,:,1:]
        cf = torch.sqrt(torch.mean(col_diff**2))
        return cf

class TotalLoss:
    def __init__(self):
        weights=[1,10,1e-1] 
        percepWei=[1e-1,1/21,10/21,10/21]
        assert len(percepWei) == 4
        self.pWei = percepWei
        self.vgg = Vgg16().to(device)
        self.vgg.eval()
        self.MSE = nn.MSELoss() 
        self.L1 = nn.L1Loss() 
        self.SSIM =SSIM(data_range=1.0, channel=1) 
        self.SF = SpatialFrequencyLoss() 
        self.VIF = VisualInformationFidelity()
        self.mi_loss = MutualInformationLoss(32,8)
        self.mi_initialized = False  # 跟踪是否已移动到设备
        ## distrubute weights
        self.contentW = weights[0]
        self.pixelW = weights[1]
        self.SF_W = weights[2]
        # self.mi_weight = 1e-3  # 互信息损失权重

    def __call__(self, data_refocus,features,pred,gt): #
        current_device = features[0].device
        event_features = features[0] # b 32 256 256
        frame_features = features[1] # b 32 256 256
        eframe_features = features[2] # b 32 256 256

        event = data_refocus[0].to(current_device)  # B 1 256 256
        frame = data_refocus[1].to(current_device)
        eframe = data_refocus[2].to(current_device)

        
        # # 第一次调用时将互信息损失移到正确设备
        # if not self.mi_initialized:
        #     self.mi_loss = self.mi_loss.to(current_device)
        #     self.mi_initialized = True


        b,c,h,w = pred.shape
        pred = pred.to(current_device) # B 1 256 256

                ## gray to 3-dim image to fit vgg16
        if c == 1:
            pred = pred.repeat(1,3,1,1)
        if gt.shape[1] == 1:
            gt = gt.repeat(1,3,1,1)

        
        # ## gray to 3-dim image to fit vgg16
        # frame = frame.repeat(1, 3, 1, 1) if frame.shape[1] == 1 else frame
        # event = event.repeat(1, 3, 1, 1) if event.shape[1] == 1 else event
        # eframe = eframe.repeat(1, 3, 1, 1) if eframe.shape[1] == 1 else eframe
        # pred = pred.repeat(1, 3, 1, 1) if pred.shape[1] == 1 else pred
            
        ## calculate perceptual loss  frame 和 预测结果
        frame_features = self.vgg(gt)
        pred_features = self.vgg(pred)
        L_frame = []
        for i in range(4):
            recon_frame = frame_features[i]
            recon_hat = pred_features[i]
            L_frame.append(self.MSE(recon_frame,recon_hat))

        L_perceptual = (self.pWei[0]*L_frame[0] + self.pWei[1]*L_frame[1] + self.pWei[2]*L_frame[2] + self.pWei[3]*L_frame[3])

      
        ## calculate pixel loss
        # L_SSIM = self.SSIM(pred,frame) + 1e-4*self.SSIM(pred,eframe) + 1e-2*self.SSIM(pred,event)
        # L_L1 = 10*self.L1(pred,frame) + 1e-4*self.L1(pred,eframe) + 1e-2*self.L1(pred,event)
        # L_SSIM = self.SSIM(pred,gt) 
        L_L1 = 10*self.L1(pred,gt)
        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        # 传统版本
        # diff_i = torch.sum(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        # diff_j = torch.sum(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        # L_tv = (diff_i + diff_j) / float(c * h * w)
        L_SF= self.SF(pred)
        
        # 计算互信息损失
        # L_mutual_info = self.mi_loss(event_features, frame_features, eframe_features)
        ## total lossss
        # total_loss =   1e-2*L_SF + 1e-1*L_mutual_info + 10*L_SSIM
        total_loss =   1e-2*L_SF  + L_L1 +L_perceptual
        

        return total_loss