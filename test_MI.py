from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision import models
from torch import randn
from pytorch_msssim import SSIM
from torchmetrics.image import VisualInformationFidelity
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import logging_presets
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
        # mi_fe = torch.clip(self.mi_frame_event(frame, event),-30,30)
        # mi_ff = torch.clip(self.mi_frame_eframe(frame, eframe),-30,30)
        # mi_ef = torch.clip(self.mi_event_eframe(event, eframe),-30,30)
        mi_fe = self.mi_frame_event(frame, event)
        mi_ff = self.mi_frame_eframe(frame, eframe)
        mi_ef = self.mi_event_eframe(event, eframe)

        # 返回总的互信息损失
        return mi_fe + 0.1*mi_ff + 0.1*mi_ef
class ImprovedMutualInfoLoss(nn.Module):
    def __init__(self, reduction='mean', temperature=1.0, negative_weight=0.5):
        """
        改进的互信息损失函数
        
        参数:
            reduction (str): 如何处理批次维度('mean'或'sum')
            temperature (float): 温度参数，控制相似度分布的锐度
            negative_weight (float): 控制负样本对的权重
        """
        super(ImprovedMutualInfoLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.negative_weight = negative_weight
        self.eps = 1e-8  # 避免数值问题的小常数
        
    def compute_similarity(self, x, y):
        """计算两个特征张量之间的余弦相似度"""
        # 将特征展平为2D tensor: [batch_size, features]
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        y_flat = y.view(batch_size, -1)
        
        # 归一化特征
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)
        
        # 计算余弦相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.mm(x_norm, y_norm.t()) / self.temperature
        
        # 对角线元素是正样本对（同一位置的特征）
        positive_samples = torch.diag(similarity_matrix)
        
        # 非对角线元素是负样本对（不同位置的特征）
        # 创建一个掩码矩阵，对角线为0，其他位置为1
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
        negative_samples = similarity_matrix[mask].view(batch_size, -1)
        
        # 计算每个样本的对比损失
        # 正样本对的相似度除以所有相似度的和
        pos_term = positive_samples
        neg_term = torch.logsumexp(negative_samples, dim=1) * self.negative_weight
        
        # 互信息估计: 正样本相似度减去负样本相似度
        mi_estimate = pos_term - neg_term
        
        # 应用减少方法
        if self.reduction == 'mean':
            return mi_estimate.mean()
        elif self.reduction == 'sum':
            return mi_estimate.sum()
        else:
            return mi_estimate
        
    def forward(self, event, frame, eframe):
        """
        计算三种特征之间的互信息
        
        参数:
            event: 事件特征 [batch, channels, height, width]
            frame: 帧特征 [batch, channels, height, width]
            eframe: 事件帧特征 [batch, channels, height, width]
            
        返回:
            加权互信息估计
        """
        # 计算不同特征对之间的互信息
        mi_ef = self.compute_similarity(event, frame)  # 事件与帧
        mi_ee = self.compute_similarity(event, eframe) # 事件与事件帧
        mi_fe = self.compute_similarity(frame, eframe) # 帧与事件帧
        
        # 可配置的权重
        w1, w2, w3 = 1.0, 0.5, 0.5
        
        # 返回加权互信息估计（高值表示高互信息）
        return w1 * mi_ef + w2 * mi_ee + w3 * mi_fe
    

if __name__ == '__main__':
    a= torch.randn(4, 32, 256, 256)
    # normal = torch.sum(a, dim=1, keepdim=True)
    # max_val = normal.max(a,
    # # 创建两组相同的特征（互信息最大）
    # mi_loss = MutualInformationLoss(32, 8)
    # identical_features = torch.randn(4, 32, 256, 256)
    # mi_same = mi_loss(identical_features, identical_features, identical_features)

    # # 创建两组独立的特征（互信息较低）
    # independent_features1 = torch.randn(4, 32, 256, 256)
    # independent_features2 = torch.randn(4, 32, 256, 256)
    # independent_features3 = torch.randn(4, 32, 256, 256)
    # mi_different = mi_loss(independent_features1, independent_features2, independent_features3)

    # print(f"相同特征的互信息损失: {mi_same.item()}")
    # print(f"不同特征的互信息损失: {mi_different.item()}")