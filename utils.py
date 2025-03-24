import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

@torch.no_grad()
def tb_image(opt,tb,step,group,name,srcs,num_vis=(4,8),from_range=(0,1),cmap="gray"):
    images = preprocess_vis_image(opt,srcs,from_range=from_range,cmap=cmap)
    num_H,num_W = num_vis 
    images = images[:num_H*num_W]
    image_grid = torchvision.utils.make_grid(images[:,:3],nrow=num_W,pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:,3:],nrow=num_W,pad_value=1.)[:1]
        image_grid = torch.cat([image_grid,mask_grid],dim=0)
    tag = f"{group}/{name}"
    tb.add_image(tag,image_grid,step)

def preprocess_vis_image(opt,images,from_range=(0,1),cmap="gray"):
    min,max = from_range
    images = (images-min)/(max-min)
    images = images.clamp(min=0,max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt,images.squeeze(1).cpu(),cmap=cmap)
    return images

def get_heatmap(opt,gray,cmap): # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())  # N H W 4
    color = torch.from_numpy(color[...,:3]).permute(0,3,1,2).float() # [N,3,H,W]
    return color

@torch.no_grad()
def frame_refocus(frame, threshold=1e-6, norm_type='minmax', use_mask=False):
    """
    优化版本的frame_refocus函数，专门针对PyTorch tensor操作
    Args:
        frame: 输入tensor，形状为[N,C,H,W]
        threshold: 非零元素的判定阈值，默认1e-6
        norm_type: 归一化方式，可选 'minmax' 或 'mean'，默认'minmax'
        use_mask: 是否使用mask来过滤非零元素，默认False
    Returns:
        归一化后的tensor，形状为[N,1,H,W]
    """
    if not frame.is_floating_point():
        frame = frame.float()
    frame_copy = frame.clone()
    batch_size = frame.shape[0]
    
    # 根据use_mask参数决定是否使用mask
    if use_mask:
        mask = frame_copy.abs().gt(threshold)
        summed = torch.sum(frame_copy.masked_fill(~mask, 0), dim=1, keepdim=True)
        valid_channels = torch.sum(mask, dim=1, keepdim=True).clamp_(min=threshold)
        normalized = summed.div(valid_channels)
        normalized.mul_(valid_channels.gt(threshold))
    else:
        # 直接在通道维度上求和
        normalized = torch.sum(frame_copy, dim=1, keepdim=True)
    
    # 使用批量操作进行归一化 - 避免for循环
    if norm_type == 'minmax':
        if use_mask:
            non_zero = normalized.abs().gt(threshold)
            # 为每个样本计算最小值和最大值
            flattened = normalized.view(batch_size, -1)
            flattened_mask = non_zero.view(batch_size, -1)
            
            # 创建一个大值填充矩阵用于min计算
            # 和一个小值填充矩阵用于max计算
            filled_for_min = flattened.clone()
            filled_for_max = flattened.clone()
            
            # 将非有效元素替换为不影响min/max计算的值
            filled_for_min.masked_fill_(~flattened_mask, float('inf'))
            filled_for_max.masked_fill_(~flattened_mask, float('-inf'))
            
            # 并行计算每个样本的min/max
            min_vals = filled_for_min.min(dim=1, keepdim=True)[0]
            max_vals = filled_for_max.max(dim=1, keepdim=True)[0]
            
            # 处理全部被掩盖的情况（所有值都是inf/-inf）
            all_masked = (~torch.any(flattened_mask, dim=1)).view(-1, 1)
            min_vals.masked_fill_(all_masked, 0.0)
            max_vals.masked_fill_(all_masked, 0.0)
            
            # 重塑为广播兼容的形状
            min_vals = min_vals.view(batch_size, 1, 1, 1)
            max_vals = max_vals.view(batch_size, 1, 1, 1)
            scales = (max_vals - min_vals).clamp(min=threshold)
            
            # 创建一个掩码，表示哪些样本需要归一化
            valid_scales = scales > threshold
            
            # 对每个样本应用min-max归一化
            normalized = torch.where(
                valid_scales,
                (normalized - min_vals) / scales,
                normalized
            )
            # 应用非零掩码
            normalized = normalized * non_zero.float()
        else:
            # 对每个样本单独计算min/max
            min_vals = normalized.view(batch_size, -1).min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
            max_vals = normalized.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
            scales = (max_vals - min_vals).clamp(min=threshold)
            
            # 创建一个掩码，表示哪些样本需要归一化
            valid_scales = scales > threshold
            
            # 对每个样本应用min-max归一化
            normalized = torch.where(
                valid_scales,
                (normalized - min_vals) / scales,
                normalized
            )
    
    elif norm_type == 'mean':
        if use_mask:
            non_zero = normalized.abs().gt(threshold)
            # 为每个样本计算均值和标准差
            flattened = normalized.view(batch_size, -1)
            flattened_mask = non_zero.view(batch_size, -1)
            
            # 使用掩码和加权操作并行计算均值
            # 创建浮点型掩码(1.0表示有效值, 0.0表示无效值)
            float_mask = flattened_mask.float()
            
            # 计算每个样本的有效元素数量
            valid_counts = float_mask.sum(dim=1, keepdim=True).clamp(min=threshold)
            
            # 计算每个样本有效元素的总和
            masked_sums = (flattened * float_mask).sum(dim=1, keepdim=True)
            
            # 计算均值 (总和/有效数量)
            means = masked_sums / valid_counts
            
            # 计算标准差: sqrt(E[(X - mean)²])
            # 先计算每个样本中(x - mean)²的和
            sq_diff_sum = ((flattened - means) * float_mask).pow(2).sum(dim=1, keepdim=True)
            
            # 计算标准差
            stds = (sq_diff_sum / valid_counts).sqrt()
            
            # 处理全部被掩盖的情况
            all_masked = (valid_counts <= threshold).view(-1, 1)
            means.masked_fill_(all_masked, 0.0)
            stds.masked_fill_(all_masked, 0.0)
            
            # 重塑为广播兼容的形状
            means = means.view(batch_size, 1, 1, 1)
            stds = stds.view(batch_size, 1, 1, 1)
            
            # 创建一个掩码，表示哪些样本需要归一化
            valid_stds = stds > threshold
            
            # 对每个样本应用均值-标准差归一化
            normalized = torch.where(
                valid_stds,
                (normalized - means) / stds.clamp(min=threshold),
                normalized
            )
            # 应用非零掩码
            normalized = normalized * non_zero.float()
        else:
            # 对每个样本单独计算均值和标准差
            means = normalized.view(batch_size, -1).mean(dim=1, keepdim=True).view(batch_size, 1, 1, 1)
            stds = normalized.view(batch_size, -1).std(dim=1, keepdim=True).view(batch_size, 1, 1, 1)
            
            # 创建一个掩码，表示哪些样本需要归一化
            valid_stds = stds > threshold
            
            # 对每个样本应用均值-标准差归一化
            normalized = torch.where(
                valid_stds,
                (normalized - means) / stds.clamp(min=threshold),
                normalized
            )
    
    return torch.clamp(normalized, 0, 1)



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
        # mi_fe = self.mi_frame_event(frame, event)
        # mi_ff = self.mi_frame_eframe(frame, eframe)
        # mi_ef = self.mi_event_eframe(event, eframe)

        # 返回总的互信息损失
        return mi_fe + 0.1*mi_ff + 0.1*mi_ef


