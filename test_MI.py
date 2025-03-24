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
import utils
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import shutil
import matplotlib.pyplot as plt
from utils import frame_refocus

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
    

def detect_and_manage_anomalies(base_path, mode="train", threshold=0.10, 
                               record_file="anomaly_files.txt", 
                               anomaly_ratio=0.95):
    """
    检测数据集中的异常图像，记录它们，并提供删除选项
    
    Args:
        base_path: 数据集根目录
        mode: 'train' 或 'test'
        threshold: 判断黑白像素的阈值
        record_file: 记录异常文件列表的文件名
        anomaly_ratio: 判断为异常的像素比例阈值
    """
    # 设置目录路径
    event_dir = f"{base_path}/{mode}/Event/"
    aps_dir = f"{base_path}/{mode}/Aps/"
    frame_dir = f"{base_path}/{mode}/Frame/"
    eframe_dir = f"{base_path}/{mode}/Eframe/"
    
    # 检查目录是否存在
    if not os.path.exists(frame_dir):
        print(f"错误: 目录 {frame_dir} 不存在!")
        return
    
    # 获取所有frame文件 (支持.npy, .npz, .png, .jpg)
    frame_files = [f for f in os.listdir(frame_dir) if f.endswith(('.npy', '.npz', '.png', '.jpg'))]
    
    # 创建记录文件
    anomaly_records = []
    
    print(f"开始扫描 {mode} 数据集中的 {len(frame_files)} 个文件...")
    
    # 遍历所有文件检测异常
    for filename in tqdm(frame_files):
        # 读取frame文件
        frame_path = os.path.join(frame_dir, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            if file_ext in ['.npy', '.npz']:
                # 使用numpy读取
                if file_ext == '.npy':
                    frame_data = np.load(frame_path)
                else:  # .npz
                    frame_data = np.load(frame_path)
                    # 假设npz文件中的数据存储在第一个键下
                    frame_data = frame_data[list(frame_data.keys())[0]]
                
                # 转换为PyTorch tensor
                frame_tensor = torch.from_numpy(frame_data).float()
                
                # 确保tensor有正确的维度 [N,C,H,W]
                if len(frame_tensor.shape) == 2:  # [H,W]
                    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                elif len(frame_tensor.shape) == 3:  # [C,H,W]
                    frame_tensor = frame_tensor.unsqueeze(0)  # [1,C,H,W]
                
                # 使用frame_refocus处理
                frame_processed = frame_refocus(frame_tensor, threshold=1e-5, norm_type='minmax')
                
                # 转换为numpy进行分析
                frame = frame_processed.squeeze().cpu().numpy()
            else:
                # 使用opencv读取图像文件
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    print(f"警告: 无法读取文件 {frame_path}")
                    continue
                frame = frame / 255.0  # 归一化
        except Exception as e:
            print(f"处理文件 {frame_path} 时出错: {e}")
            continue
        
        # 计算黑白像素比例
        black_ratio = np.mean(frame < threshold)
        white_ratio = np.mean(frame > (1 - threshold))
        
        # 如果超过设定比例是纯黑或纯白，记录为异常
        if black_ratio > anomaly_ratio or white_ratio > anomaly_ratio:
            # 获取基本文件名（不含扩展名）
            basename = os.path.splitext(filename)[0]
            
            # 获取对应的其他文件路径 - 检查各种可能的扩展名
            event_path = None
            for ext in ['.npz', '.npy', '.png', '.jpg']:
                temp_path = os.path.join(event_dir, basename + ext)
                if os.path.exists(temp_path):
                    event_path = temp_path
                    break
            
            gt_path = None
            for ext in ['.npz', '.npy', '.png', '.jpg']:
                temp_path = os.path.join(aps_dir, basename + ext)
                if os.path.exists(temp_path):
                    gt_path = temp_path
                    break
            
            eframe_path = None
            for ext in ['.npz', '.npy', '.png', '.jpg']:
                temp_path = os.path.join(eframe_dir, basename + ext)
                if os.path.exists(temp_path):
                    eframe_path = temp_path
                    break
            
            # 记录异常信息
            anomaly_info = {
                'filename': basename,
                'black_ratio': black_ratio,
                'white_ratio': white_ratio,
                'frame_path': frame_path,
                'event_path': event_path,
                'gt_path': gt_path,
                'eframe_path': eframe_path
            }
            anomaly_records.append(anomaly_info)
            
            # 打印发现的异常
            print(f"\n发现异常图像: {basename}")
            print(f"  黑色比例: {black_ratio:.4f}, 白色比例: {white_ratio:.4f}")
    
    # 如果没有异常，提前返回
    if not anomaly_records:
        print("未发现异常图像!")
        return
    
    # 将异常记录写入文件
    with open(record_file, 'w') as f:
        f.write(f"发现 {len(anomaly_records)} 个异常图像\n")
        f.write("-" * 80 + "\n")
        
        for i, record in enumerate(anomaly_records):
            f.write(f"异常 #{i+1}: {record['filename']}\n")
            f.write(f"  黑色比例: {record['black_ratio']:.4f}, 白色比例: {record['white_ratio']:.4f}\n")
            f.write(f"  Frame路径: {record['frame_path']}\n")
            f.write(f"  Event路径: {record['event_path'] if record['event_path'] else '未找到'}\n")
            f.write(f"  GT路径: {record['gt_path'] if record['gt_path'] else '未找到'}\n")
            f.write(f"  Eframe路径: {record['eframe_path'] if record['eframe_path'] else '未找到'}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\n已将 {len(anomaly_records)} 个异常图像记录到 {record_file}")
    
    # 提供交互式处理选项
    print("\n请选择处理方式:")
    print("1. 查看并逐个决定是否删除异常文件")
    print("2. 直接删除所有异常文件")
    print("3. 退出不做任何操作")
    
    choice = input("请输入选择 (1/2/3): ")
    
    if choice == '1':
        # 逐个决定是否删除
        for i, record in enumerate(anomaly_records):
            print(f"\n异常 #{i+1}/{len(anomaly_records)}: {record['filename']}")
            print(f"  黑色比例: {record['black_ratio']:.4f}, 白色比例: {record['white_ratio']:.4f}")
            
            sub_choice = input(f"是否删除此异常及相关文件? (y/n/查看图像[v]/退出[q]): ")
            
            if sub_choice.lower() == 'q':
                print("操作已取消，退出处理")
                break
            elif sub_choice.lower() == 'v':
                # 根据文件类型查看图像
                try:
                    file_ext = os.path.splitext(record['frame_path'])[1].lower()
                    if file_ext in ['.npy', '.npz']:
                        # 加载并处理NPY/NPZ文件
                        if file_ext == '.npy':
                            frame_data = np.load(record['frame_path'])
                        else:  # .npz
                            frame_data = np.load(record['frame_path'])
                            frame_data = frame_data[list(frame_data.keys())[0]]
                        
                        frame_tensor = torch.from_numpy(frame_data).float()
                        if len(frame_tensor.shape) == 2:
                            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)
                        elif len(frame_tensor.shape) == 3:
                            frame_tensor = frame_tensor.unsqueeze(0)
                        
                        frame_processed = frame_refocus(frame_tensor, threshold=1e-5, norm_type='minmax')
                        frame_vis = frame_processed.squeeze().cpu().numpy()
                        
                        plt.figure(figsize=(10, 8))
                        plt.imshow(frame_vis, cmap='gray')
                        plt.title(f"异常图像 #{i+1}: {record['filename']}")
                        plt.colorbar()
                        plt.show()
                    else:
                        # 使用OpenCV读取和显示
                        frame = cv2.imread(record['frame_path'])
                        if frame is not None:
                            cv2.imshow(f"异常图像 #{i+1}: {record['filename']}", frame)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                except Exception as e:
                    print(f"显示图像时出错: {e}")
                
                sub_choice = input(f"查看后决定: 是否删除此异常及相关文件? (y/n/退出[q]): ")
                
            if sub_choice.lower() == 'y':
                # 删除相关文件
                files_to_delete = [
                    record['frame_path'],
                    record['event_path'],
                    record['gt_path'],
                    record['eframe_path']
                ]
                
                deleted_count = 0
                for file_path in files_to_delete:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"删除文件 {file_path} 时出错: {e}")
                
                print(f"已删除 {deleted_count} 个与 {record['filename']} 相关的文件")
            elif sub_choice.lower() == 'q':
                print("操作已取消，退出处理")
                break
            else:
                print(f"保留 {record['filename']} 相关文件")
    
    elif choice == '2':
        # 确认是否删除所有异常文件
        confirm = input(f"确定要删除全部 {len(anomaly_records)} 个异常文件集? (y/n): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            file_count = 0
            
            for record in anomaly_records:
                files_to_delete = [
                    record['frame_path'],
                    record['event_path'],
                    record['gt_path'],
                    record['eframe_path']
                ]
                
                for file_path in files_to_delete:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            file_count += 1
                        except Exception as e:
                            print(f"删除 {file_path} 时出错: {e}")
                
                deleted_count += 1
            
            print(f"已删除 {deleted_count}/{len(anomaly_records)} 个异常数据集，共 {file_count} 个文件")
        else:
            print("操作已取消，未删除任何文件")
    
    else:
        print("操作已取消，未删除任何文件")

# 主函数入口
if __name__ == '__main__':
    # 配置参数 - 可以通过命令行参数传入
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "/home_ssd/sjy/EVI-SAI/EF_Dataset"
    
    # 可选参数
    mode = "train" if len(sys.argv) <= 2 else sys.argv[2]
    threshold = 0.10 if len(sys.argv) <= 3 else float(sys.argv[3])
    
    # 执行异常检测和管理
    detect_and_manage_anomalies(
        base_path=base_path,
        mode=mode,
        threshold=threshold,
        record_file=f"anomaly_files_{mode}.txt",
        anomaly_ratio=0.95
    )