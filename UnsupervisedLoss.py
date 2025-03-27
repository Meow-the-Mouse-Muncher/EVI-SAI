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
def adjust(init, fin, step, fin_step):
    if fin_step == 0:
        return  fin
    deta = fin - init
    adj = min(init + deta * step / fin_step, fin)
    return adj
# os.environ['CUDA_VISIBLE_DEVICES']="1" # choose GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cos_sim = nn.CosineSimilarity(dim=1).cuda(device)
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
    


class TenengradSharpnessLoss(nn.Module):
    def __init__(self, threshold=0.1):
        super(TenengradSharpnessLoss, self).__init__()
        self.threshold = threshold
        
        # 定义Sobel算子
        self.sobel_x_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y_kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, x):
        # 移动卷积核到正确的设备
        self.sobel_x_kernel = self.sobel_x_kernel.to(x.device)
        self.sobel_y_kernel = self.sobel_y_kernel.to(x.device)
        
        b, c, h, w = x.shape
        # 并行处理所有通道 - 使用分组卷积
        sobel_x_kernel_expanded = self.sobel_x_kernel.repeat(c, 1, 1, 1)
        sobel_y_kernel_expanded = self.sobel_y_kernel.repeat(c, 1, 1, 1)
        
        # 重新组织输入数据，将每个通道作为单独样本处理
        x_reshaped = x.view(b*c, 1, h, w)
        
        # 应用Sobel算子
        grad_x = F.conv2d(x_reshaped, sobel_x_kernel_expanded, padding=1, groups=c)
        grad_y = F.conv2d(x_reshaped, sobel_y_kernel_expanded, padding=1, groups=c)
        
        # 计算梯度幅度
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # 应用阈值（可选）
        if self.threshold > 0:
            gradient_magnitude = gradient_magnitude * (gradient_magnitude > self.threshold)
        
        # 计算Tenengrad值
        tenengrad_val = torch.mean(gradient_magnitude**2)
        
        # 归一化到合理范围
        sharpness = torch.tanh(tenengrad_val / 100)
        return 1 - sharpness  # 返回损失值
    


    
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
    def __init__(self,MI_loss_model):
        weights=[1,10,1e-1] 
        percepWei=[1e-1,1/21,10/21,10/21]
        assert len(percepWei) == 4
        self.pWei = percepWei
        self.vgg = Vgg16().to(device)
        self.vgg.eval()
        self.MSE = nn.MSELoss() 
        self.L1 = nn.L1Loss() 
        self.SSIM =SSIM(data_range=1.0, channel=1,size_average=False) 
        self.SF = SpatialFrequencyLoss()
        self.VIF = VisualInformationFidelity().to(device)
        self.mi_loss = MI_loss_model
        self.sharpness_loss = TenengradSharpnessLoss()
        ## distrubute weights
        self.contentW = weights[0]
        self.pixelW = weights[1]
        self.SF_W = weights[2]
        # self.mi_weight = 1e-3  # 互信息损失权重

    def __call__(self, data_refocus,features,ori_features,pred,gt,weight_EF,epoch,num_epochs,sam_model=None): #
        event_features = features[0] # b 32 256 256
        frame_features = features[1] # b 32 256 256
        eframe_features = features[2] # b 32 256 256

        event_refocus = data_refocus[0]  # B 1 256 256
        frame_refocus = data_refocus[1]
        eframe_refocus = data_refocus[2]

        e_frame = ori_features[0]
        f_frame = ori_features[1]
        ef_frame = ori_features[2]

        e_weight =  weight_EF[:, 0:1, :, :] 
        f_weight =  weight_EF[:, 1:2, :, :]
        ef_weight =  weight_EF[:, 2:3, :, :]
 
        # ## gray to 3-dim image to fit vgg16
        # frame = frame.repeat(1, 3, 1, 1) if frame.shape[1] == 1 else frame
        # event = event.repeat(1, 3, 1, 1) if event.shape[1] == 1 else event
        # eframe = eframe.repeat(1, 3, 1, 1) if eframe.shape[1] == 1 else eframe
        # pred = pred.repeat(1, 3, 1, 1) if pred.shape[1] == 1 else pred
            
        ## calculate perceptual loss  frame 和 预测结果
        ## gray to 3-dim image to fit vgg16
        # if c == 1:
        #     pred = pred.repeat(1,3,1,1)
        # if gt.shape[1] == 1:
        #     gt = gt.repeat(1,3,1,1)
        # frame_features = self.vgg(gt)
        # pred_features = self.vgg(pred)
        # L_frame = []
        # for i in range(4):
        #     recon_frame = frame_features[i]
        #     recon_hat = pred_features[i]
        #     L_frame.append(self.MSE(recon_frame,recon_hat))

        # L_perceptual = (self.pWei[0]*L_frame[0] + self.pWei[1]*L_frame[1] + self.pWei[2]*L_frame[2] + self.pWei[3]*L_frame[3])

      
        ## calculate pixel loss
        # L_SSIM = -(weight_EF[1]*self.SSIM(pred,frame) + weight_EF[2]*self.SSIM(pred,eframe) + weight_EF[0]*self.SSIM(pred,event))
        
        # L_L1 = self.L1(pred,frame) + 1e-4*self.L1(pred,eframe) + 1e-2*self.L1(pred,event)
        # L_SSIM = self.SSIM(pred,gt) 
        # L_L1 = 10*self.L1(pred,gt)
        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        # 传统版本
        # diff_i = torch.sum(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        # diff_j = torch.sum(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        # L_tv = (diff_i + diff_j) / float(c * h * w)

        L_sharpness_loss = self.sharpness_loss(pred)
        # L_SF= 1-self.SF(pred) # 空间分辨率最好高一点
        # 计算 SimSiam 损失
        simsiam_e_loss, simsiam_f_loss, simsiam_ef_loss = sam_model(pred, e_frame, f_frame, ef_frame,weight_EF)
        simsiam_e_loss = torch.mean(simsiam_e_loss)
        simsiam_f_loss = torch.mean(simsiam_f_loss)
        simsiam_ef_loss = torch.mean(simsiam_ef_loss)
        # 计算特征间的互信息损失
        mi_f_e ,mi_f_ef ,mi_ef_f = self.mi_loss(event_features, frame_features, eframe_features)
        mi_f_e = torch.mean(mi_f_e)
        mi_f_ef = torch.mean(mi_f_ef)
        mi_ef_f = torch.mean(mi_ef_f)
        # # 计算 SSIM 损失
        ssim_e , ssim_f , ssim_ef = self.SSIM(pred,event_refocus), self.SSIM(pred,frame_refocus), self.SSIM(pred,eframe_refocus)
        eps = 1e-5
        #记录一下loss
        # loss_unnormalized = L_sharpness_loss \
        # +ssim_e+ssim_f+ssim_ef\
        # +mi_f_e+mi_f_ef+mi_ef_f\
        # +simsiam_e_loss+simsiam_f_loss+simsiam_ef_loss
        
        # 提取的特征之间的互信息损失最小化
        L_mutual_info = 3*mi_f_e/(mi_f_e.abs()+eps) \
                        + mi_f_ef/(mi_f_ef.abs()+eps)\
                        + mi_ef_f/(mi_ef_f.abs()+eps)
        # 预测图和输入图的互信息损失最大化
        simsiam_loss_total = simsiam_e_loss/(simsiam_e_loss.abs()+eps) \
                            + simsiam_f_loss/(simsiam_f_loss.abs()+eps) \
                            + simsiam_ef_loss/(simsiam_ef_loss.abs()+eps)
        # 预测图和fsai方法的ssim最大化 加权平均啊
        L_SSIM = -((e_weight*ssim_e + f_weight*ssim_f + ef_weight*ssim_ef)/(\
                e_weight.abs()+f_weight.abs()+ef_weight.abs()+eps)\
                    ).mean()
        L_SSIM = -(0.5*ssim_e + 5*ssim_f + 0.05*ssim_ef).mean()
        # 平均梯度最大化
        L_sharpness_loss = L_sharpness_loss/(L_sharpness_loss.abs()+eps)

        ## total lossss
        # * adjust(0, 1, epoch, num_epochs) 
        total_loss =   1e-1*L_sharpness_loss + L_SSIM+ L_mutual_info + simsiam_loss_total
        print(f"e_weight.grad={e_weight.grad},f_weight.grad={f_weight.grad},ef_weight.grad={ef_weight.grad}")
        print(L_mutual_info.grad)
        # total_loss =   L_sharpness_loss + 100*L_SSIM + L_mutual_info + simsiam_loss_total

        

        return total_loss