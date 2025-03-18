import torch
import torch.nn as nn
from torchvision import models
from torch import randn
from pytorch_msssim import SSIM
from torchmetrics.image import VisualInformationFidelity
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
        ## distrubute weights
        self.contentW = weights[0]
        self.pixelW = weights[1]
        self.tvW = weights[2]
        self.frame_weight = 1
        self.eframe_weight = 1e-4
        self.event_weight = 1e-1

    def __call__(self, frame,event,eframe,pred): #
        
        frame = frame.to(device)
        event = event.to(device)
        eframe = eframe.to(device)
        pred = pred.to(device)

        b, c, h, w = frame.shape
        ## gray to 3-dim image to fit vgg16
        frame = frame.repeat(1, 3, 1, 1) if frame.shape[1] == 1 else frame
        event = event.repeat(1, 3, 1, 1) if event.shape[1] == 1 else event
        eframe = eframe.repeat(1, 3, 1, 1) if eframe.shape[1] == 1 else eframe
        pred = pred.repeat(1, 3, 1, 1) if pred.shape[1] == 1 else pred
            
        ## calculate perceptual loss  frame 和 预测结果
        frame_features = self.vgg(frame)
        eframe_features = self.vgg(eframe)
        event_features = self.vgg(event)
        pred_features = self.vgg(pred)
        L_frame = []
        L_eframe = []
        L_event = []
        for i in range(4):
            recon_frame = frame_features[i]
            recon_event = event_features[i]
            recon_eframe = eframe_features[i]
            recon_hat = pred_features[i]
            L_frame.append(self.MSE(recon_frame,recon_hat))
            L_eframe.append(self.MSE(recon_eframe,recon_hat))
            L_event.append(self.MSE(recon_event,recon_hat))
        L_perceptual = self.frame_weight*(self.pWei[0]*L_frame[0] + self.pWei[1]*L_frame[1] + self.pWei[2]*L_frame[2] + self.pWei[3]*L_frame[3])
        +self.eframe_weight*(self.pWei[0]*L_eframe[0] + self.pWei[1]*L_eframe[1] + self.pWei[2]*L_eframe[2] + self.pWei[3]*L_eframe[3])
        +self.event_weight*(self.pWei[0]*L_event[0] + self.pWei[1]*L_event[1] + self.pWei[2]*L_event[2] + self.pWei[3]*L_event[3])
      
        ## calculate pixel loss
        # L_SSIM = self.frame_weight*self.SSIM(pred,frame) + self.eframe_weight*self.SSIM(pred,eframe) + self.event_weight*self.SSIM(pred,event)
        L_L1 = self.frame_weight*self.L1(pred,frame) + self.eframe_weight*self.L1(pred,eframe) + self.event_weight*self.L1(pred,event)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        # 传统版本
        # diff_i = torch.sum(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        # diff_j = torch.sum(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        # L_tv = (diff_i + diff_j) / float(c * h * w)
        L_SF= self.SF(pred)
        
        ## total loss
        total_loss = self.contentW*L_perceptual + self.pixelW*L_L1 + self.tvW*L_SF
        
        return total_loss

