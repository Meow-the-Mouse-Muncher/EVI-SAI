import numpy as np
import torch
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
        use_mask: 是否使用mask来过滤非零元素，默认True
    Returns:
        归一化后的tensor，形状为[N,1,H,W]
    """
    if not frame.is_floating_point():
        frame = frame.float()
    
    # 根据use_mask参数决定是否使用mask
    if use_mask:
        mask = frame.abs().gt(threshold)
        summed = torch.sum(frame.masked_fill(~mask, 0), dim=1, keepdim=True)
        valid_channels = torch.sum(mask, dim=1, keepdim=True).clamp_(min=threshold)
        normalized = summed.div(valid_channels)
        normalized.mul_(valid_channels.gt(threshold))
    else:
        # 直接在通道维度上求和
        normalized = torch.sum(frame, dim=1, keepdim=True)
    
    # 使用tensor操作进行归一化
    if norm_type == 'minmax':
        if use_mask:
            non_zero = normalized.abs().gt(threshold)
            if torch.any(non_zero):
                valid_values = normalized.masked_select(non_zero)
                min_val = valid_values.min()
                max_val = valid_values.max()
                scale = max_val.sub(min_val)
                
                if scale > threshold:
                    normalized.sub_(min_val).div_(scale)
                    normalized.mul_(non_zero)
        else:
            min_val = normalized.min()
            max_val = normalized.max()
            scale = max_val.sub(min_val)
            if scale > threshold:
                normalized.sub_(min_val).div_(scale)
    
    elif norm_type == 'mean':
        if use_mask:
            non_zero = normalized.abs().gt(threshold)
            if torch.any(non_zero):
                valid_values = normalized.masked_select(non_zero)
                mean = valid_values.mean()
                std = valid_values.std()
                
                if std > threshold:
                    normalized.sub_(mean).div_(std)
                    normalized.mul_(non_zero)
        else:
            mean = normalized.mean()
            std = normalized.std()
            if std > threshold:
                normalized.sub_(mean).div_(std)
    
    return normalized.clamp_(0, 1)


