import os
import cv2
import yaml
import gc
import random
import numpy as np
import argparse
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from math import log10
import torch.distributed as dist
import sys
from code.EF_Dataset import Dataset_EFNet
from code.Networks.EF_SAI_Net import EF_SAI_Net
from UnsupervisedLoss import TotalLoss
from code.Networks.submodules import SimSiam,SimSiamLight_loss,MutualInformationLoss
import torchvision.models as models
import utils
from pytorch_msssim import SSIM
import torch.nn as nn
from torch.distributions import kl, Normal, Independent
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"  # choose GPU

def eval_bn(m):
    if type(m) == torch.nn.BatchNorm2d:
        m.eval()
        
def train_bn(m):
    if type(m) == torch.nn.BatchNorm2d:
        m.train()

def psnr(img1:Tensor,img2:Tensor):
    assert img1.shape == img2.shape
    mse = torch.sum((img1-img2)**2)/img1.numel()
    psnr = 10*log10(1/mse)
    return psnr
def adjust(init, fin, step, fin_step):
    if fin_step == 0:
        return  fin
    deta = fin - init
    adj = min(init + deta * step / fin_step, fin)
    return adj
#
    

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/home_ssd/sjy/EVI-SAI/EF_Dataset", help="validation aps path")
    opts=parser.parse_args()

    with open(os.path.abspath('./config.yaml'),'r') as f:
        opt = edict(yaml.safe_load(f))
    # seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    # data

    train_dataset = Dataset_EFNet(mode="test",base_path=opts.base_path, norm=False)
    train_dataloader = DataLoader(train_dataset,batch_size=opt.bs,pin_memory=True,num_workers=min(os.cpu_count()//2,8),shuffle=True,drop_last=True )
    test_dataset = Dataset_EFNet(mode='test',base_path=opts.base_path, norm=False)
    test_dataloader = DataLoader(test_dataset,batch_size=opt.bs,pin_memory=True,num_workers=min(os.cpu_count()//2,8),shuffle=True,drop_last=True )
    # save dir
    # 创建结果目录
    results_dir = os.path.abspath('./Results')
    os.makedirs(results_dir,exist_ok=True)
    exp_name = opt.exp_name
    os.makedirs(f"{results_dir}/{exp_name}",exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #查看gpu数量
    print(f"gpu num :{torch.cuda.device_count()}")

    net = EF_SAI_Net()
    net = net.to(device) 
    net = torch.nn.DataParallel(net)


    # sam_model = SimSiam(models.__dict__['resnet50'],dim=2048,pred_dim=512 )
    sam_model = SimSiamLight_loss(dim=512, pred_dim=128,input_channels=30)
    sam_model = sam_model.to(device) 
    sam_model = torch.nn.DataParallel(sam_model)
    # 无bn和droupt层，所以MIloss不需要train和eval
    MILoss_model = MutualInformationLoss(32, 8)
    MILoss_model = MILoss_model.to(device)
    MILoss_model = torch.nn.DataParallel(MILoss_model)

    tb = SummaryWriter(log_dir=f"{results_dir}/{exp_name}", flush_secs=10)  
        
    if os.path.exists(f"{results_dir}/{exp_name}/model/checkpoint.pth"):
        print("load model from checkpoint ...")
        checkpoint = torch.load(f"{results_dir}/{exp_name}/model/checkpoint.pth", map_location=device)
        # 加载所有模型权重
        net.module.load_state_dict(checkpoint['net'])
        sam_model.module.load_state_dict(checkpoint['sam_model'])
        MILoss_model.module.load_state_dict(checkpoint['MILoss_model'])
    net.train()
    sam_model.train()
    params = list(net.parameters()) + list(sam_model.parameters()) + list(MILoss_model.parameters())
    optimizer = torch.optim.Adam(params,lr=opt.lr) # default: 5e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,128)

    criterion = TotalLoss(MILoss_model)
    # train
    max_epochs = opt.max_epoch
    fix_bn_epochs = opt.fix_bn_epoch
    save_model_epochs = opt.save_model_epoch
    save_skip = opt.save_skip
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=(5.e-5/5.e-4)**(1/max_epochs))
    ssim_module = SSIM(data_range=1.0, channel=1) 
   
    for epoch in range(max_epochs):
        # # BN eval 在xx次数后冻结bn的参数
        # if epoch == fix_bn_epochs:
        #     print("turn BN to eval mode ...")
        #     net.apply(eval_bn)
        # Train
        loss_record = dict()
        loss_record['train'],loss_record['val'] = 0,0
        psnr_record = dict()
        psnr_record['train'],psnr_record['val'] = 0,0
        fsai_psnr=0
        fsai_ssim =0
        ssim_record = dict()
        ssim_record['train'],ssim_record['val'] = 0,0

        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description_str(f'[epoch {epoch}|Train]')
            net.train()
            sam_model.train()
            for i,(event,frame,eframe,gt) in enumerate(train_dataloader):
                time_step = event.shape[1]
                event = event.to(device)
                gt = gt.to(device)
                frame = frame.to(device)
                eframe = eframe.to(device)
                optimizer.zero_grad(set_to_none=True)  # 更彻底地清理梯度
                ## 数据集提前归一化好了
                pred,features,weight_EF = net(event, frame, eframe, time_step)

                frame_refocus=utils.frame_refocus(frame, threshold=1e-5, norm_type='minmax')
                eframe_refocus=utils.frame_refocus(eframe, threshold=1e-5, norm_type='minmax')
                # 将event n c 2 h w 转换为 n c h w  对2 进行绝对值求和并归一化
                event_frame = torch.sum(torch.abs(event),dim=2,keepdim=False) 
                event_frame = (event_frame - event_frame.min())/(event_frame.max()-event_frame.min()+1e-5)
                event_refocus=utils.frame_refocus(event_frame, threshold=1e-5, norm_type='minmax')
                refocus_data = [event_refocus,frame_refocus,eframe_refocus]
                # event_features,frame_features,eframe_features = features[0],features[1],features[2]   # b 32 256 256
                ori_image = [event_frame,frame,eframe]
                loss = criterion(refocus_data,features,ori_image,pred,gt,weight_EF,epoch,max_epochs,sam_model)
                loss_record['train'] += loss.item()
                loss.backward()
                optimizer.step()   

                psnr_record['train'] += psnr(pred,gt)
                ssim_record['train'] += ssim_module(pred,gt).item()
                fsai_psnr+=psnr(frame_refocus,gt)
                fsai_ssim+=ssim_module(frame_refocus,gt).item()
                pbar.set_postfix_str(f"loss:{loss.item():.4f}")
                pbar.update(1)
                # tb.add_scalar(f"train/loss",loss_record['train']/len(train_dataloader),epoch)
                # tb.add_scalar(f"train/psnr",psnr_record['train']/len(train_dataloader),epoch)
                # tb.add_scalar(f"train/ssim",ssim_record['train']/len(train_dataloader),epoch)
                # tb.add_scalar(f"train/fsai_ssim", fsai_ssim/len(train_dataloader),epoch)
                # tb.add_scalar(f"train/fsai_psnr",fsai_psnr/len(train_dataloader),epoch)
                # tb.add_scalar(f"train/lr",optimizer.param_groups[0]["lr"],epoch)
            tb.add_scalar(f"train/psnr",psnr_record['train']/len(train_dataloader),epoch)
            tb.add_scalar(f"train/loss",loss_record['train']/len(train_dataloader),epoch)
            tb.add_scalar(f"train/ssim",ssim_record['train']/len(train_dataloader),epoch)
            tb.add_scalar(f"train/fsai_ssim", fsai_ssim/len(train_dataloader),epoch)
            tb.add_scalar(f"train/fsai_psnr",fsai_psnr/len(train_dataloader),epoch)
            tb.add_scalar(f"train/lr",optimizer.param_groups[0]["lr"],epoch)
            print(f"[epoch {epoch}|train]: average loss: {loss_record['train']/len(train_dataloader)}.")
            print(f"[epoch {epoch}|train]: average psnr: {psnr_record['train']/len(train_dataloader)}.")
            # view
            with torch.no_grad():
                for i,(event,frame,eframe,gt) in enumerate(train_dataloader):
                    if (i+1) == save_skip:
                        time_step = event.shape[1]
                        net.eval()
                        sam_model.eval()
                        event = event.to(device)
                        gt = gt.to(device)
                        frame = frame.to(device)
                        eframe = eframe.to(device)
                        pred,_,weight_EF= net(event, frame, eframe, time_step)
                        frame_refocus=utils.frame_refocus(frame, threshold=1e-5, norm_type='minmax')
                        eframe_refocus=utils.frame_refocus(eframe, threshold=1e-5, norm_type='minmax')
                        # 将event n c 2 h w 转换为 n c h w  对2 进行绝对值求和并归一化
                        event_refocus = torch.sum(torch.abs(event),dim=2,keepdim=False) 
                        event_refocus = (event_refocus - event_refocus.min())/(event_refocus.max()-event_refocus.min())
                        event_refocus=utils.frame_refocus(event_refocus, threshold=1e-5, norm_type='minmax')
                        utils.tb_image(opt,tb,epoch,'train',f"event_refocus_{i:04d}",event_refocus[0:1,...])
                        utils.tb_image(opt,tb,epoch,'train',f"eframe_refocus_{i:04d}",eframe_refocus[0:1,...])
                        utils.tb_image(opt,tb,epoch,'train',f"frame_refocus_{i:04d}",frame_refocus[0:1,...])
                        utils.tb_image(opt,tb,epoch,'train',f"pred_{i:04d}",pred[0:1,...])
                        utils.tb_image(opt,tb,epoch,'train',f"gt_{i:04d}",(gt[0:1,...]))
                        tb.add_scalar(f"train/e_weight",weight_EF[0].sum().item(),epoch)
                        tb.add_scalar(f"train/f_weight",weight_EF[1].sum().item(),epoch)
                        tb.add_scalar(f"train/ef_weight",weight_EF[2].sum().item(),epoch)
                        break
            scheduler.step()
        # eval
        # with torch.no_grad():
        #     with tqdm(total=len(test_dataloader)) as pbar:
        #         pbar.set_description_str(f'[epoch {epoch}|Val]')
        #         net.eval()
        #         sam_model.eval()
        #         for i,(event,frame,eframe,gt) in enumerate(test_dataloader):
        #             time_step = event.shape[1]
        #             event = event.to(device)
        #             gt = gt.to(device)
        #             frame = frame.to(device)
        #             eframe = eframe.to(device) 
        #             ## 数据集提前归一化好了
        #             pred,features,weight_EF = net(event, frame, eframe, time_step)
        #             frame_refocus=utils.frame_refocus(frame, threshold=1e-5, norm_type='minmax')
        #             eframe_refocus=utils.frame_refocus(eframe, threshold=1e-5, norm_type='minmax')
        #             # 将event n c 2 h w 转换为 n c h w  对2 进行绝对值求和并归一化
        #             event_frame = torch.sum(torch.abs(event),dim=2,keepdim=False) 
        #             event_refocus = (event_frame - event_frame.min())/(event_frame.max()-event_frame.min())
        #             event_refocus=utils.frame_refocus(event_refocus, threshold=1e-5, norm_type='minmax')
        #             refocus_data = [event_refocus,frame_refocus,eframe_refocus]
        #             # event_features,frame_features,eframe_features = features[0],features[1],features[2]   # b 32 256 256
        #             ori_image = [event_frame,frame,eframe]
        #             loss= criterion(refocus_data,features,ori_image,pred,gt,weight_EF,epoch,max_epochs,sam_model)
        #             loss_record['val'] += loss.item()
        #             psnr_record['val'] += psnr(pred,gt)
        #             pbar.set_postfix_str(f"loss:{loss.item():.4f}")
        #             pbar.update(1)
        #             if i in [0,1]:
        #                 utils.tb_image(opt,tb,epoch,'val',f"pred_{i:04d}",pred[0:1,...])
        #                 utils.tb_image(opt,tb,epoch,'val',f"gt_{i:04d}",gt[0:1,...])
        #         tb.add_scalar(f"val/loss",loss_record['val']/len(test_dataloader),epoch)
        #         tb.add_scalar(f"val/psnr",psnr_record['val']/len(test_dataloader),epoch)
        #         print(f"[epoch {epoch}|val]: average loss: {loss_record['val']/len(test_dataloader)}.")
        #         print(f"[epoch {epoch}|val]: average psnr: {psnr_record['val']/len(test_dataloader)}.")
        # 清理一下缓存
        gc.collect()
        torch.cuda.empty_cache()
        # save model
        if epoch % save_model_epochs == 0 or epoch == max_epochs:
            checkpoint = {
                'net': net.module.state_dict(),
                'sam_model': sam_model.module.state_dict(), 
                'MILoss_model': MILoss_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            os.makedirs(f"{results_dir}/{exp_name}/model",exist_ok=True)
            torch.save(checkpoint, f"{results_dir}/{exp_name}/model/epoch_{epoch:04d}.pth")

