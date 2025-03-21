import os
import cv2
import yaml
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import sys
from code.EF_Dataset import Dataset_EFNet
from code.Networks.EF_SAI_Net import EF_SAI_Net
from UnsupervisedLoss import TotalLoss
from code.Networks.submodules import SimSiam,SimSiamLight
import torchvision.models as models
import utils
from pytorch_msssim import SSIM
import torch.nn as nn
from torch.distributions import kl, Normal, Independent
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"  # choose GPU
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

def check_unused_parameters(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} has no gradient")

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/mnt/home_ssd/sjy/EF-SAI/EF_Dataset", help="validation aps path")
    # 添加分布式训练相关参数
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)))
    opts=parser.parse_args()
    local_rank = opts.local_rank

    with open(os.path.abspath('./config.yaml'),'r') as f:
        opt = edict(yaml.safe_load(f))
    # seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    # 首先初始化分布式环境
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'  # 使用环境变量初始化
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    # data
    train_dataset = Dataset_EFNet(mode="train",base_path=opts.base_path, norm=False)
    # 然后创建分布式采样器
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.bs,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # train_dataloader = DataLoader(train_dataset,batch_size=opt.bs,pin_memory=True,num_workers=4,shuffle=True)
    # test_dataset = Dataset_EFNet(mode='test',base_path=opts.base_path, norm=False)
    # test_dataloader = DataLoader(test_dataset,batch_size=1,pin_memory=True,num_workers=4,shuffle=False)
    # save dir
    # 创建结果目录
    results_dir = os.path.abspath('./Results')
    os.makedirs(results_dir,exist_ok=True)
    exp_name = opt.exp_name
    os.makedirs(f"{results_dir}/{exp_name}",exist_ok=True)

    net = EF_SAI_Net()
    # net = torch.nn.DataParallel(net)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.cuda(local_rank)
    net = DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=False)


    # sam_model = SimSiam(models.__dict__['resnet50'],dim=2048,pred_dim=512 )

    # sam_model = SimSiamLight(dim=512, pred_dim=128)
    # sam_model = sam_model.cuda(local_rank)
    # sam_model = DistributedDataParallel(sam_model, device_ids=[local_rank], find_unused_parameters=False)


    if local_rank == 0:  # 只在主进程记录日志
        tb = SummaryWriter(log_dir=f"{results_dir}/{exp_name}", flush_secs=10)  
        
    if os.path.exists(f"{results_dir}/{exp_name}/model/checkpoint.pth"):
        # 加载到正确设备并处理DDP模型
        checkpoint = torch.load(f"{results_dir}/{exp_name}/model/checkpoint.pth", 
                          map_location=f'cuda:{local_rank}')
        net.module.load_state_dict(checkpoint)
    net = net.train()
    optimizer = torch.optim.Adam(net.parameters(),lr=opt.lr) # default: 5e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,64)
    criterion = TotalLoss()
    # train
    max_epochs = opt.max_epoch
    fix_bn_epochs = opt.fix_bn_epoch
    save_model_epochs = opt.save_model_epoch
    save_skip = opt.save_skip
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=(5.e-5/5.e-4)**(1/max_epochs))
    ssim_module = SSIM(data_range=1.0, channel=1) 
    torch.autograd.set_detect_anomaly(True)  # 开启异常检测
    for epoch in range(max_epochs):
        train_sampler.set_epoch(epoch)  # 添加这行确保每个epoch数据分布不同
        # BN eval 在xx次数后冻结bn的参数
        if epoch == fix_bn_epochs:
            print("turn BN to eval mode ...")
            net.apply(eval_bn)
        # Train
        loss_record = dict()
        loss_record['train'],loss_record['val'] = 0,0
        psnr_record = dict()
        psnr_record['train'],psnr_record['val'] = 0,0
        fsai_psnr=0
        fsai_ssim =0
        ssim_record = dict()
        ssim_record['train'],ssim_record['val'] = 0,0
        # train
        accumulation_steps = 4  # 每4步更新一次参数
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description_str(f'[epoch {epoch}|Train]')
            net = net.train()
            for i,(event,frame,eframe,gt) in enumerate(train_dataloader):
                time_step = event.shape[1]
                event = event.to(device)
                gt = gt.to(device)
                frame = frame.to(device)
                eframe = eframe.to(device)
                net.zero_grad()
                optimizer.zero_grad()
                ## 数据集提前归一化好了
                pred,features = net(event, frame, eframe, time_step)
                check_unused_parameters(net.module)

                frame_refocus=utils.frame_refocus(frame, threshold=1e-5, norm_type='minmax')
                eframe_refocus=utils.frame_refocus(eframe, threshold=1e-5, norm_type='minmax')
                # 将event n c 2 h w 转换为 n c h w  对2 进行绝对值求和并归一化
                event_refocus = torch.sum(torch.abs(event),dim=2,keepdim=False) 
                event_refocus = (event_refocus - event_refocus.min())/(event_refocus.max()-event_refocus.min())
                event_refocus=utils.frame_refocus(event_refocus, threshold=1e-5, norm_type='minmax')
                refocus_data = [event_refocus,frame_refocus,eframe_refocus]

                # # 计算 SimSiam 损失
                # cos_sim = nn.CosineSimilarity(dim=1).cuda(local_rank)
                
                # # 计算各对图像之间的 SimSiam 损失
                # p1_ep, p2_ep, z1_ep, z2_ep = sam_model(x1=pred.clone(), x2=event_refocus.clone())
                # simsiam_loss_ep = -(cos_sim(p1_ep, z2_ep.detach()).mean() + cos_sim(p2_ep, z1_ep.detach()).mean()) * 0.5

                # # 2. frame_refocus 和 pred 之间的损失
                # p1_fp, p2_fp, z1_fp, z2_fp = sam_model(x1=pred.clone(), x2=frame_refocus.clone())
                # simsiam_loss_fp = -(cos_sim(p1_fp, z2_fp.detach()).mean() + cos_sim(p2_fp, z1_fp.detach()).mean()) * 0.5

                # # 3. eframe_refocus 和 pred 之间的损失
                # p1_efp, p2_efp, z1_efp, z2_efp = sam_model(x1=pred.clone(), x2=eframe_refocus.clone())
                # simsiam_loss_efp = -(cos_sim(p1_efp, z2_efp.detach()).mean() + cos_sim(p2_efp, z1_efp.detach()).mean()) * 0.5

                # # 合并所有 SimSiam 损失
                # simsiam_loss_total = simsiam_loss_ep *1e-1 + simsiam_loss_fp * 1e1 + simsiam_loss_efp * 1e-3

                content_loss = criterion(refocus_data,features,pred)
                # loss = content_loss +  1e-1 * simsiam_loss_total
                loss = content_loss
                loss_record['train'] += loss.item()
                loss.backward()
                optimizer.step()
                
                # 清理缓存
                torch.cuda.empty_cache()
                    
                psnr_record['train'] += psnr(pred,gt)
                ssim_record['train'] += ssim_module(pred,gt)
                fsai_psnr+=psnr(frame_refocus,gt)
                fsai_ssim+=ssim_module(frame_refocus,gt)
                pbar.set_postfix_str(f"loss:{loss.item():.4f}")
                pbar.update(1)
            if local_rank == 0:  # 只在主进程记录日志
                tb.add_scalar(f"train/loss",loss_record['train']/len(train_dataloader),epoch)
                tb.add_scalar(f"train/psnr",psnr_record['train']/len(train_dataloader),epoch)
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
                        net = net.eval()
                        event = event.to(device)
                        gt = gt.to(device)
                        frame = frame.to(device)
                        eframe = eframe.to(device)
                        pred,_ = net(event, frame, eframe, time_step)
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
            scheduler.step()

        # eval
        '''
        with torch.no_grad():
            with tqdm(total=len(test_dataloader)) as pbar:
                pbar.set_description_str(f'[epoch {epoch}|Val]')
                net = net.eval()
                for i,(event,frame,eframe,gt) in enumerate(test_dataloader):
                    time_step = event.shape[1]
                    event = event.to(device)
                    gt = gt.to(device)
                    frame = frame.to(device)
                    eframe = eframe.to(device) 
                    net.zero_grad()
                    optimizer.zero_grad()
                    # event,frame,eframe,gt = event.to(device),(frame/255.0).unsqueeze(dim=1).to(device),(eframe/255.0).unsqueeze(dim=1).to(device),(gt/255.0).unsqueeze(dim=1).to(device)
                    pred = net(event, frame, eframe, time_step)
                    loss = criterion(pred,gt)
                    loss_record['val'] += loss.item()
                    psnr_record['val'] += psnr(pred,gt)
                    pbar.set_postfix_str(f"loss:{loss.item():.4f}")
                    pbar.update(1)
                    if i in [0,1]:
                        utils.tb_image(opt,tb,epoch,'val',f"pred_{i:04d}",pred[0:1,...])
                        utils.tb_image(opt,tb,epoch,'val',f"gt_{i:04d}",gt[0:1,...])
                tb.add_scalar(f"val/loss",loss_record['val']/len(test_dataloader),epoch)
                tb.add_scalar(f"val/psnr",psnr_record['val']/len(test_dataloader),epoch)
                print(f"[epoch {epoch}|val]: average loss: {loss_record['val']/len(test_dataloader)}.")
                print(f"[epoch {epoch}|val]: average psnr: {psnr_record['val']/len(test_dataloader)}.")
        '''
        # save model
        if local_rank == 0:  # 只在主进程记录日志
            if epoch % save_model_epochs == 0 or epoch == max_epochs:
                os.makedirs(f"{results_dir}/{exp_name}/model",exist_ok=True)
                torch.save(net.module.state_dict(),f"{results_dir}/{exp_name}/model/epoch_{epoch:04d}.pth")

