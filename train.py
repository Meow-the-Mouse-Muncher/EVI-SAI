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
import sys
from code.EF_Dataset import Dataset_EFNet
from code.Networks.EF_SAI_Net import EF_SAI_Net
from loss import TotalLoss
import utils
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # choose GPU

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

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/home_ssd/sjy/EF-SAI/EF_Dataset", help="validation aps path")
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
    train_dataloader = DataLoader(train_dataset,batch_size=opt.bs,pin_memory=True,num_workers=4,shuffle=True)
    test_dataset = Dataset_EFNet(mode='test',base_path=opts.base_path, norm=False)
    test_dataloader = DataLoader(test_dataset,batch_size=1,pin_memory=True,num_workers=4,shuffle=False)
    # save dir
    results_dir = os.path.abspath('./Results')
    os.makedirs(results_dir,exist_ok=True)
    exp_name = opt.exp_name
    os.makedirs(f"{results_dir}/{exp_name}",exist_ok=True)
    tb = SummaryWriter(log_dir=f"{results_dir}/{exp_name}",flush_secs=10)
    # net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = EF_SAI_Net()
    net = torch.nn.DataParallel(net)
    if os.path.exists(f"{results_dir}/{exp_name}/model/checkpoint.pth"):
        net.load_state_dict(torch.load(f"{results_dir}/{exp_name}/model/checkpoint.pth"))
    net = net.to(device)
    net = net.train()
    optimizer = torch.optim.Adam(net.parameters(),lr=opt.lr) # default: 5e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,64)
    criterion = TotalLoss(weights=[1e0,8,2e-4],percepWei=[1e-1,1/21,10/21,10/21])
    # criterion = TotalLoss(weights=[1e0,32,2e-4],percepWei=[1e-1,1/21,10/21,10/21])
    # train
    max_epochs = opt.max_epoch
    fix_bn_epochs = opt.fix_bn_epoch
    save_model_epochs = opt.save_model_epoch
    save_skip = opt.save_skip
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=(5.e-5/5.e-4)**(1/max_epochs))
    for epoch in range(max_epochs):
        # BN eval 在xx次数后冻结bn的参数
        if epoch == fix_bn_epochs:
            print("turn BN to eval mode ...")
            net.apply(eval_bn)
        # Train
        loss_record = dict()
        loss_record['train'],loss_record['val'] = 0,0
        psnr_record = dict()
        psnr_record['train'],psnr_record['val'] = 0,0
        # train
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
                pred = net(event, frame, eframe, time_step)
                loss = criterion(pred,gt)
                loss_record['train'] += loss.item()
                loss.backward()
                # #  梯度裁剪防止爆炸
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimizer.step()
                psnr_record['train'] += psnr(pred,gt)
                pbar.set_postfix_str(f"loss:{loss.item():.4f}")
                pbar.update(1)
            tb.add_scalar(f"train/loss",loss_record['train']/len(train_dataloader),epoch)
            tb.add_scalar(f"train/psnr",psnr_record['train']/len(train_dataloader),epoch)
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
                        pred = net(event, frame, eframe, time_step)
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
        if epoch % save_model_epochs ==0 or epoch == max_epochs:
            os.makedirs(f"{results_dir}/{exp_name}/model",exist_ok=True)
            torch.save(net.state_dict(),f"{results_dir}/{exp_name}/model/epoch_{epoch:04d}.pth")

