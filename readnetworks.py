from __future__ import print_function
import netron
import torch.onnx
from torch.autograd import Variable
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import os
import cv2 as cv
import argparse
import numpy as np
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # choose GPU
from code.Networks.EF_SAI_Net import EF_SAI_Net
from code.EF_Dataset import Dataset_EFNet
from prefetch_generator import BackgroundGenerator
def eval_bn(m):
    if type(m) == torch.nn.BatchNorm2d:
        m.eval()

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
##===================================================##
##********** Configure training settings ************##
##===================================================##
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/home_ssd/sjy/EF-SAI/EF_Dataset", help="validation aps path")
    opts=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    model_path = "./PreTrained/EF_SAI_Net.pth"

    img_w = 256
    img_h = 256
    #valDataset = TrainSet_Hybrid_singleF(opts.valEvent, opts.valAPS, opts.valFrame, opts.valEframe, norm=False, aps_n = 0, if_exposure = False, exposure = 0.04)
    valDataset = Dataset_EFNet(mode='test',base_path=opts.base_path, norm=False)
    valLoader = DataLoaderX(valDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)

    # ##===================================================##
    # ##****************** Create model *******************##
    # ##===================================================##
    print("Begin testing network ...")
    net = EF_SAI_Net()
    net = net.to(device)

    # 添加详细的模型加载调试
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型权重
        print(f"正在加载模型: {model_path}")
        state_dict = torch.load(model_path)
        
        # 打印模型结构信息
        print("\n==== 模型结构 ====")
        print(net)
        
        # 比较模型和权重文件中的键
        model_keys = set(net.state_dict().keys())
        weight_keys = set(state_dict.keys())
        
        print("\n==== 键比较 ====")
        print(f"模型中的键数量: {len(model_keys)}")
        print(f"权重文件中的键数量: {len(weight_keys)}")
        
        missing_keys = model_keys - weight_keys
        unexpected_keys = weight_keys - model_keys
        
        if missing_keys:
            print("\n缺失的键:")
            for key in missing_keys:
                print(f"- {key}")
        
        if unexpected_keys:
            print("\n多余的键:")
            for key in unexpected_keys:
                print(f"- {key}")
        
        # 尝试不同的加载方式
        try:
            # 1. 标准加载
            net.load_state_dict(state_dict)
            print("标准方式加载成功!")
        except Exception as e1:
            print(f"\n标准加载失败: {str(e1)}")
            
            try:
                # 2. 处理多GPU训练的模型
                print("\n尝试移除'module.'前缀...")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "") 
                    new_state_dict[name] = v
                net.load_state_dict(new_state_dict)
                print("移除'module.'前缀后加载成功!")
            except Exception as e2:
                print(f"\n移除前缀加载失败: {str(e2)}")
                
                try:
                    # 3. 非严格加载（忽略不匹配的键）
                    print("\n尝试非严格加载...")
                    net.load_state_dict(state_dict, strict=False)
                    print("非严格方式加载成功（部分权重可能未加载）!")
                except Exception as e3:
                    print(f"\n非严格加载也失败: {str(e3)}")
                    raise RuntimeError("无法加载模型权重，请检查模型结构是否匹配")
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        exit(1)

    net = net.eval()
    if (True):
        print("turn BN to eval mode ...")
        net.apply(eval_bn)


    # modelData = "./demo.pth"  # 定义模型数据保存的路径
    modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 

    # ##===================================================##
    # ##****************** Test model *******************##
    # ##===================================================##
    with torch.no_grad():
        # 获取一个样本数据
        event, frame, eframe, gt = next(iter(valLoader))
        
        # 确保数据在正确的设备上
        event = event.to(device)
        frame = frame.to(device)
        eframe = eframe.to(device)
        timeWin = event.shape[1]
        
        # 添加更多输入输出信息
        torch.onnx.export(net, 
                     (event, frame, eframe, timeWin),
                     modelData,
                     input_names=['event', 'frame', 'eframe', 'time_window'],
                     output_names=['output'],
                     opset_version=11,
                     keep_initializers_as_inputs=False,  # 减少初始化参数显示
                     do_constant_folding=True,  # 折叠常量
                     operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                     verbose=False)  # 关闭详细输出
        
        # 启动 netron 查看器
        print(f"正在打开模型可视化，保存路径: {modelData}")
        netron.start(modelData)
        exit()


