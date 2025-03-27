from torch.utils.data import Dataset
import torch
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def getFileName(path,suffix):
    ## function used to get file names
    NameList=[]
    FileList = os.listdir(path)
    for i in FileList:
        if os.path.splitext(i)[1] == suffix:
            NameList.append(i)
    return NameList

def normalization(data, maxR = 0.99, minR = 40):
    ## normalize data into range (0, 1), input format: tensor
    Imin = data.min()
    Irange = data.max() - Imin
    Imax = Imin + Irange * maxR
    Imin = Imin + Irange * minR
    data = (data - Imin) / (Imax - Imin)
    data.clamp_(0.0, 1.0)
    return data

def reshapeTimeStep(pos, neg, timeStep):
    interval = int(pos.shape[0] / timeStep)
    posNew = np.zeros((timeStep, pos.shape[1], pos.shape[2]))
    negNew = np.zeros((timeStep, neg.shape[1], neg.shape[2]))
    for i in range(timeStep):
        posNew[i,:] = pos[i*interval:(i+1)*interval,:].sum(0)
        negNew[i,:] = neg[i*interval:(i+1)*interval,:].sum(0)
    return posNew, negNew
class Dataset_EFNet(Dataset):
    def __init__(self, mode, base_path, norm=False, timeStep=30, validate=False):
        ## EventDir: directory path of event data for train
        ## FrameDir: directory path of frame data
        ## EframeDir: directory path of E→F data
        ## GTDir: directory path of APS data for train
        self.mode = mode
        self.base_path = base_path
        self.EventDir = f"{self.base_path}/{self.mode}/Event/"
        self.GTDir = f"{self.base_path}/{self.mode}/Aps/"
        self.FrameDir = f"{self.base_path}/{self.mode}/Frame/"
        self.EframeDir = f"{self.base_path}/{self.mode}/Eframe/"
        self.timeStep = timeStep
        self.norm = norm

        self.EventNames = getFileName(self.EventDir, '.npy')
        self.GTNames = getFileName(self.GTDir, '.png')
        self.FrameNames = getFileName(self.FrameDir, '.npy')
        self.EframeNames = getFileName(self.EframeDir, '.npy')
        self.EventNames.sort()
        self.GTNames.sort()
        self.FrameNames.sort()
        self.EframeNames.sort()

        # 文件验证相关
        self.error_log_path = os.path.join(base_path, f'{mode}_corrupt_files.txt')
        self.valid_indices = []  # 默认所有索引都有效
        if validate:
            print(f"正在验证{mode}数据集文件...")
            self.validate_files()
        else:
            self.valid_indices = list(range(len(self.EventNames))) 

    def validate_files(self):
        with open(self.error_log_path, 'w') as f:
            for idx in range(len(self.EventNames)):
                if idx %50==0:
                    print(idx)
                try:
                    EventPath = os.path.join(self.EventDir, self.EventNames[idx])
                    GTPath = os.path.join(self.GTDir, self.GTNames[idx])
                    FramePath = os.path.join(self.FrameDir, self.FrameNames[idx])
                    EframePath = os.path.join(self.EframeDir, self.EframeNames[idx])
                    
                    # 检查文件是否存在
                    if not all(os.path.exists(p) for p in [EventPath, GTPath, FramePath, EframePath]):
                        f.write(f"文件不存在: {self.EventNames[idx]}\n")
                        continue
                    
                    # 检查npy文件
                    try:
                        np.load(EventPath, allow_pickle=True)
                        np.load(FramePath)
                        np.load(EframePath)
                    except Exception as e:
                        f.write(f"NPY文件损坏: {self.EventNames[idx]}, 错误: {str(e)}\n")
                        continue
                    
                    # 检查PNG文件
                    try:
                        from PIL import Image
                        Image.open(GTPath).verify()  # verify()会检查文件完整性
                    except Exception as e:
                        f.write(f"PNG文件损坏: {self.GTNames[idx]}, 错误: {str(e)}\n")
                        continue
                    
                    # 如果所有检查都通过，添加到有效索引列表
                    self.valid_indices.append(idx)
                except Exception as e:
                    f.write(f"未知错误: {self.EventNames[idx]}, 错误: {str(e)}\n")

        print(f"有效文件数: {len(self.valid_indices)}/{len(self.EventNames)}")
        if len(self.valid_indices) == 0:
            raise RuntimeError("没有有效的数据文件！")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # 使用valid_indices获取实际的索引
        real_index = self.valid_indices[index]
        
        EventName = self.EventNames[real_index]
        GTName = self.GTNames[real_index]
        FrameName = self.FrameNames[real_index]
        EframeName = self.EframeNames[real_index]

        # 构建完整路径
        EventPath = os.path.join(self.EventDir, EventName)
        GTPath = os.path.join(self.GTDir, GTName)
        FramePath = os.path.join(self.FrameDir, FrameName)
        EframePath = os.path.join(self.EframeDir, EframeName)
        ## warping event data
        EventData = np.load(EventPath, allow_pickle=True).item()
        pos = EventData['Pos']
        neg = EventData['Neg']
        assert pos.shape[0] % self.timeStep == 0, "Inappropriate time step"
        if (pos.shape[0] != self.timeStep):
            pos, neg = reshapeTimeStep(pos, neg, self.timeStep)
        pos = np.expand_dims(pos, axis=1)
        neg = np.expand_dims(neg, axis=1)
        EventInput = torch.FloatTensor(np.concatenate((pos, neg), axis=1))  ## EventInput = (step, channel, H, W)

        ## warping aps data
        GTInput = plt.imread(GTPath, plt.cm.gray)
        if GTInput.ndim == 3:  # get 1 dim image
            GTInput = GTInput[:, :, 0]
        GTInput = torch.FloatTensor(np.expand_dims(GTInput, axis=0))

        FrameData = np.load(FramePath)
        EframeData = np.load(EframePath)
        FrameInput = torch.FloatTensor(FrameData)
        EframeInput = torch.FloatTensor(EframeData)
        if self.norm:
            EventInput = normalization(EventInput)
            GTInput = normalization(GTInput)
            FrameInput = normalization(FrameInput)
            EframeInput = normalization(EframeInput)

        return EventInput, FrameInput, EframeInput, GTInput