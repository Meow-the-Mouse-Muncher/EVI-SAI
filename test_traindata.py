import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import yaml
from tqdm import tqdm
import sys
import csv

from codes.EF_Dataset import Dataset_EFNet
from codes.Networks.EF_SAI_Net import EF_SAI_Net
from codes.Networks.submodules import SimSiamLight
import utils

def check_dataset(dataset, num_samples=5):
    """检查数据集中的样本是否包含 NaN 值"""
    print(f"检查数据集中的 {num_samples} 个样本...")
    for i in range(min(num_samples, len(dataset))):
        event, frame, eframe, gt = dataset[i]
        print(f"\n样本 {i}:")
        check_tensor_for_nan(event, f"样本 {i} 的事件数据")
        check_tensor_for_nan(frame, f"样本 {i} 的帧数据")
        check_tensor_for_nan(eframe, f"样本 {i} 的eframe数据")
        check_tensor_for_nan(gt, f"样本 {i} 的真值数据")
        
        # 检查数据范围
        print(f"  事件数据范围: [{event.min():.4f}, {event.max():.4f}]")
        print(f"  帧数据范围: [{frame.min():.4f}, {frame.max():.4f}]")
        print(f"  eframe数据范围: [{eframe.min():.4f}, {eframe.max():.4f}]")
        print(f"  真值数据范围: [{gt.min():.4f}, {gt.max():.4f}]")

def check_tensor_for_nan(tensor, tensor_name="未命名张量"):
    """检查张量中是否包含 NaN 值"""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total_elements = tensor.numel()
        nan_percentage = (nan_count / total_elements) * 100
        print(f"警告: {tensor_name} 包含 {nan_count} 个 NaN 值 (占比 {nan_percentage:.2f}%)")
        
        # 如果是 4D 张量，按通道检查 NaN
        if len(tensor.shape) == 4:
            for c in range(tensor.shape[1]):
                channel_nan = torch.isnan(tensor[:, c]).sum().item()
                if channel_nan > 0:
                    print(f"  - 通道 {c}: {channel_nan} 个 NaN 值")
        return True
    
    # 检查inf值
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        total_elements = tensor.numel()
        inf_percentage = (inf_count / total_elements) * 100
        print(f"警告: {tensor_name} 包含 {inf_count} 个 Inf 值 (占比 {inf_percentage:.2f}%)")
        return True
        
    return False

def check_all_dataset_samples(dataset, output_file="problem_samples.csv"):
    """检查数据集中的所有样本，记录有问题的样本信息"""
    print(f"扫描整个数据集中的所有样本...")
    
    # 创建CSV文件记录有问题的数据
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['sample_index', 'problem_type', 'data_type', 'min_value', 'max_value', 'nan_count', 'inf_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 使用tqdm显示进度
        for i in tqdm(range(len(dataset)), desc="检查数据集样本"):
            try:
                event, frame, eframe, gt = dataset[i]
                
                # 检查每个张量
                problem_found = False
                
                for data_type, data in [('event', event), ('frame', frame), ('eframe', eframe), ('gt', gt)]:
                    # 检查NaN值
                    has_nan = torch.isnan(data).any().item()
                    nan_count = torch.isnan(data).sum().item() if has_nan else 0
                    
                    # 检查Inf值
                    has_inf = torch.isinf(data).any().item()
                    inf_count = torch.isinf(data).sum().item() if has_inf else 0
                    
                    # 检查极端值 (可选，根据数据特性设置阈值)
                    min_val = data.min().item()
                    max_val = data.max().item()
                    extreme_values = abs(min_val) > 1e6 or abs(max_val) > 1e6
                    
                    if has_nan or has_inf or extreme_values:
                        problem_found = True
                        problem_type = []
                        if has_nan: problem_type.append("NaN")
                        if has_inf: problem_type.append("Inf")
                        if extreme_values: problem_type.append("Extreme")
                        
                        # 记录问题数据
                        writer.writerow({
                            'sample_index': i,
                            'problem_type': '+'.join(problem_type),
                            'data_type': data_type,
                            'min_value': min_val,
                            'max_value': max_val,
                            'nan_count': nan_count,
                            'inf_count': inf_count
                        })
                
                # 如果样本有问题，打印详细信息
                if problem_found:
                    print(f"\n问题样本 {i}:")
                    # 尝试获取文件名 (需要修改Dataset_EFNet类以支持此功能)
                    try:
                        if hasattr(dataset, 'get_sample_path'):
                            file_path = dataset.get_sample_path(i)
                            print(f"文件路径: {file_path}")
                    except:
                        pass
                    
                    print(f"  事件数据范围: [{event.min():.4f}, {event.max():.4f}], NaN: {torch.isnan(event).sum().item()}, Inf: {torch.isinf(event).sum().item()}")
                    print(f"  帧数据范围: [{frame.min():.4f}, {frame.max():.4f}], NaN: {torch.isnan(frame).sum().item()}, Inf: {torch.isinf(frame).sum().item()}")
                    print(f"  eframe数据范围: [{eframe.min():.4f}, {eframe.max():.4f}], NaN: {torch.isnan(eframe).sum().item()}, Inf: {torch.isinf(eframe).sum().item()}")
                    print(f"  真值数据范围: [{gt.min():.4f}, {gt.max():.4f}], NaN: {torch.isnan(gt).sum().item()}, Inf: {torch.isinf(gt).sum().item()}")
            
            except Exception as e:
                print(f"\n处理样本 {i} 时出错: {e}")
                # 记录出错的样本
                writer.writerow({
                    'sample_index': i,
                    'problem_type': 'Exception',
                    'data_type': 'all',
                    'min_value': 'N/A',
                    'max_value': 'N/A',
                    'nan_count': 'N/A',
                    'inf_count': 'N/A'
                })
    
    # 显示问题样本统计
    try:
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            problem_count = sum(1 for _ in reader)
        
        print(f"\n发现 {problem_count} 个问题样本，已保存到 {output_file}")
    except:
        print("无法读取问题样本统计")

def analyze_dataset_statistics(dataset, output_file="dataset_stats.csv"):
    """分析数据集的统计信息，寻找异常值"""
    print("分析数据集统计信息...")
    
    stats = {
        'event': {'min': float('inf'), 'max': float('-inf'), 'mean_list': [], 'std_list': []},
        'frame': {'min': float('inf'), 'max': float('-inf'), 'mean_list': [], 'std_list': []},
        'eframe': {'min': float('inf'), 'max': float('-inf'), 'mean_list': [], 'std_list': []},
        'gt': {'min': float('inf'), 'max': float('-inf'), 'mean_list': [], 'std_list': []}
    }
    
    # 收集所有样本的统计数据
    for i in tqdm(range(min(len(dataset), 1000)), desc="收集统计数据"):  # 最多使用1000个样本进行统计
        try:
            event, frame, eframe, gt = dataset[i]
            
            # 更新最大最小值
            for data_type, data in [('event', event), ('frame', frame), ('eframe', eframe), ('gt', gt)]:
                if not torch.isnan(data).any() and not torch.isinf(data).any():
                    stats[data_type]['min'] = min(stats[data_type]['min'], data.min().item())
                    stats[data_type]['max'] = max(stats[data_type]['max'], data.max().item())
                    stats[data_type]['mean_list'].append(data.mean().item())
                    stats[data_type]['std_list'].append(data.std().item())
        except:
            continue
    
    # 计算均值和标准差
    for data_type in stats:
        if stats[data_type]['mean_list']:
            stats[data_type]['mean'] = np.mean(stats[data_type]['mean_list'])
            stats[data_type]['std'] = np.mean(stats[data_type]['std_list'])
            stats[data_type]['mean_std'] = np.std(stats[data_type]['mean_list'])
            stats[data_type]['std_std'] = np.std(stats[data_type]['std_list'])
    
    # 输出结果
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['数据类型', '最小值', '最大值', '均值', '标准差', '均值的标准差', '标准差的标准差'])
        
        for data_type in stats:
            if 'mean' in stats[data_type]:
                writer.writerow([
                    data_type,
                    f"{stats[data_type]['min']:.4f}",
                    f"{stats[data_type]['max']:.4f}",
                    f"{stats[data_type]['mean']:.4f}",
                    f"{stats[data_type]['std']:.4f}",
                    f"{stats[data_type]['mean_std']:.4f}",
                    f"{stats[data_type]['std_std']:.4f}"
                ])
    
    print(f"数据集统计信息已保存到 {output_file}")
    return stats

def find_outliers_in_dataset(dataset, stats, output_file="outlier_samples.csv"):
    """根据统计数据找出异常样本"""
    print("寻找异常值样本...")
    
    # 设置异常检测阈值 (均值 ± 3 * 标准差)
    thresholds = {}
    for data_type in stats:
        if 'mean' in stats[data_type]:
            mean = stats[data_type]['mean']
            std = stats[data_type]['std']
            thresholds[data_type] = {
                'min_mean': mean - 3 * stats[data_type]['mean_std'],
                'max_mean': mean + 3 * stats[data_type]['mean_std'],
                'min_std': std - 3 * stats[data_type]['std_std'],
                'max_std': std + 3 * stats[data_type]['std_std']
            }
    
    # 检查异常样本
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['sample_index', 'data_type', 'reason', 'value', 'threshold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 遍历数据集
        outlier_count = 0
        for i in tqdm(range(len(dataset)), desc="检查异常值"):
            try:
                event, frame, eframe, gt = dataset[i]
                is_outlier = False
                
                # 检查每种数据
                for data_type, data in [('event', event), ('frame', frame), ('eframe', eframe), ('gt', gt)]:
                    if data_type not in thresholds:
                        continue
                        
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        continue  # 已经在之前的检查中标记过
                    
                    # 检查均值是否异常
                    mean_val = data.mean().item()
                    if mean_val < thresholds[data_type]['min_mean'] or mean_val > thresholds[data_type]['max_mean']:
                        writer.writerow({
                            'sample_index': i,
                            'data_type': data_type,
                            'reason': '均值异常',
                            'value': f"{mean_val:.4f}",
                            'threshold': f"[{thresholds[data_type]['min_mean']:.4f}, {thresholds[data_type]['max_mean']:.4f}]"
                        })
                        is_outlier = True
                    
                    # 检查标准差是否异常
                    std_val = data.std().item()
                    if std_val < thresholds[data_type]['min_std'] or std_val > thresholds[data_type]['max_std']:
                        writer.writerow({
                            'sample_index': i,
                            'data_type': data_type,
                            'reason': '标准差异常',
                            'value': f"{std_val:.4f}",
                            'threshold': f"[{thresholds[data_type]['min_std']:.4f}, {thresholds[data_type]['max_std']:.4f}]"
                        })
                        is_outlier = True
                
                if is_outlier:
                    outlier_count += 1
            except:
                continue
        
        print(f"找到 {outlier_count} 个异常值样本，已保存到 {output_file}")

def find_filenames_of_problematic_samples(dataset, problem_indices):
    """尝试找出有问题样本的文件名"""
    print("尝试获取有问题样本的文件名...")
    
    filenames = {}
    
    # 检查Dataset类是否有提供文件名的方法
    has_filename_method = hasattr(dataset, 'get_filename') or hasattr(dataset, 'get_sample_path')
    
    if not has_filename_method:
        print("警告: 数据集类没有提供获取文件名的方法，无法确定具体文件名")
        return filenames
    
    # 获取有问题样本的文件名
    for idx in problem_indices:
        try:
            if hasattr(dataset, 'get_filename'):
                filenames[idx] = dataset.get_filename(idx)
            elif hasattr(dataset, 'get_sample_path'):
                filenames[idx] = dataset.get_sample_path(idx)
        except Exception as e:
            print(f"获取样本 {idx} 的文件名时出错: {e}")
    
    return filenames

def modify_dataset_class():
    """尝试修改Dataset_EFNet类，添加获取文件名的方法"""
    # 检查Dataset_EFNet类是否已有get_filename方法
    if hasattr(Dataset_EFNet, 'get_filename') or hasattr(Dataset_EFNet, 'get_sample_path'):
        return True
    
    # 如果没有，尝试查看Dataset_EFNet类的实现细节
    try:
        import inspect
        source = inspect.getsource(Dataset_EFNet)
        print("Dataset_EFNet类的实现:\n")
        print(source[:500] + "...")  # 只打印前500个字符
        
        # 分析类结构，查找包含文件路径的变量
        print("\n分析Dataset_EFNet类结构...")
        init_method = inspect.getsource(Dataset_EFNet.__init__)
        print("__init__方法:\n")
        print(init_method)
        
        # 查找可能的文件列表变量
        getitem_method = inspect.getsource(Dataset_EFNet.__getitem__)
        print("\n__getitem__方法:\n")
        print(getitem_method)
        
        # 提示用户如何修改Dataset_EFNet类
        print("\n要获取样本文件名，请修改Dataset_EFNet类，添加get_filename方法")
        return False
    except Exception as e:
        print(f"分析Dataset_EFNet类时出错: {e}")
        return False

def check_dataloader_with_progress(dataloader, device, num_batches=None):
    """带进度条地检查数据加载器中的所有批次"""
    print(f"\n检查数据加载器中的所有批次...")
    problem_batches = []
    
    with tqdm(total=len(dataloader) if num_batches is None else min(num_batches, len(dataloader))) as pbar:
        for i, (event, frame, eframe, gt) in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break
                
            # 检查CPU张量
            has_problem = False
            for data_name, data in [("event", event), ("frame", frame), ("eframe", eframe), ("gt", gt)]:
                if check_tensor_for_nan(data, f"批次 {i} 的 {data_name}"):
                    has_problem = True
            
            # 检查GPU张量
            event_cuda = event.to(device)
            frame_cuda = frame.to(device)
            eframe_cuda = eframe.to(device)
            gt_cuda = gt.to(device)
            
            for data_name, data in [("event", event_cuda), ("frame", frame_cuda), ("eframe", eframe_cuda), ("gt", gt_cuda)]:
                if check_tensor_for_nan(data, f"批次 {i} 的 {data_name} (GPU)"):
                    has_problem = True
            
            if has_problem:
                problem_batches.append(i)
            
            pbar.update(1)
    
    if problem_batches:
        print(f"发现 {len(problem_batches)} 个问题批次: {problem_batches}")
    else:
        print("所有批次均正常")
    
    return problem_batches

def main():
    parser = argparse.ArgumentParser(description="检查训练数据中的 NaN 值")
    parser.add_argument("--base_path", type=str, default="/home_ssd/sjy/EVI-SAI/EF_Dataset", help="数据集路径")
    parser.add_argument("--check_grad", action="store_true", help="是否检查梯度")
    parser.add_argument("--check_all", action="store_true", help="检查所有数据样本")
    parser.add_argument("--results_dir", type=str, default="./data_check_results", help="结果保存目录")
    args = parser.parse_args()

    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 载入配置
    with open(os.path.abspath('./config.yaml'), 'r') as f:
        opt = edict(yaml.safe_load(f))
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        # 创建数据集
        print("创建数据集...")
        train_dataset = Dataset_EFNet(mode="train", base_path=args.base_path, norm=False)
        test_dataset = Dataset_EFNet(mode="test", base_path=args.base_path, norm=False)
        
        # 尝试修改数据集类以获取文件名
        modify_dataset_class()
        
        if args.check_all:
            # 检查所有数据集样本
            train_problem_file = os.path.join(args.results_dir, "train_problem_samples.csv")
            test_problem_file = os.path.join(args.results_dir, "test_problem_samples.csv")
            
            print("\n检查所有训练数据样本...")
            check_all_dataset_samples(train_dataset, train_problem_file)
            
            print("\n检查所有测试数据样本...")
            check_all_dataset_samples(test_dataset, test_problem_file)
            
            # 分析数据集统计信息
            train_stats_file = os.path.join(args.results_dir, "train_dataset_stats.csv")
            test_stats_file = os.path.join(args.results_dir, "test_dataset_stats.csv")
            
            print("\n分析训练数据集统计信息...")
            train_stats = analyze_dataset_statistics(train_dataset, train_stats_file)
            
            print("\n分析测试数据集统计信息...")
            test_stats = analyze_dataset_statistics(test_dataset, test_stats_file)
            
            # 查找异常样本
            train_outliers_file = os.path.join(args.results_dir, "train_outlier_samples.csv")
            test_outliers_file = os.path.join(args.results_dir, "test_outlier_samples.csv")
            
            print("\n查找训练数据集异常值...")
            find_outliers_in_dataset(train_dataset, train_stats, train_outliers_file)
            
            print("\n查找测试数据集异常值...")
            find_outliers_in_dataset(test_dataset, test_stats, test_outliers_file)
        else:
            # 只检查少量样本
            print("\n检查少量训练数据样本...")
            check_dataset(train_dataset, num_samples=5)
        
        # 创建数据加载器
        print("\n创建数据加载器...")
        train_dataloader = DataLoader(train_dataset, batch_size=opt.bs, 
                                     pin_memory=True, num_workers=0, 
                                     shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, 
                                    pin_memory=True, num_workers=0, 
                                    shuffle=False)
        
        # 检查数据加载器
        print("\n检查训练数据加载器...")
        problem_batches = check_dataloader_with_progress(train_dataloader, device, num_batches=10)
        
    except Exception as e:
        print(f"出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()