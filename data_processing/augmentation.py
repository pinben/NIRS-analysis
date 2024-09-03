import torch
import numpy as np
def augment_data(x, augmentation_types=['noise', 'time_warp', 'scaling', 'magnitude_warp', 'window_warp', 'time_shift']):
    """
    對輸入數據進行數據增強。

    參數:
    - x (torch.Tensor): 輸入數據張量，形狀為 (batch_size, channels, time_steps)
    - augmentation_types (list of str): 要應用的增強方法列表，默認包括所有可用方法

    返回:
    torch.Tensor: 經過增強的數據張量，形狀與輸入相同

    可用的增強方法:
    - 'noise' : 添加高斯噪聲
    - 'time_warp' : 時間扭曲，對時間維度進行非線性變換
    - 'scaling' : 對整個信號進行縮放
    - 'magnitude_warp' : 對信號幅度進行扭曲
    - 'window_warp' : 對信號的局部窗口進行扭曲
    - 'time_shift' : 對信號進行時間平移

    說明:
    此函數對輸入數據應用多種數據增強技術。每種增強方法都有特定的參數和實現方式：
    - 噪聲增強：添加標準差為 0.05 的高斯噪聲
    - 時間扭曲：累積和標準差為 0.1 的隨機值
    - 縮放：使用平均值為 1，標準差為 0.1 的因子進行縮放
    - 幅度扭曲：使用平均值為 1，標準差為 0.1 的因子對幅度進行扭曲
    - 窗口扭曲：隨機選擇 10% 的時間窗口進行線性插值
    - 時間平移：在 ±10% 的範圍內隨機平移信號

    注意：
    - 所有操作都在 GPU 上進行（如果輸入張量在 GPU 上）
    - 增強操作是累積的，按照 augmentation_types 中指定的順序應用
    """
    device = x.device
    augmented = x.clone()
    
    for aug_type in augmentation_types:
        if aug_type == 'noise':
            noise = torch.randn_like(x, device=device) * 0.05
            augmented += noise
        
        elif aug_type == 'time_warp':
            time_warp = torch.cumsum(torch.randn_like(x, device=device) * 0.1, dim=2)
            augmented += time_warp
        
        elif aug_type == 'scaling':
            scaling_factor = (torch.randn(x.size(0), 1, 1, device=device) * 0.1 + 1)
            augmented *= scaling_factor
        
        elif aug_type == 'magnitude_warp':
            magnitude_warp = torch.from_numpy(np.random.normal(loc=1.0, scale=0.1, size=x.shape)).float().to(device)
            augmented *= magnitude_warp
        
        elif aug_type == 'window_warp':
            window_size = int(x.size(2) * 0.1)
            for i in range(x.size(0)):
                start = np.random.randint(0, x.size(2) - window_size)
                end = start + window_size
                augmented[i, :, start:end] = torch.nn.functional.interpolate(
                    augmented[i, :, start:end].unsqueeze(0),
                    size=window_size,
                    mode='linear',
                    align_corners=True
                ).squeeze(0)
        
        elif aug_type == 'time_shift':
            shift = np.random.randint(-x.size(2)//10, x.size(2)//10)
            augmented = torch.roll(augmented, shifts=shift, dims=2)
    
    return augmented