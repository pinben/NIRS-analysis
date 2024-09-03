import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from data_processing.preprocessing import get_data 
class OxyDataset(Dataset):
    """
    自定義數據集，用於加載和處理Oxy、Deoxy和Total數據。

    此數據集設計用於處理三類數據:正常(normal)、患者(patient)和 BPD。
    從指定的根目錄加載 CSV 文件，並將數據轉換為適合模型輸入的格式。

    參數:
    - root_dir (str): 數據文件的目錄路徑。
    - num_classes (int): 類別的數量，默認為 3。

    屬性:
    - root_dir (str): 數據文件的根目錄路徑。
    - samples (list): 存儲所有樣本的列表，每個元素是 (數據, 標籤) 的tuple。
    - num_classes (int): 類別的數量。
    - max_channels (int): 所有樣本中的最大通道數。
    - targets (list): 所有樣本的標籤列表。

    方法:
    - __len__(): 返回數據集中樣本的數量。
    - __getitem__(idx): 根據索引返回一個樣本。
    """
    def __init__(self, root_dir, num_classes=3):
        self.root_dir = root_dir
        self.samples = []
        self.num_classes = num_classes
        self.max_channels = 0
        self.targets = []

        for category in ['normal', 'patient', 'BPD']:
            category_path = os.path.join(root_dir, category)
            for file in os.listdir(category_path):
                if file.endswith('_Total.csv'):
                    file_path = os.path.join(category_path, file)
                    data = get_data(file_path, data_type='Total')
                    self.max_channels = max(self.max_channels, data.shape[1])
                    label = {'normal': 0, 'patient': 1, 'BPD': 2}[category]
                    self.samples.append((data, label))
                    self.targets.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        data = torch.tensor(data.values, dtype=torch.float32)        
        return data, label

# 在主程序使用的方式
# dataset = OxyDataset(root_dir='path/to/your/data')