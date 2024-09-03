import torch
import numpy as np
import random
import os

def set_seed(seed = 55):
    """
    設置全局隨機種子，以確保實驗的可重現性。
    這個函數覆蓋了數據處理、模型初始化、訓練過程等各個方面。

    參數:
    seed (int): 要使用的隨機種子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def seed_worker(worker_id = 42):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)