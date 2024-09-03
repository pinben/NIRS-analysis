import optuna
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import optuna
from optuna.samplers import TPESampler
from optuna.samplers import RandomSampler

from datasets import OxyDataset
from model import TransformerModel
from training.train import train
from visualization.evaluate_model import evaluate
from others.debug import adjust_d_model

def setup_logger(log_dir='log', log_file='optuna_optimization.log'):
    """
    設置日誌記錄器。

    參數:
    log_dir (str): 日誌文件夾路徑
    log_file (str): 日誌文件名

    返回:
    logging.Logger: 配置好的日誌記錄器
    """
    full_log_dir = os.path.join(os.getcwd(), log_dir)
    os.makedirs(full_log_dir, exist_ok=True)
    log_path = os.path.join(full_log_dir, log_file)
    
    logger = logging.getLogger('optuna_optimization')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

def objective(trial):
    """
    Optuna 優化的目標函數。

    參數:
    trial (optuna.trial.Trial): Optuna trial 對象
    seed (int): 設置隨機種子(在py檔案修改)
    返回:
    float: 模型在測試集上的準確率
    """
    # # 設置隨機種子
    # seed = 8
    # set_seed(seed)

    # 定義超參數搜索空間
    d_model = trial.suggest_int('d_model', 16, 256, step=8)  # 確保是 8 的倍數
    max_heads = min(9, d_model // 8)  # 限制最大頭數為 9 或 d_model // 8 中的較小值
    channel_nhead = trial.suggest_int('channel_nhead', 1, max_heads)
    nhead = trial.suggest_int('nhead', 1, max_heads) 
    num_layers = trial.suggest_int('num_layers', 1, 10)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.7)
    embedding_dropout = trial.suggest_float('embedding_dropout', 0.0, 0.5)
    dim_feedforward = trial.suggest_int('dim_feedforward', 128, 2048, step=128)
    # use_augmentation = trial.suggest_categorical('use_augmentation', [True, False])
    # num_epochs = trial.suggest_int('num_epochs', 50, 300)
    patience = trial.suggest_int('patience', 5, 100)
    grad_clip_value = trial.suggest_float('grad_clip_value', 0.1, 10.0)

    # 固定參數
    input_dim = 52
    num_classes = 3
    use_augmentation = False # <0.01
    # learning_rate = 5.97930254444544e-05 # 0.01
    # dim_feedforward = 896 # <0.01
    num_epochs = 1000

    # 確保 d_model 能被 nhead 跟 channel_nhead 整除
    d_model = adjust_d_model(d_model, channel_nhead, nhead)


    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載數據
    dataset = OxyDataset(root_dir=r'E:\JoFz\GTC4000\data')
    
    # 處理類別不平衡
    class_counts = np.bincount(dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[dataset.targets]

    # 分割數據
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.3, stratify=dataset.targets)
    # train_indices, test_indices = train_test_split(
    #     range(len(dataset)), test_size=0.3, stratify=dataset.targets, random_state=seed)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # 創建固定的生成器
    # g = torch.Generator()
    # g.manual_seed(seed)

    # 創建 DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                            sampler=WeightedRandomSampler(sample_weights[train_indices], len(train_indices), generator=g), generator=g)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               sampler=WeightedRandomSampler(sample_weights[train_indices], len(train_indices)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 創建模型
    # set_seed(seed) # 重現用途
    # model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes, dropout, embedding_dropout, dim_feedforward, seed).to(device)
    model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes, channel_nhead, dropout, embedding_dropout, dim_feedforward).to(device)

    # # 初始化模型權重
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)
    # model.apply(init_weights)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    T_max = num_epochs  # 設置訓練epoch數
    eta_min = 1e-6  # 最小學習率
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


    # 訓練模型
    model, train_losses, val_losses = train(model, train_loader, test_loader, criterion, optimizer, 
                                            scheduler, num_epochs, device, use_augmentation, 
                                            grad_clip_value, patience)

    # 評估模型
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # 記錄一些額外的信息，可能對後續分析有幫助
    trial.set_user_attr('train_losses', train_losses)
    trial.set_user_attr('val_losses', val_losses)
    trial.set_user_attr('final_test_accuracy', test_acc)
    
    # 記錄詳細的日誌
    logger.info(f"Trial {trial.number + 1}:")
    logger.info(f"  Params:")
    for key, param in trial.params.items():
        logger.info(f"    {key}: {param}")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    
    # Optuna 會最小化目標函數的返回值，所以我們返回錯誤率
    return test_acc

def run_optimization(n_trials=5000, seed=22):
    """
    運行 Optuna 優化過程。

    參數:
    n_trials (int): 優化試驗的次數

    返回:
    dict: 最佳試驗的參數
    """
    logger.info("Starting Optuna optimization")
    # study = optuna.create_study(direction='maximize', sampler = TPESampler(seed=seed))
    study = optuna.create_study(direction='maximize', sampler = TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Optimization finished")
    logger.info(f"Best trial:")
    logger.info(f"  Value: {study.best_trial.value:.4f}")
    logger.info(f"  Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # 將最佳結果保存到文件
    log_dir = os.path.join(os.getcwd(), 'log')
    os.makedirs(log_dir, exist_ok=True)
    best_trial_path = os.path.join(log_dir, 'best_trial.txt')
    with open(best_trial_path, 'w') as f:
        f.write(f"Best trial:\n")
        f.write(f"  Value: {study.best_trial.value}\n")
        f.write(f"  Trial number: {study.best_trial.number+1}\n")
        f.write("  Params:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"    {key} = {value}\n")

    # 生成並保存可視化
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "optimization_history.png"), dpi=1200)
    plt.show()
    # plt.close()

    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "param_importances.png"), dpi=1200)
    # plt.close()

    logger.info(f"Optimization completed. Results saved to {best_trial_path}")
    logger.info(f"Visualization saved in {results_dir}")

    return study.best_trial.params

if __name__ == "__main__":
    run_optimization()
# 這段程式的含義是：
# 如果 optuna_optimizer.py 被直接運行（例如在命令行中輸入 python optuna_optimizer.py），那麼 run_optimization() 函數會被執行。
# 如果 optuna_optimizer.py 被導入到另一個 Python 文件中（例如在主程序），這個 if 語句下的代碼就不會執行。