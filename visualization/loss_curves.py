import matplotlib.pyplot as plt
import os

def plot_loss_curves(train_losses, val_losses, save_dir='results', file_name='loss_curves.png'):
    """
    繪製訓練和驗證損失曲線並保存為圖片。

    參數:
    - train_losses : list, 訓練loss歷史
    - val_losses : list, 驗證loss歷史
    - save_dir : str, 保存圖片的目錄 (默認為 'results')
    - file_name : str, 保存的文件名 (默認為 'loss_curves.png')

    返回:
    None
    """
    # 使用相對路徑
    full_save_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(full_save_dir, exist_ok=True)
    
    save_path = os.path.join(full_save_dir, file_name)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_path, dpi=1200)
    plt.close()