import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def plot_confusion_matrix(model, test_loader, device, save_dir='results', file_name='confusion_matrix.png'):
    """
    計算混淆矩陣並將其繪製和保存為圖片。

    參數:
    - model : 訓練好的模型
    - test_loader : 測試數據的 DataLoader
    - device : 使用的設備 (CPU 或 GPU)
    - save_dir : str, 保存圖片的目錄 (默認為 'results')
    - file_name : str, 保存的文件名 (默認為 'confusion_matrix.png')

    返回:
    None
    """
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    full_save_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(full_save_dir, exist_ok=True)
    
    save_path = os.path.join(full_save_dir, file_name)

    plt.figure(figsize=(12, 12))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=1200)
    plt.close()
