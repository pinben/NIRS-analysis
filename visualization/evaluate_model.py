import torch
import logging

def evaluate(model, dataloader, criterion, device):
    """
    評估模型在給定數據集上的性能。

    參數:
    - model (torch.nn.Module): 要評估的PyTorch模型。
    - dataloader (torch.utils.data.DataLoader): 包含評估數據的DataLoader。
    - criterion (torch.nn.Module): 用於計算loss的損失函數。
    - device (torch.device): 用於計算的設備(CPU或GPU)。

    返回:
    tuple: 包含兩個浮點數的元組：
    - avg_loss (float): 在評估數據集上的平均損失。
    - accuracy (float): 在評估數據集上的準確率。

    說明:
    此函數將模型設置為評估模式(modal.eval())，然後在不計算梯度的情況下遍歷數據集。
    然後計算總loss、正確預測的數量，並返回平均損失和準確率。
    評估結果會通過logging模塊記錄。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    logging.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy