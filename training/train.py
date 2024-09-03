import torch
from data_processing.augmentation import augment_data
from visualization.evaluate_model import evaluate
import os

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, use_augmentation=True, grad_clip_value=1.0, patience=40):
    """
    訓練模型的主函數。

    參數:
    - model: 要訓練的模型
    - train_loader: 訓練數據加載器
    - val_loader: 驗證數據加載器
    - criterion: loss function
    - optimizer: 優化器
    - scheduler: 學習率調整器
    - num_epochs: 訓練模型的次數
    - device: 使用的設備 (CPU/GPU)
    - use_augmentation: 是否使用數據增強
    - grad_clip_value: 梯度裁剪值
    - patience: Early Stopping的耐心

    備註:
    - 要確保 d_model 能被 nhead 整除

    返回:
    - model: 訓練後的模型
    - train_losses: 訓練loss歷史
    - val_losses: 驗證loss歷史
    """
    full_save_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(full_save_dir, exist_ok=True)
    save_path = os.path.join(full_save_dir, 'best_model.pth')

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if use_augmentation:
                augmented_inputs = augment_data(inputs)
                combined_inputs = torch.cat([inputs, augmented_inputs], dim=0)
                combined_labels = torch.cat([labels, labels], dim=0)
            else:
                combined_inputs = inputs
                combined_labels = labels

            optimizer.zero_grad()
            outputs = model(combined_inputs)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load(save_path))
    print("Training completed")
    
    return model, train_losses, val_losses