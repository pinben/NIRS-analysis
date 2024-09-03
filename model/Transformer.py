import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class PositionalEncoding(nn.Module):
    """
    Positional Encoding layer，用於為序列數據添加位置信息

    參數:
    - d_model (int): 模型的維度
    - dropout (float): Dropout 率，默認為 0.1
    - max_len (int): 最大序列長度，默認為 500

    屬性:
    - dropout (nn.Dropout): Dropout 層
    - pe (Tensor): 預計算的位置編碼

    方法:
    forward(x): 將位置編碼添加到輸入張量
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # 初始化位置編碼矩陣(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))# 長度為 d_model//2
        # -math.log(10000.0) / d_model 和取 exp 的方式，源自於 Transformer 模型中位置編碼（Positional Encoding）的設計原理
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # 確保不會超出範圍
    
        pe = pe.unsqueeze(0).transpose(0, 1) # pe.unsqueeze(0) -> (1, max_len, d_model);.transpose(0, 1) -> (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ChannelMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ChannelMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: (batch_size, time_steps, channels)             
        attn_output, _ = self.multihead_attn(x, x, x)  # 應用多頭注意力
        return attn_output

class TransformerModel(nn.Module):
    """
    基於 Transformer 架構的神經網絡模型。

    參數:
    - input_dim (int): 輸入特徵的維度
    - d_model (int): 模型的維度
    - nhead (int): multi-head 注意力中的頭數
    - num_layers (int): Transformer 編碼器層的數量
    - num_classes (int): 輸出類別的數量
    - dropout (float): Dropout 率，默認為 0.3
    - embedding_dropout (float): 嵌入層的 Dropout 率，默認為 0.1
    - dim_feedforward (int): 前饋網絡的隱藏層維度，默認為 256

    模型架構:
    - embedding (nn.Linear): 輸入特徵的線性嵌入層
    - pos_encoder (PositionalEncoding): 位置編碼層
    - transformer_encoder (nn.TransformerEncoder): Transformer 編碼器
    - fc1 (nn.Linear): 第一個全連接層
    - fc2 (nn.Linear): 第二個全連接層（輸出層）
    - dropout (nn.Dropout): Dropout 層

    方法:
    forward(x): 定義模型的前向傳播

    參數:
    - x (Tensor): 輸入張量，形狀為 (batch, channels, time_steps)

    返回:
    - Tensor: 模型的輸出，形狀為 (batch, num_classes)
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, channel_nhead=3, dropout=0.3, embedding_dropout=0.1, dim_feedforward=256, seed=22):
        super(TransformerModel, self).__init__()
        # torch.manual_seed(seed) # 重現用途
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, embedding_dropout)
        self.channel_attn = ChannelMultiheadAttention(d_model, channel_nhead) # 新增通道間多頭注意力機制
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        # self._initialize_weights() # 初始化權重(重現用途)
    # 初始化權重(重現用途)
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.01)
    #         elif isinstance(m, nn.TransformerEncoderLayer):
    #             for sub_m in m.modules():
    #                 if isinstance(sub_m, nn.Linear):
    #                     nn.init.xavier_uniform_(sub_m.weight)
    #                     if sub_m.bias is not None:
    #                         nn.init.constant_(sub_m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, time_steps, channels)
        x = self.embedding(x)
        x = self.channel_attn(x) # 使用通道注意力機制
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)  # 將形狀改為 (batch, channels, time_steps) 用於池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1) # 自適應池化到固定大小
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# forward 順序理由：
# 嵌入（Embedding）：先進行嵌入操作，將原始特徵轉換為更有意義的表示。這為後續的注意力機制提供了更好的輸入。
# 通道注意力（Channel Attention）：應用通道間的注意力機制。允許模型在進行時間序列處理之前，先捕捉不同通道之間的相互關係。這可能有助於模型更好地理解各個通道之間的相互影響。
# 位置編碼（Positional Encoding）：在處理時間序列之前添加位置信息。這對於後續的 transformer 編碼器來說是必要的，因為它需要知道序列中每個元素的相對位置。
# Transformer 編碼器：最後應用 transformer 編碼器，這將處理每個通道內的時間序列信息。此時，模型已經具備了通道間的關係信息和位置信息。

# 順序的優點：
# 1.先處理了跨通道的信息，這可能有助於後續的時間序列處理。
# 2.在序列處理之前添加位置編碼。(Transformer的核心思想)
# 3.允許模型在不同的抽象層次上逐步構建其表示：從原始特徵到通道關係，再到時間序列動態。