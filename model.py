## model.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, bidirectional=True, drop=0.5):
        super().__init__()
        self.name = 'LSTMRegressor'
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0  # LSTM层间dropout，仅当层数>1时有效
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc_xyz = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop),  # 全连接层后添加dropout
            nn.Linear(64, 3)
        )
        self.fc_time = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop),  # 全连接层后添加dropout
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        # x: (batch, max_len, 63)
        # lengths: (batch,)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)

        # h: (num_layers * num_directions, batch, hidden_dim)
        feat = torch.cat([h[-2], h[-1]], dim=1)  # (batch, 128*2)

        pred_xyz = self.fc_xyz(feat)         # (batch, 3)
        pred_time = self.fc_time(feat).squeeze(1)  # (batch,)

        return pred_xyz, pred_time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Conv1DResidual(nn.Module):
    """1D卷积残差块，用于提取姿态内部特征"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接适配层
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        # x: (batch, seq_len, num_points, 3) -> 经过permute后为(batch, seq_len, 3, num_points)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class AttentionLayer(nn.Module):
    """注意力层，用于突出重要帧"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, x, lengths):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        
        # 计算注意力权重
        attn_weights = self.attention(x).squeeze(2)  # (batch, seq_len)
        
        # 对填充部分进行掩码
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        attn_weights.masked_fill_(mask, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # (batch, 1, seq_len)
        
        # 应用注意力
        weighted_sum = torch.bmm(attn_weights, x).squeeze(1)  # (batch, hidden_dim)
        return weighted_sum, attn_weights

class ImprovedLSTMRegressor(nn.Module):
    def __init__(self, num_points=21, input_dim=3, conv_dims=[32, 64], 
                 hidden_dim=64, num_layers=2, bidirectional=True, drop=0.5):
        super().__init__()
        self.name = 'ImprovedLSTMRegressor'
        self.num_points = num_points
        self.bidirectional = bidirectional
        
        # 1D卷积特征提取部分
        conv_layers = []
        in_channels = input_dim  # 每个点有3个坐标(x,y,z)
        for out_channels in conv_dims:
            conv_layers.append(Conv1DResidual(in_channels, out_channels))
            in_channels = out_channels
        
        self.conv_extractor = nn.Sequential(*conv_layers)
        self.conv_out_dim = conv_dims[-1]
        self.point_feature_dim = self.conv_out_dim * num_points  # 每个姿态的总特征维度
        
        # LSTM时序特征提取
        self.lstm = nn.LSTM(
            input_size=self.point_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0
        )
        
        # 注意力层
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = AttentionLayer(lstm_out_dim)
        
        # 输出预测头
        self.fc_xyz = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.fc_time = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        # x: (batch, max_len, num_points*3) -> 需要先转换为(batch, max_len, num_points, 3)
        batch_size, max_len, _ = x.size()
        
        # 重塑输入以分离每个点的坐标
        x = x.view(batch_size, max_len, self.num_points, 3)  # (batch, max_len, num_points, 3)
        
        # 调整维度用于1D卷积: (batch, max_len, 3, num_points)
        x = x.permute(0, 1, 3, 2)
        
        # 提取每个姿态的空间特征
        conv_feat = []
        for t in range(max_len):
            # 对每个时间步的姿态应用卷积提取器
            frame_feat = self.conv_extractor(x[:, t, :, :])  # (batch, conv_out_dim, num_points)
            frame_feat = frame_feat.flatten(1)  # (batch, conv_out_dim * num_points)
            conv_feat.append(frame_feat)
        
        # 堆叠成序列: (batch, max_len, point_feature_dim)
        conv_seq = torch.stack(conv_feat, dim=1)
        
        # LSTM处理时序特征
        packed = pack_padded_sequence(conv_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # (batch, max_len, lstm_out_dim)
        
        # 应用注意力机制
        attn_feat, attn_weights = self.attention(lstm_out, lengths)  # (batch, lstm_out_dim)
        
        # 预测输出
        pred_xyz = self.fc_xyz(attn_feat)         # (batch, 3)
        pred_time = self.fc_time(attn_feat).squeeze(1)  # (batch,)
        
        return pred_xyz, pred_time

class SimplifiedLSTMRegressor(nn.Module):
    def __init__(self, num_points=21, input_dim=3, conv_dims=[32, 64],  # 减少卷积维度
                 hidden_dim=64, num_layers=1, bidirectional=False,  # 简化LSTM
                 drop=0.3):  # 降低dropout率
        super().__init__()
        self.name = 'SimplifiedLSTMRegressor'
        self.num_points = num_points
        self.bidirectional = bidirectional
        
        # 简化卷积特征提取：减少层数和维度，移除残差连接
        conv_layers = []
        in_channels = input_dim
        for out_channels in conv_dims:
            # 使用更大的步长减少计算量，移除 BatchNorm
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            # conv_layers.append(nn.BatchNorm1d(out_channels))  # 新增 BatchNorm
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.conv_extractor = nn.Sequential(*conv_layers)
        self.conv_out_dim = conv_dims[-1]
        self.point_feature_dim = self.conv_out_dim * num_points
        
        # 简化LSTM：减少层数，改为单向
        self.lstm = nn.LSTM(
            input_size=self.point_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0
        )
        
        # 移除注意力机制，直接使用LSTM最后一个时间步的输出
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        
        # 简化全连接层：减少层数和维度
        self.fc_xyz = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 3)
        )
        
        self.fc_time = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        # x: (batch, max_len, num_points*3)
        batch_size, max_len, _ = x.size()
        
        # 重塑输入
        x = x.view(batch_size, max_len, self.num_points, 3)  # (batch, max_len, num_points, 3)
        x = x.permute(0, 1, 3, 2)  # (batch, max_len, 3, num_points)
        
        # 卷积特征提取：批量处理所有时间步（替代循环）
        # 调整维度以便并行处理所有帧：(batch*max_len, 3, num_points)
        x_reshaped = x.reshape(-1, 3, self.num_points)
        conv_feat = self.conv_extractor(x_reshaped)  # (batch*max_len, conv_out_dim, num_points)
        conv_feat = conv_feat.flatten(1)  # (batch*max_len, conv_out_dim*num_points)
        
        # 恢复时序结构：(batch, max_len, point_feature_dim)
        conv_seq = conv_feat.view(batch_size, max_len, -1)
        
        # LSTM处理
        packed = pack_padded_sequence(conv_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # 只保留最后一个时间步的隐状态
        
        # 取最后一层的输出
        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # 双向时拼接正反方向
        else:
            feat = h_n[-1]  # 单向时直接取最后一层
        
        # 预测输出
        pred_xyz = self.fc_xyz(feat)
        pred_time = self.fc_time(feat).squeeze(1)
        
        return pred_xyz, pred_time
    
if __name__ == "__main__":
    from dataset import load_all_samples, BadmintonDataset, collate_fn_dynamic

    def test_model(data_folder='/home/zhaoxuhao/badminton_xh/20250809_Seq_data', batch_size=8):
        """
        测试模型 forward 是否正常
        """
        # 1. 加载数据
        samples = load_all_samples(data_folder)
        print(f"Loaded {len(samples)} samples.")

        dataset = BadmintonDataset(samples, min_len=40, max_len=50)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_dynamic)

        # 2. 构建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMRegressor(input_dim=63, hidden_dim=128, num_layers=2, bidirectional=True)
        model.to(device)
        model.eval()

        # 3. 测试一个 batch
        with torch.no_grad():
            for batch in dataloader:
                seqs, lengths, masks, xyzs, times = batch
                seqs = seqs.to(device)
                lengths = lengths.to(device)

                pred_xyz, pred_time = model(seqs, lengths)
                print("pred_xyz:", pred_xyz.shape, pred_xyz)   # (B, 3)
                print("pred_time:", pred_time.shape, pred_time) # (B,)
                break

    test_model()