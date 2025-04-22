from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_fft(data: np.ndarray) -> np.ndarray:
    """在points维度执行FFT并计算功率谱密度"""
    # 输入数据形状: (time, points, channels)
    fft_data = np.fft.fft(data, axis=1)  # 沿points维度做FFT
    power_spectrum = np.abs(fft_data) ** 2 / data.shape[1]
    return power_spectrum[:, 1:1281, :]  # 移除直流分量，保留前1280点


def minmax_normalize(data: np.ndarray, axis: int = None) -> np.ndarray:
    """改进的归一化函数，支持任意维度"""
    data_min = np.min(data, axis=axis, keepdims=True)
    data_max = np.max(data, axis=axis, keepdims=True)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0  # 防止除以零
    return (data - data_min) / data_range


class RULDataset(Dataset):
    """优化后的数据集类，支持特征工程和滑动窗口生成"""

    def __init__(self,
                 dataset: Dict,
                 sequence_length: int,
                 bearing_ids: List[str],
                 use_fft: bool = True,
                 step_size: int = 1):
        """
        参数：
            dataset: 包含所有轴承数据的字典
            sequence_length: 输入序列的时间步长
            bearing_ids: 要使用的轴承ID列表
            use_fft: 是否使用FFT特征
            step_size: 滑动窗口步长
        """

        self.sequences = []
        self.labels = []

        for bid in bearing_ids:
            if bid not in dataset:
                raise KeyError(f"轴承 {bid} 不存在于数据集中")

            bearing_data = dataset[bid]['data']  # 形状 (time, 2560, 2)
            total_samples = dataset[bid]['quantity']

            # 数据预处理管道
            processed = self._preprocess(bearing_data, use_fft)

            # 生成滑动窗口
            seqs, lbls = self._generate_sequences(
                data=processed,
                seq_len=sequence_length,
                step=step_size,
                total_len=total_samples
            )

            self.sequences.append(seqs)
            self.labels.append(lbls)

        # 合并所有轴承数据
        self.sequences = np.concatenate(self.sequences, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def _preprocess(self, data: np.ndarray, use_fft: bool) -> np.ndarray:
        """数据预处理管道"""
        if use_fft:
            # 执行FFT并归一化
            data = compute_fft(data)  # 输出形状 (time, 1280, 2)
            data = minmax_normalize(data, axis=1)  # 沿points维度归一化
        else:
            # 直接归一化原始振动信号
            data = minmax_normalize(data, axis=1)
        return data

    def _generate_sequences(self,
                            data: np.ndarray,
                            seq_len: int,
                            step: int,
                            total_len: int) -> (np.ndarray, np.ndarray):
        """向量化滑动窗口生成"""
        num_sequences = (total_len - seq_len) // step + 1
        indices = np.arange(0, num_sequences * step, step)

        # 边界检查
        indices = indices[indices + seq_len <= total_len]

        # 生成序列 (优化内存访问模式)
        sequences = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=(seq_len,), axis=0
        )[indices]

        # 调整维度顺序为 (batch, seq_len, channels, points)
        sequences = np.transpose(sequences, (0, 3, 2, 1))

        # 计算归一化RUL标签
        labels = (total_len - (indices + seq_len)) / total_len

        return sequences, labels.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """返回可写入的张量数据"""
        seq = torch.as_tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import pickle
    import matplotlib.pyplot as plt

    # 参数配置
    CONFIG = {
        'batch_size': 128,
        'sequence_length': 5,
        'bearings': [
            'Bearing1_3', 'Bearing1_4', 'Bearing1_5',
            'Bearing1_6', 'Bearing1_7', 'Bearing2_3',
            'Bearing2_4', 'Bearing2_5', 'Bearing2_6',
            'Bearing2_7', 'Bearing3_3'
        ],
        'use_fft': True,
        'num_workers': 4
    }

    # 加载数据
    with open('pkl_data/phm_dataset.pkl', 'rb') as f:
        raw_dataset = pickle.load(f)

    # 创建数据集和数据加载器
    dataset = RULDataset(
        dataset=raw_dataset,
        sequence_length=CONFIG['sequence_length'],
        bearing_ids=CONFIG['bearings'],
        use_fft=CONFIG['use_fft'],
        step_size=1
    )

    loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    # 测试数据加载
    for batch in loader:
        data, labels = batch
        print(f"批次数据形状: {data.shape}")  # 应显示 (batch, seq_len, channels, points)

        # 可视化第一个样本的第一个时间步
        plt.figure(figsize=(10, 4))
        plt.plot(data[0, 0, 0, :].numpy())  # 第一个样本，第一个时间步，第一个通道，所有点
        plt.title('振动信号示例 (归一化后)')
        plt.xlabel('采样点')
        plt.ylabel('幅值')
        plt.show()
        break

    print("数据管道验证成功！")
