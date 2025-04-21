import numpy as np
import torch
from torch.utils.data import Dataset


def _fft(data):
    fft_data = np.fft.fft(data, axis=1)
    fft_data = (np.abs(fft_data)) ** 2 / data.shape[1]
    fft_data = fft_data[:, 1:1281]
    return fft_data


def _normalize(data, dim=None):
    mmrange = np.max(data, axis=dim, keepdims=True) - np.min(data, axis=dim, keepdims=True)
    r_data = (data - np.min(data, axis=dim, keepdims=True).repeat(data.shape[dim], axis=dim)) / mmrange.repeat(
        data.shape[dim], axis=dim)
    return r_data


class Dataload(Dataset):
    def __init__(self, dataset, num_sequence, bearings, fft):

        alldata = [dataset[bearing]['data'] for bearing in bearings]
        quantity = [dataset[bearing]['quantity'] for bearing in bearings]

        self.label = []
        self.data = []  # 创建一个列表来保存数据序列

        for i, x in enumerate(alldata):
            data_len = quantity[i]
            for index in range(0, data_len - num_sequence + 1, num_sequence):
                sequence_data = []  # 创建一个列表来保存单个序列的数据
                for seq in range(num_sequence):
                    data_one = x[index + seq, :, :].transpose()
                    if fft:
                        data_one = _fft(data_one)
                        data_one = _normalize(data_one, dim=1)
                    sequence_data.append(data_one)
                self.label.append((data_len - index - num_sequence) / data_len)
                self.data.append(np.asarray(sequence_data))  # 添加序列数据到列表

        self.label = np.asarray(self.label).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = torch.from_numpy(self.data[i]).float()
        return data, self.label[i]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import pickle

    with open('pkl_data/phm_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    batch_size   = 128
    sequence_length = 5

    bearings = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
                'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
                'Bearing3_3']

    train_dataset = Dataload(dataset, sequence_length, bearings, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    print(train_loader)
    for data, label in train_loader:
        # print(data, label)
        # plt.ion()
        plt.figure()
        plt.plot(data[0, 0, 0, :])
        plt.show()
        # plt.ioff()

    print('ok')
