import pickle

from data_loader import RULDataset
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import os
from model import Model


def test(dataset, test_bearing, sequence_length, batch_size, criterion, plot=False):
    model_root = 'models'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_dataset = RULDataset(dataset, sequence_length, test_bearing, True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    my_net = Model()
    my_net.load_state_dict(torch.load(os.path.join(model_root, 'best_model.pth')))
    my_net = my_net.eval()

    my_net = my_net.to(device, dtype=torch.float64)

    len_dataloader = len(test_loader)
    data_target_iter = iter(test_loader)

    i = 0
    total_loss = 0
    total_RMSE = 0
    if plot:
        all_preds = []
        all_labels = []

    while i < len_dataloader:
        # test model using target data
        data_target = data_target_iter.__next__()
        data, label = data_target
        data = data.to(device, dtype=torch.float64)
        label = label.to(device, dtype=torch.float64)

        pre_output = my_net(data)

        loss = criterion(pre_output, label)
        RMSE = loss ** 0.5

        total_loss += loss.data.cpu().item()
        total_RMSE += RMSE.data.cpu().item()

        if plot:
            all_preds.append(pre_output.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

        i += 1

    if plot:
        # 拼接并绘图
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        plt.figure(figsize=(10, 5))
        plt.plot(all_labels, label='True RUL', linewidth=2)
        plt.plot(all_preds, label='Predicted RUL', linewidth=2)
        plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel('RUL')
        plt.title('True vs Predicted RUL')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return total_loss / len_dataloader, total_RMSE / len_dataloader


if __name__ == "__main__":
    with open('pkl_data/phm_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    test_bearing = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
                    'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
                    'Bearing3_3']
    # test_bearing = ['Bearing1_6']
    loss, rmse = test(dataset, test_bearing, sequence_length=5, batch_size=128, criterion=nn.MSELoss(), plot=True)
