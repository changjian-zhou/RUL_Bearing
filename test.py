import pickle

from data_loader import Dataload
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import os
from model import Model


def test(dataset, test_bearing, sequence_length, batch_size, criterion):
    model_root = 'models'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_dataset = Dataload(dataset, sequence_length, test_bearing, True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    my_net = Model()
    my_net.load_state_dict(torch.load(os.path.join(model_root, 'model_epoch_current.pth')))
    my_net = my_net.eval()

    my_net = my_net.to(device, dtype=torch.float64)

    len_dataloader = len(test_loader)
    data_target_iter = iter(test_loader)

    i = 0
    total_loss = 0
    total_RMSE = 0

    while i < len_dataloader:
        # test model using target data
        data_target = data_target_iter.__next__()
        data, label = data_target
        data = data.to(device, dtype=torch.float64)
        label = label.to(device, dtype=torch.float64)

        pre_output = my_net(input_data=data)

        loss = criterion(pre_output, label)
        RMSE = loss ** 0.5

        total_loss += loss.data.cpu().item()
        total_RMSE += RMSE.data.cpu().item()

        i += 1

    return total_loss / len_dataloader, total_RMSE / len_dataloader


if __name__ == "__main__":
    with open('pkl_data/phm_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    test_bearing = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
                    'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
                    'Bearing3_3']
    # test_bearing = ['Bearing1_3']
    loss, rmse = test(dataset, test_bearing, sequence_length=5, batch_size=128, criterion=nn.MSELoss())
