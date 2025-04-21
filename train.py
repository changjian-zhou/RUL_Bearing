import pickle
from pathlib import Path
from data_loader import Dataload
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import os
from model import Model
from test import test
import pandas as pd
from datetime import datetime


# 设置随机种子
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果都一样
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class RUL():
    def __init__(self):
        with open('pkl_data/phm_dataset.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
        self.train_bearing = ['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2']
        self.test_bearing = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
                             'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
                             'Bearing3_3']
        self.sequence_length = 5
        self.batch_size = 128
        self.lr = 1e-3
        self.epoch = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = Path(r"./models")
        model_path.mkdir(parents=True, exist_ok=True)
        self.net_RUL_model_save_path = './models/model_epoch_current.pth'
        self.net_bestRUL_model_save_path = './models/model_best.pth'

    def train(self):
        train_dataset = Dataload(self.dataset, self.sequence_length, self.train_bearing, True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        net = Model().to(self.device, dtype=torch.float64)

        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        criterion_MSE = nn.MSELoss().cuda()

        # 定义要记录的指标名称
        metric_names = ['err_s_label', 'loss_s', 'rmse_s', 'loss_t', 'rmse_t', 'lr']

        # 初始化记录字典
        log = {key: [] for key in metric_names}

        # 获取当前时间的时间戳字符串
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M")

        log_path = Path(r"./log")
        log_path.mkdir(parents=True, exist_ok=True)

        # 获取log文件夹下已有的log文件数量
        existing_logs = [filename for filename in os.listdir('./log/') if filename.startswith('log')]
        run_number = len(existing_logs) + 1  # 新的log文件的序号

        # 构建新的log文件名
        log_filename = f"./log/log{run_number}_{timestamp}.csv"

        best_rmse_t = 9999

        for epoch in range(1, self.epoch + 1):
            net.train()

            total_err_s_label = 0

            len_dataloader = len(train_loader)

            data_source_iter = iter(train_loader)

            for i in range(len_dataloader):
                # training model using source data
                data_source = data_source_iter.__next__()
                s_data, s_label = data_source

                net.zero_grad()  # 首先梯度清零

                s_data = s_data.to(self.device, dtype=torch.float64)
                s_label = s_label.to(self.device, dtype=torch.float64)

                pre_output = net(input_data=s_data)
                err_s_label = criterion_MSE(pre_output, s_label)

                err = err_s_label
                total_err_s_label += err_s_label.data
                err.backward()
                optimizer.step()

            torch.save(net.state_dict(), self.net_RUL_model_save_path)
            loss_s, rmse_s = test(dataset=self.dataset,
                                  test_bearing=self.train_bearing,
                                  sequence_length=self.sequence_length, batch_size=self.batch_size,
                                  criterion=criterion_MSE)
            loss_t, rmse_t = test(dataset=self.dataset,
                                  test_bearing=self.test_bearing,
                                  sequence_length=self.sequence_length, batch_size=self.batch_size,
                                  criterion=criterion_MSE)

            print(
                "[Epoch:%d][err_s_label:%.4e][loss_s:%.4e][rmse_s:%.4f][loss_t:%.4e][rmse_t:%.4f]"
                % (epoch, total_err_s_label / len_dataloader, loss_s, rmse_s, loss_t, rmse_t)
            )

            for metric_name, metric_value in zip(metric_names, [total_err_s_label / len_dataloader,
                                                                loss_s, rmse_s, loss_t, rmse_t,
                                                                optimizer.param_groups[0]['lr']]):
                log[metric_name].append(float(metric_value))

            # 保存记录到CSV文件
            pd.DataFrame(log).to_csv(log_filename, index=False)

            if rmse_t < best_rmse_t:
                best_epoch = epoch
                best_rmse_t = rmse_t
                torch.save(net.state_dict(), self.net_bestRUL_model_save_path)
        print('Best Epoch {} Best rmse_t {:.4f}'.format(best_epoch, best_rmse_t))
        return best_rmse_t


if __name__ == '__main__':
    set_random_seed(seed=42)
    process = RUL()
    process.train()
