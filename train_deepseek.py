import pickle
from pathlib import Path
from datetime import datetime
import random
from typing import Dict, List, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm

from data_loader import RULDataset
from model import Model  # 假设使用之前优化的模型


class RULTrainer:
    """轴承剩余寿命预测训练器"""

    def __init__(self, config: Dict):
        # 初始化配置
        self._validate_config(config)
        self.config = config
        self.device = torch.device(config["device"])

        # 设置确定性训练
        self._set_seed(config["seed"])

        # 初始化路径
        self.model_dir = Path(config["model_dir"])
        self.log_dir = Path(config["log_dir"])
        self._prepare_directories()

        # 加载数据
        self.dataset = self._load_dataset(config["data_path"])

        # 初始化模型组件
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = self._init_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])

        # 训练状态
        self.best_metric = float("inf")
        self.current_epoch = 0

    def _validate_config(self, config: Dict) -> None:
        """验证配置参数有效性"""
        required_keys = {"data_path", "train_bearings", "test_bearings",
                         "sequence_length", "batch_size", "lr", "epochs"}
        missing = required_keys - set(config.keys())
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _set_seed(self, seed: int) -> None:
        """设置随机种子保证可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 保证每次结果都一样
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _prepare_directories(self) -> None:
        """创建必要的存储目录"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self, path: str) -> Dict:
        """加载预处理数据集"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _init_model(self) -> nn.Module:
        """初始化模型并移至设备"""
        model = Model().to(self.device)
        if self.config.get("pretrained"):
            model.load_state_dict(torch.load(self.config["pretrained"]))
        return model

    def _init_optimizer(self) -> optim.Optimizer:
        """初始化优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 1e-4)
        )

    def _init_scheduler(self):
        """初始化学习率调度器"""
        return optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.99
        )

    def _init_criterion(self) -> nn.Module:
        """初始化损失函数"""
        return nn.HuberLoss(delta=0.5).to(self.device)

    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建训练和验证数据加载器"""
        train_set = RULDataset(
            self.dataset,
            self.config["sequence_length"],
            self.config["train_bearings"],
            use_fft=True
        )

        val_set = RULDataset(
            self.dataset,
            self.config["sequence_length"],
            self.config["test_bearings"],  # 假设用部分测试轴承做验证
            use_fft=True
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True
        )

        return train_loader, val_loader

    def train(self) -> None:
        """执行完整训练流程"""
        train_loader, val_loader = self._create_data_loaders()
        writer = SummaryWriter(self.log_dir / "tensorboard")

        for epoch in range(self.current_epoch, self.config["epochs"]):
            self.current_epoch = epoch

            # 训练阶段
            train_metrics = self._train_epoch(train_loader)

            # 验证阶段
            val_metrics = self._validate(val_loader)

            # 更新学习率
            self.scheduler.step()

            # 记录日志
            self._log_progress(epoch, train_metrics, val_metrics, writer)

            # 保存检查点
            self._save_checkpoint(epoch, val_metrics)

    def _train_epoch(self, loader: DataLoader) -> Dict:
        """单epoch训练"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(loader, desc=f"Training Epoch {self.current_epoch}")

        for batch in progress_bar:
            inputs, targets = self._prepare_batch(batch)

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.config["use_amp"]):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return {"loss": total_loss / len(loader)}

    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备训练批次数据"""
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return inputs, targets

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict:
        """验证步骤"""
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0

        for batch in tqdm(loader, desc="Validating"):
            inputs, targets = self._prepare_batch(batch)
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            rmse = torch.sqrt(nn.functional.mse_loss(outputs, targets))

            total_loss += loss.item()
            total_rmse += rmse.item()

        metrics = {
            "loss": total_loss / len(loader),
            "rmse": total_rmse / len(loader)
        }
        return metrics

    def _log_progress(self, epoch: int, train_metrics: Dict,
                      val_metrics: Dict, writer: SummaryWriter) -> None:
        """记录训练进度"""
        # TensorBoard记录
        writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"]
        }, epoch)

        writer.add_scalar("RMSE/val", val_metrics["rmse"], epoch)
        writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

        # CSV日志记录
        log_df = pd.DataFrame({
            "epoch": [epoch],
            "train_loss": [train_metrics["loss"]],
            "val_loss": [val_metrics["loss"]],
            "val_rmse": [val_metrics["rmse"]],
            "lr": [self.optimizer.param_groups[0]["lr"]]
        })

        log_file = self.log_dir / "training_log.csv"
        if log_file.exists():
            log_df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)

    def _save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        """保存模型检查点"""
        # 保存最新检查点
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config
        }
        torch.save(checkpoint, self.model_dir / "latest_checkpoint.pth")

        # 保存最佳模型
        if metrics["rmse"] < self.best_metric:
            self.best_metric = metrics["rmse"]
            torch.save(self.model.state_dict(), self.model_dir / "best_model.pth")


if __name__ == "__main__":
    # 配置参数
    config = {
        "data_path": "pkl_data/phm_dataset.pkl",
        "train_bearings": ['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2'],
        "test_bearings": ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
                          'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
                          'Bearing3_3'],
        "sequence_length": 5,
        "batch_size": 128,
        "lr": 1e-4,
        "epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_dir": "./models",
        "log_dir": "./logs",
        "use_amp": True,  # 启用混合精度训练
        "seed": 42
    }

    # 初始化并运行训练
    trainer = RULTrainer(config)
    trainer.train()