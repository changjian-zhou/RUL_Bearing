import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=11, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4),
        )

        self.lstm = nn.LSTM(input_size=128*4, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5)  # LSTM层


        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(),
        )

        self.rul = nn.Sequential(
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )


    def forward(self, input_data):
        batch_size, sequence_len, C, fea = input_data.size()
        input_data = torch.reshape(input_data, (batch_size * sequence_len, C, fea))
        input_data = self.cnn(input_data)
        bs, C, fea = input_data.size()
        input_data = input_data.view(-1, fea * C)
        input_data = torch.reshape(input_data, (batch_size, sequence_len, -1))

        feature, _ = self.lstm(input_data)

        feature = feature[:, -1, :]
        feature = self.classifier(feature)

        rul_output = self.rul(feature)

        return rul_output


if __name__ == "__main__":
    Model = Model()
    test_tensor = torch.randn(256, 5, 2, 1280)
    check = Model(test_tensor)
    print(check.size())
