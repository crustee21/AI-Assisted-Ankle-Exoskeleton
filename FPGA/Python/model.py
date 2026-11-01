import torch
import torch.nn as nn


class FootDropCNN(nn.Module):
    def __init__(self, in_ch=2, c1=16, c2=32, lstm_hidden=32,
                 k1=5, k2=3, dropout=0.3):
        super().__init__()
        assert k1 % 2 == 1 and k2 % 2 == 1, "use odd kernels so padding=k//2 preserves length"

        self.conv1 = nn.Conv1d(in_ch, c1, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(c1,    c2, kernel_size=k2, padding=k2 // 2)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.relu  = nn.ReLU()

        self.lstm  = nn.LSTM(input_size=c2, hidden_size=lstm_hidden,
                             num_layers=1, batch_first=True)

        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        x = self.drop(h_n.squeeze(0))
        x = self.fc(x)
        return x
