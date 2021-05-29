import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d

class LSTM(nn.Module):
    def __init__(self, in_dim=5, in_channels=3, out_dim=1, seq_len=10):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels * in_dim,
                          hidden_size=8,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)
        self.out = nn.Linear(in_features=16*seq_len, out_features=out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # shape of the input:(N, C, T, D_in)
    # shape of the label:(N, 1)
    def forward(self, x):
        # x:(N, C, T, D)
        x = x.transpose(1, 2)
        # x:(N, T, C, D_in)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        # x:(N, T, C*D_in)
        x, _ = self.lstm(x)
        # x:(N, T, 16)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = self.out(x).squeeze()
        # x:(N, 1)
        return x

# test
if __name__ == '__main__':
    # (N, C, T, D_in)
    data_input = torch.normal(0, 1, size=(16, 3, 10, 5))
    model = LSTM()
    data_output = model(data_input)
    print(data_output.shape)