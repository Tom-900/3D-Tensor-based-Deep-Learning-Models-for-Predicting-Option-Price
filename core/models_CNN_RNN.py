import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d

class model(nn.Module):
    def __init__(self, resolution_ratio=4, nonlinearity="tanh", in_dim=5, in_channels=3, out_dim=1, seq_len=10):
        super(model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                           out_channels=8,
                                           kernel_size=(3, 1),
                                           padding=(1, 0),
                                           dilation=(1, 1)),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                           out_channels=8,
                                           kernel_size=(3, 1),
                                           padding=(2, 0),
                                           dilation=(2, 1)),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                           out_channels=8,
                                           kernel_size=(3, 1),
                                           padding=(3, 0),
                                           dilation=(3, 1)),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=24,
                                           out_channels=16,
                                           kernel_size=(3, 1),
                                           padding=(1, 0)),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=(3, 1),
                                           padding=(1, 0)),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=(1, 1)),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation)

        self.linear = nn.Linear(in_features=in_dim, out_features=1)

        self.gru = nn.GRU(input_size=in_channels * in_dim,
                          hidden_size=8,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)

        self.gru_out = nn.GRU(input_size=16,
                              hidden_size=8,
                              num_layers=1,
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
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1))
        # cnn_out:(N, 16, T, D_in)
        cnn_out = self.linear(cnn_out).squeeze()
        cnn_out = cnn_out.transpose(-1, -2)
        # cnn_out:(N, T, 16)

        x = x.transpose(1, 2)
        # x:(N, T, C, D_in)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        # x:(N, T, C*D_in)
        rnn_out, _ = self.gru(x)
        # rnn_out:(N, T, 16)

        x = rnn_out + cnn_out

        x, _ = self.gru_out(x)
        # x: (N, T, 16)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = self.out(x).squeeze()
        # x:(N, 1)
        return x

# test
if __name__ == '__main__':
    # (N, C, T, D_in)
    data_input = torch.normal(0, 1, size=(16, 3, 10, 5))
    model = model()
    data_output = model(data_input)
    print(data_output.shape)