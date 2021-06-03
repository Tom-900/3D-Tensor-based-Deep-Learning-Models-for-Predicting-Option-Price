import pandas as pd
import numpy as np
import torch
from tqdm import trange
from core.functions import *
import matplotlib.pyplot as plt
from functions import *

test_data_BS = pd.read_csv(r"\data\test_data_BS.csv")
data_list = []
seq_len = 10

for i in trange(test_data_BS['optID'].unique().shape[0]):
    for index in range(test_data_BS.loc[(test_data_BS['optID'] == test_data_BS['optID'].unique()[i])].shape[0] - seq_len+1):
        tmp_data = np.array(test_data_BS.loc[(test_data_BS['optID'] == test_data_BS['optID'].unique()[i])][index:index+seq_len])
        data = torch.tensor(np.array(tmp_data[seq_len-1, :], dtype=np.float64())).unsqueeze(0)
        data_list.append(data)

test_data_BS = torch.cat(data_list, dim=0)
print("Shape of the Test Data BS:", test_data_BS.shape)


BS_price = torch.zeros(size=(test_data_BS.shape[0], 1))
for i in trange(test_data_BS.shape[0]):
    maturity = test_data_BS[i, 3]
    vol = test_data_BS[i, 4]
    strike = test_data_BS[i, 5]
    spot = test_data_BS[i, 6]
    r = test_data_BS[i, 7]
    price = 0
    if test_data_BS[i, 2] == 0:
        price = call_option_pricer(spot, strike, maturity, r, vol)
    if test_data_BS[i, 2] == 1:
        price = put_option_pricer(spot, strike, maturity, r, vol)
    BS_price[i, :] = price

BS_price = BS_price.squeeze()
test_price = test_data_BS[:, 8]

corr, map, mape = metrics(BS_price, test_price)
corr = torch.mean(corr).float()
map = torch.mean(map).float()
mape = torch.mean(mape).float()
mse = torch.mean(torch.square(test_price - BS_price)).float()
rmse = torch.sqrt(mse).float()

print("MSE: {:.5f}\nRMSE: {:0.5f}\nMAP: {:0.5f}\nMAPE: {:0.5f}\nCorrelation: {:0.5f}\n".format(mse, rmse, map, mape, corr))

begin_plot = 0
plot_len = 120
plt.plot(np.arange(plot_len), test_price[begin_plot:begin_plot + plot_len].numpy(), color='blue', label="True Price")
plt.plot(np.arange(plot_len), BS_price[begin_plot:begin_plot + plot_len].numpy(), color='red', label="Estimated Price")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Option Price')
plt.title("Comparison of Estimated Price and True Price")

plt.show()
