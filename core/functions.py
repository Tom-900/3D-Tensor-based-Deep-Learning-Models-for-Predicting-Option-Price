import numpy as np
import torch
from torch import log, sqrt, exp
from scipy.stats import norm


#%% Normalization
class Normalization:
    def __init__(self, mean_val=None, std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        return (x-self.mean_val)/self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val


#%% Metrics
def metrics(y, x):
    #x: reference signal
    #y: estimated signal
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    # corrlation
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True)
    corr = np.mean((x-x_mean)*(y-y_mean), axis=0, keepdims=True)/(x_std*y_std)

    # MAP
    map = np.mean(np.abs(x-y), axis=0, keepdims=True)

    # MAPE
    mape = np.mean(np.abs((x - y) / x), axis=0, keepdims=True)


    return torch.tensor(corr), torch.tensor(map), torch.tensor(mape)


def call_option_pricer(spot, strike, maturity, r, vol):
    d1 = (log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / (vol * sqrt(maturity))
    d2 = d1 - vol * sqrt(maturity)

    price = spot * norm.cdf(d1) - strike * exp(-r * maturity) * norm.cdf(d2)
    return price


def put_option_pricer(spot, strike, maturity, r, vol):
    d1 = (log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / (vol * sqrt(maturity))
    d2 = d1 - vol * sqrt(maturity)

    price = -spot * norm.cdf(-d1) + strike * exp(-r * maturity) * norm.cdf(-d2)
    return price

