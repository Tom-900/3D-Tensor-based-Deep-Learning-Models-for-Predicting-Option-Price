import pandas as pd
import numpy as np
import torch
from tqdm import trange

train_data = pd.read_csv(r"\train_data.csv")
test_data = pd.read_csv(r"\test_data.csv")

train_input = []
train_label = []
seq_len = 10

for i in trange(train_data['optID'].unique().shape[0]):
    for index in range(train_data.loc[(train_data['optID'] == train_data['optID'].unique()[i])].shape[0] - seq_len+1):
        input = torch.zeros((3, 10, 5))
        tmp_input = np.array(train_data.loc[(train_data['optID'] == train_data['optID'].unique()[i])][index:index+seq_len])
        input[0, :] = torch.tensor(np.array(tmp_input[0:seq_len, 3:8], dtype=np.float64))
        input[1, :] = torch.tensor(np.array(tmp_input[0:seq_len, 8:13], dtype=np.float64))
        input[2, :] = torch.tensor(np.array(tmp_input[0:seq_len, 13:18], dtype=np.float64))
        label = torch.tensor(np.array(tmp_input[seq_len-1, 18], dtype=np.float64))
        train_input.append(input)
        train_label.append(label)

zeros = torch.zeros((len(train_input), 3, 10, 5))
for i in range(zeros.shape[0]):
    zeros[i] = train_input[i]
train_input = zeros
print("Shape of the Training Input:", train_input.shape)

train_label = torch.tensor(train_label)
print("Shape of the Training Label:", train_label.shape)


test_input = []
test_label = []

for i in trange(test_data['optID'].unique().shape[0]):
    for index in range(test_data.loc[(test_data['optID'] == test_data['optID'].unique()[i])].shape[0] - seq_len+1):
        input = torch.zeros((3, 10, 5))
        tmp_input = np.array(test_data.loc[(test_data['optID'] == test_data['optID'].unique()[i])][index:index+seq_len])
        input[0, :] = torch.tensor(np.array(tmp_input[0:seq_len, 3:8], dtype=np.float64))
        input[1, :] = torch.tensor(np.array(tmp_input[0:seq_len, 8:13], dtype=np.float64))
        input[2, :] = torch.tensor(np.array(tmp_input[0:seq_len, 13:18], dtype=np.float64))
        label = torch.tensor(np.array(tmp_input[seq_len-1, 18], dtype=np.float64))
        test_input.append(input)
        test_label.append(label)

zeros = torch.zeros((len(test_input), 3, 10, 5))
for i in range(zeros.shape[0]):
    zeros[i] = test_input[i]
test_input = zeros
print("Shape of the Test Input:", test_input.shape)

test_label = torch.tensor(test_label)
print("Shape of the Test Label:", test_label.shape)

torch.save(train_input, r"\torch-data\train_input.pt")
torch.save(train_label, r"\torch-data\train_label.pt")
torch.save(test_input, r"\torch-data\test_input.pt")
torch.save(test_label, r"\torch-data\test_label.pt")









