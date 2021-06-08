# 3D Tensor-based Deep Learning Models for Predicting Option Price
This code is written for the manuscript: 3D Tensor-based Deep Learning Models for Predicting Option Price. 

There are three folders in the project. In "data" part, there is 50ETF option and stock option data collected from Chinese stock and fund market. The data is divided into training set and test set. "test_data_BS.py" is specially used for B-S model. There is also a folder named "torch-data", which saves the torch tensors created by "data_process.py".

In "core" part, you can find four files used for constructing the models mentioned in the manuscript, and a single file for data processing.

In "main" part, we could apply the models and the data to predict option price, and compare their strengths and weaknesses through five measurements: MSE, RMSE, MAP, MAPE and PCC.

For more details, please refer to the original manuscript: 3D Tensor-based Deep Learning Models for Predicting Option Price. arXiv: https://arxiv.org/abs/2106.02916
