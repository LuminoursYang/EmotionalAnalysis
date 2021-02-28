"""
coding: utf-8
Date  : 2020/9/2 14:29
File  : EmtNet.py
Software: PyCharm
Author: Lawrence.Yang
Email: Lawrence.Yang@connext.com.cn
"""
import torch
import torchsnooper
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import MinMaxScaler





class ResidualDense(nn.Module):
    def __init__(self, seq, batch_first=True):
        super(ResidualDense, self).__init__()
        self.seq = seq  # embagging shape
        self.width = 64
        self.avtivation = nn.LeakyReLU(negative_slope=1e-3)
        self.lstm1 = nn.LSTM(self.seq, self.width, 2, batch_first=batch_first)
        self.lstm2 = nn.LSTM(self.seq + self.width * 1, self.width, 2, batch_first=batch_first, dropout=0.3)
        self.lstm3 = nn.LSTM(self.seq + self.width * 2, self.width, 2, batch_first=batch_first, dropout=0.3)



    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.avtivation(out1)
        input1 = torch.cat([x, out1], dim=2)
        out2, _ = self.lstm2(input1)
        out2 = self.avtivation(out2)
        input2 = torch.cat([x, out1, out2], dim=2)
        out3, _ = self.lstm3(input2)
        out3 = self.avtivation(out3)

        return x + 0.2*out3




# @torchsnooper.snoop()
class EmtNet(nn.Module):
    def __init__(self, seq):
        super(EmtNet, self).__init__()
        self.width = 64
        self.seq = seq  # embagging shape
        self.lstm1 = nn.LSTM(self.seq, self.width, 2, batch_first=True, dropout=0.3)
        self.resblock1 = ResidualDense(self.width)
        self.resblock2 = ResidualDense(self.width)
        self.linear = nn.Linear(20*self.width, 3)
        self.activation = nn.LeakyReLU(negative_slope=1e-3)


    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = self.linear(x)
        x = self.activation(x)
        return x






if __name__ == '__main__':
    device = torch.device("cuda:0")
    x = torch.rand(1, 20, 100).to(device)
    net = EmtNet(x.shape[2]).to(device)
    print(net(x))
    # print(nn.LSTM(100, 1)(x))