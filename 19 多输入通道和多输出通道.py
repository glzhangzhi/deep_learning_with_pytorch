# 构造的卷积核的通道数应与输入的通道数相同
# 每个通道与其相对应的卷积核做互相关运算
# 将得到的输出的所有通道相加

import torch
from torch import nn
import sys


def corr2d(X, K):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

output = corr2d_multi_in(X, K)

print(X.shape, K.shape, output.shape)


def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K + 1, K + 2])
K.shape  # torch.Size([3, 2, 2, 2])

corr2d_multi_in_out(X, K)


def corr2d_multi_in_out_1x1(X, K):
        # X [ci, h, w]
        # K [co, ci, kh, kw]
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    # X [ci, h*w]
    K = K.view(c_o, c_i)
    # K [co, ci]
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    # Y [co, h*w]
    return Y.view(c_o, h, w)
    # Y [co, h, w]


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

(Y1 - Y2).norm().item() < 1e-6
