import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

num_inputs = 2  # 输入维度
num_examples = 1000  # 输入样本数
true_w = [2, -3.4]  # 真实值w
true_b = 4.2  # 真实值b
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)  # 生成指定形状的输入数据
labels = true_w[0] * features[:, 0] + true_w[1] * \
    features[:, 1] + true_b  # 根据公式生成标签值
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size(), dtype=torch.float32))  # 加入噪音


def use_svg_display():
    '''
    设置图片格式

    '''
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 3.5)):
    '''
    设置图片尺寸

    '''
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

# 书中建议将以上两个函数保存为d2lzh_pytorch包中，以后就直接调用d2lzh_pytorch.plt
# 以完成画图，直接调用d2lzh_pytorch.set_figsize设置图尺寸


def data_iter(batch_size, features, labels):
    '''
    将所有输入数据按批次分割迭代

    arg:
            batch_size:批次大小
            features:输入特征
            labels:输入标签
    return:
            可迭代的批次容器

    '''
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        # 这一句使用min是因为最后一个批次可能不足一个完整批次
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X)
    print(y)
    break
# 尝试打印第一个批次

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 因为w和b是需要更新的参数，所有要把它们的梯度监控打开

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型


def linreg(X, w, b):
    '''
    线性回归计算表达式 Y = X * w + b

    '''
    return torch.mm(X, w) + b

# 定义平均损失函数


def squared_loss(y_hat, y):
    '''
    平方损失函数

    '''
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    '''
    随机梯度下降法

    arg:
            params:参数的容器
            lr:学习率
            batch_size:批次大小

    '''
    for param in params:
        param.data -= lr * param.grad / batch_size


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels).mean().item()
    print('epoch %d, loss %f' % (epoch + 1, train_l))
