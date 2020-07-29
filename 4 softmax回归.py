import torch
import torchvision
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 输入图片尺寸为28x28，输出类别为10
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)))
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 只对行或列求和，并保留行列分布
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))

# 以下代码实现softmax算法
def softmax(X):
	X_exp = X.exp()
	partition = X_exp.sum(dim=1, keepdim=True)
	return X_exp / partition

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))

# 定义模型
def net(X):
	return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# 定义损失函数，使用gather函数，例如
# y_hat为两个样本对应三个类别的预测概率，y为样本对应的label
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))

def cross_entropy(y_hat, y):
	return - torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
	return (y_hat.argmax(dim=1) == y).float().mean().item()
# argmax(dim=1)返回每行中最大值的索引

def evaluate_accuracy(data_iter, net):
	'''
	计算模型在数据集上的准确率
	arg:
		data_iter: 数据集(X, y)
		net: 模型y_hat=net(X)

	'''
	acc_sum, n = 0.0, 0
	for X, y in data_iter:
		# acc_sum += (net(X).argmax(dim=1) == y).float().mean().item()
		acc_sum += accuracy(net(X), y)
		n += y.shape[0]
	return acc_sum / n

num_epochs, lr = 5, 0.1

# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
	'''
	训练模型，打印每一个批次的损失值，训练集准确率，测试集准确率
	arg:
		net:定义的模型
		train_iter:训练集迭代器
		test_iter:测试集迭代器
		loss:损失函数
		num_epochs:训练批次数
		batch_size:批次大小
		params:目标训练参数
		lr:学习率
		optimizer:优化器

	'''
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])
