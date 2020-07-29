'''
torchvision包的构成
1. torchvision.datasets 加载数据的函数和常用的数据集接口
2. torchvision.models 包含常用的模型结构和预训练模型
3. torchvision.transforms 常用的图片变换
4. torchvision.utils 其他方法
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='D:\Github\Datasets\FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='D:\Github\Datasets\FashionMNIST', train=False, download=False, transform=transforms.ToTensor())
'''
transforms.ToTensor()会把PIL图片转换为Tensor，具体是
将尺寸为(H, W, C)且数据位于[0, 255]的PIL图片或数据类型为np.uint8的numpy数据转换为
尺寸为(C, H, W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor
'''

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, lable)

def get_fashion_mnist_labels(labels):
	'''
	将数字转换成对应标签文本

	'''
	text_labels = ['t恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
	return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
	'''
	传入若干张图片和对应的标签，用一行排列显示

	'''
	d2l.use_svg_display()
	_, figs = plt.subplots(1, len(images), figsize=(12, 12))
	for f, img, lbl in zip(figs, images, labels):
		f.imshow(img.view((28, 28)).numpy())
		f.set_title(lbl)
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
	plt.show()

# 显示前十个样本
X, y = [], []
for i in range(10):
	X.append(mnist_train[i][0])
	y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 通过数据加载器读取数据
def load_data_fashion_mnist(batch_size):
	'''
	将样本按给定批次大小划分并读取

	'''
	if sys.platform.startswith('win'):
		num_workers = 0
	else:
		num_workers = 4
	train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_iter, test_iter

# 查看读取一遍训练集需要的时间
start = time.time()
for X, y in train_iter:
	continue
print('%.2f sec' % (time.time() - start))

