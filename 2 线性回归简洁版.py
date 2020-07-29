from collections import OrderedDict
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(
	0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01,
										size=labels.size()), dtype=torch.float)


batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


for X, y in data_iter:
	print(X, y)
	break


class LinearNet(nn.Module):
	def __init__(self, n_feature):
		super(LinearNet, self).__init__()
		self.linear = nn.Linear(n_feature, 1)
	# forward 定义前向传播

	def forward(self, x):
		y = self.linear(x)
		return y


net = LinearNet(num_inputs)
print(net)  # 使用print可以打印出网络的结构

# 以下将使用Sequential的各种写法搭建相同的网络

net = nn.Sequential(
	nn.Linear(num_inputs, 1)
)

net = nn.Sequential()
net.add_module('Linear', nn.Linear(num_inputs, 1))

net = nn.Sequential(OrderedDict([
	('linear', nn.Linear(num_inputs, 1))
]))

print(net)  # 查看网络结构
print(net[0])  # 查看某层信息

# net.parameters()所有可学习参数的生成器

for param in net.parameters():
	print(param)

# 模型参数初始化，这里示范用平均值为0，标准差为1的正态分布

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
# 只有当net是使用ModuleList或者Sequential定义时才可以使用下标访问
# 否则应该使用net.linear.weight

# 均方误差
loss = nn.MESLoss()

# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 也可以对不同子网络设置不同的学习率,如果对某个参数不指定学习率，就使用最外层的默认学习率
optimizer = optim.SGD([
	{'params': net.subnet1.parameters()},
	{'params': net.subnet2.parameters(), 'lr': 0.01}
], lr=0.03)

# 调整学习率
for param_group in optimizer.param_groups:
	param_group['lr'] *= 0.1  # 学习率为之前的0.1倍

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
	for X, y in data_iter:
		output = net(X)
		l = loss(output, y.view(-1, 1))
		optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
		l.backward()
		optimizer.step()
	print('epoch %d, loss: %f' % (epoch, l.item()))

# 对比学习到的参数和真实参数
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
