import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(),
                    nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())
# 通过迭代器访问所有的参数及其名字，也可以使用net.parameters()访问


class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass


n = MyModel()
for name, param in n.named_parameters():
    print(name)
# 如上，weight1作为参数加入的网络结构中，所以能够使用named_parameters方法查到
# 而weight2则没有

weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)  # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)
# parameter是Tensor的一个子类

for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
# 使用正态分布来初始化权重
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)
# 使用常数初始化权重


def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)
# 自定义一个初始化方法，注意带上torch.no_grad()

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)
# 利用data来改变参数的值同时不影响参数的梯度计算

linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
# 通过利用序列重复调用同一个层对象，实现层参数的共享
