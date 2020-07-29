import torch
from torch import nn

print(torch.cuda.is_available())  # 输出 True
# 查看GPU是否可用

print(torch.cuda.device_count())  # 输出 1
# 查看可用的GPU数量

torch.cuda.current_device()  # 输出 0
# 当前使用的设备序号

torch.cuda.get_device_name(0)  # 输出 'GeForce GTX 1050'
# 使用设备序号查看设备名称

x = torch.tensor([1, 2, 3])
x
# 默认下，tensor会被存在内存上

x = x.cuda(0)  # 或x = x.cuda()
x
# 将tensor转移到0号设备上

print(x.device)
# 查看对象所在的设备

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 检测并定义设备对象
x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
# 在创建时使用device属性或者to()方法指定所在设备

net = nn.Linear(3, 1)
list(net.parameters())[0].device
net.cuda()
list(net.parameters())[0].device
# 模型也可以放到GPU上
