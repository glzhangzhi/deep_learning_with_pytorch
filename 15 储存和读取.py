import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
x2
# 向量的写和读

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
xy_list
# list的读和写

torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
xy
# 字典的写和读


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
net.state_dict()
# 模型的state_dict
# 注意只有具有可学习参数的层才有state_dict

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()
# 优化器中也有自己的state_dict，保存了优化器的状态和所使用的超参

torch.save(model.state_dict(), PATH)  # 推荐的文件后缀名是pt或pth
# 模型的保存

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
# 模型的加载

torch.save(model, PATH)
model = torch.load(PATH)
# 整个模型的保存和加载

X = torch.randn(2, 3)
Y = net(X)

PATH = "./net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
Y2 == Y
