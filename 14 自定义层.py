import torch
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
# 自定义了一个层的计算，这个模型是不可学习的，没有可学习的参数


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


net = MyDense()
print(net)
# 定义了一个含有可学习参数的层，参数定义为parameter类并附加在层类的属性中


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})  # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])


net = MyDictDense()
print(net)
# 通过ParameterDict构造参数，能够使用很多字典相关的参数，例如update, key, item
