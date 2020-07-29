'''
如果输入的尺寸是[nh, nw]，卷积核窗口尺寸是[kh, kw]
那么输出尺寸为[nh - kh + 1, nw - kw + 1]


# 填充padding
在输入两侧填充0元素
如果在高的两侧一共填充ph行，在宽的两侧一共填充pw列
那么输出尺寸为[nh - kh + ph + 1, nw - kw + pw + 1]
所以通常我们会设置ph = kh - 1, pw = kp - 1
这样就会将输出和输入维持在同一个尺寸
'''
import torch
from torch import nn


def comp_conv2d(conv2d, X):
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape

# 改变步幅，当高上步幅为sh，宽上步幅为sw时，输出形状为
# [(nh - kh + ph + sh)/sh, (nw - kw + pw + sw)/sw]
# 如果像之前说的那样设置填充的距离，那么输出形状为
# [(nh + sh - 1) / sh, (nw + sw - 1) / sw]
# 如果输入的高和宽分别能被高和宽上的步幅整除，那么输出形状为
# [nh / sh, nw / sw]
# 如果令高和宽上的步幅为2，就可以使输入的高和宽减半

con2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(con2d, X).shape

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
