import torch

# 创建一个2x2的张量，所有元素都为1，requires_grad=True表示需要计算梯度
x = torch.ones(2, 2, requires_grad=True)

# 对x每个元素加2，得到的新张量y
y = x + 2

# 打印y的grad_fn属性，显示创建y的操作函数
print(y.grad_fn)  # 这里会输出 <AddBackward0>

# 打印张量x的值
print(x)  # 这里会输出 tensor([[1., 1.], [1., 1.]], requires_grad=True)

# 打印张量y的值
print(y)  # 这里会输出 tensor([[3., 3.], [3., 3.]], grad_fn=<AddBackward0>)

# 对y的每个元素进行平方，然后乘以3，得到新张量z
z = y * y * 3

# 计算z的所有元素的平均值，得到标量out
out = z.mean()

# 打印张量z的值和标量out的值
print(z, out)  # z会输出 tensor([[27., 27.], [27., 27.]], grad_fn=<MulBackward0>)，out会输出 tensor(27., grad_fn=<MeanBackward0>)

# 打印x的梯度属性，初始值为None，因为还没有进行反向传播
print(x.grad)  # 这里会输出 None

# 对out进行反向传播，计算x的梯度
out.backward()

# 打印x的梯度
print(x.grad)  # 这里会输出 tensor([[4.5, 4.5], [4.5, 4.5]])

"""
解释：当我们对out调用backward()时，计算x的梯度，即对out关于x的导数。
out = (3 * (x + 2)^2).mean()
d(out)/d(x) = d(3 * (x + 2)^2) / d(x) * (1 / 4) = 4.5
"""

# 创建一个3x4的随机张量，并设置requires_grad=True
x = torch.randn(3, 4, requires_grad=True)
print(x)  # 输出一个3x4的随机张量，requires_grad=True

# 再次创建一个3x4的随机张量，但这次之后设置requires_grad=True
x = torch.randn(3, 4)
x.requires_grad = True
print(x)  # 输出另一个3x4的随机张量，requires_grad=True

# 创建一个3x4的随机张量b，并设置requires_grad=True
b = torch.randn(3, 4, requires_grad=True)

# x和b相加，得到张量t
t = x + b

# 对t的元素求和，得到标量y
y = t.sum()

# 打印标量y的值
print(y)  # 输出t的所有元素的和

# 对y进行反向传播，计算b的梯度
y.backward()

# 打印b的梯度
print(b.grad)  # 输出b的梯度

# 打印x, b和t的requires_grad属性
print(x.requires_grad, b.requires_grad, t.requires_grad)  # 输出 True, True, True
