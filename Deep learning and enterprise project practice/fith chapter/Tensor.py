import torch
import numpy as np

a = np.array([3.0, 4.0, 5.0])
"""numpy -> tensor"""
b = torch.from_numpy(a)
print(b)
"""make tensor directely"""
a = torch.tensor([[3, 4, 5], [6, 7, 8]])
print(a)
print(a.type())
"""input num to create tensor"""
b = torch.FloatTensor([[3, 4, 5], [6, 7, 8]])
print(b)
print(b.type())
"""full of 0 or 1"""
a = torch.full((2, 3), 1)
print(a)
print(a.type())
"""step by step number"""
a = torch.arange(0, 10, 2)
print(a)
print(a.type())
"""use random num"""
a = torch.rand(2, 3)
print(a)
b = torch.randint(1, 10, (2, 3))
print(b)
x = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
"""just like x"""
c = torch.rand_like(x)
print(c)
"""randn 正态分布，normal 为均值，标准差，形状，输出一个满足上述参数的广义正态分布张量"""
a = torch.randn(2, 3)
print(a)
print(a.type())
b = torch.normal(1, 10, (2, 3))
print(b)
print(b.type())
"""张量基本操作"""
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[5, 6], [7, 8]])
print("a + b: ", torch.add(a, b))
print("a * b: ", torch.mul(a, b))
print("a / b: ", torch.div(a, b))
print("a ** b: ", torch.pow(a, b))
print("a - b: ", torch.sub(a, b))
print("|a|: ", torch.abs(a))
"""张量矩阵运算"""
c = torch.matmul(a, b)
print(c)
"""张量逐元素乘法"""
d = torch.mul(a, b)
print(d)
"""切片"""
aa = torch.randn(4, 3)
print(aa)
print(aa[-1:])
print(aa[-2:])
print(aa[:2])
print(aa[:-1])
