import numpy as np

a = np.array([1, 2, 3])
print(a)
"""列表创建数组"""
a1 = np.array([1, 2, 3, 4])
lst1 = [3.14, 2.17, 0, 1, 2]
a2 = np.array([[1, 2], [3, 4], [5, 6]])
a3 = np.array(lst1)
print(a2)
print(a3)
"""NumPy函数创造"""
print(np.arange(12))
print(np.arange(5, 10))
print(np.random.randn(3, 3))
print(np.random.random([3, 4]))
print(np.ones((2, 4)))
print(np.zeros((3, 4)))
print(np.linspace(2, 8, 3, dtype=np.int32))
print("NumPy的算术运算")
"""同一数组"""
arr = np.arange(10)
print(arr + 1)
print(arr - 2)
print(arr * 3)
print(arr / 2)
"""不同数组"""
a = np.arange(1, 7).reshape((2, 3))
b = np.array([[6, 7, 8], [9, 10, 11]])
print("a + b:\n", a + b)
print("a - b:\n", a - b)
print("a * b:\n", a * b)
print("a / b:\n", a / b)
"""广播机制"""
b = np.arange(3)
print("a.shape:", a.shape)
print("b.shape:", b.shape)
print("a + b:\n", a + b)
print("a - b:\n", a - b)
print("a * b:\n", a * b)
""""行相同时，列也可以进行上述操作"""
a=np.arange(1,13).reshape((4,3))
b=np.arange(1,5).reshape((4,1))
print("a:\n:", a)
print("b:\n:", b)
print("a.shape:", a.shape)
print("b.shape:", b.shape)
print("a+b:\n", a+b)
""""数组变形"""
a=np.arange(1,10).reshape((3,3))
print(a)
""""reshape实现一维数组转二维"""
x=np.array([1,2,3])
y=x[np.newaxis,:]
z=x[:,np.newaxis]
print(x)
print(y)
print(z)
