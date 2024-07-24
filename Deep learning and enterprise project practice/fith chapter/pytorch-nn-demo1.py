import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
"""
当然可以！我会逐行解释这个神经网络代码的每个部分及其功能和输出。

```python
import torch
import torch.nn.functional as F
import torch.nn as nn
```
- 这几行代码导入了PyTorch库中的几个模块，`torch`是主要的PyTorch库，`torch.nn`包含神经网络构建模块，`torch.nn.functional`包含激活函数等常用的功能函数。

```python
class Net(nn.Module):
```
- 这定义了一个名为`Net`的类，继承自`nn.Module`。`nn.Module`是所有神经网络模块的基类。

```python
    def __init__(self):
        super(Net, self).__init__()
```
- 这是类的初始化函数，用于定义网络的结构。`super(Net, self).__init__()`调用父类的初始化方法，确保所有父类的功能被正确初始化。

```python
        self.conv1 = nn.Conv2d(1, 6, 5)
```
- 这是第一个卷积层，`nn.Conv2d(1, 6, 5)`表示输入是1个通道（例如灰度图像），输出是6个通道，卷积核大小为5x5。

```python
        self.conv2 = nn.Conv2d(6, 16, 5)
```
- 这是第二个卷积层，输入是6个通道，输出是16个通道，卷积核大小为5x5。

```python
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
```
- 这是第一个全连接层，输入是`16 * 5 * 5`个神经元，输出是120个神经元。`16 * 5 * 5`是经过卷积和池化后的特征图展平后的大小。

```python
        self.fc2 = nn.Linear(120, 84)
```
- 这是第二个全连接层，输入是120个神经元，输出是84个神经元。

```python
        self.fc3 = nn.Linear(84, 10)
```
- 这是第三个全连接层，输入是84个神经元，输出是10个神经元。这通常用于10分类任务，如MNIST数字识别。

```python
    def forward(self, x):
```
- 这是定义了前向传播函数`forward`，表示数据如何通过网络进行传递。

```python
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
```
- 输入x通过第一个卷积层`self.conv1`，然后通过ReLU激活函数`F.relu`，接着通过最大池化层`F.max_pool2d`，池化窗口大小为2x2。

```python
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
```
- 输入x通过第二个卷积层`self.conv2`，然后通过ReLU激活函数，接着通过最大池化层，池化窗口大小为2x2。

```python
        x = x.view(-1, self.num_flat_features(x))
```
- 将x展平为一个一维向量，准备输入全连接层。`-1`表示自动计算该维度的大小，`self.num_flat_features(x)`计算展平后的特征数。

```python
        x = F.relu(self.fc1(x))
```
- x通过第一个全连接层`self.fc1`，然后通过ReLU激活函数。

```python
        x = F.relu(self.fc2(x))
```
- x通过第二个全连接层`self.fc2`，然后通过ReLU激活函数。

```python
        x = self.fc3(x)
```
- x通过第三个全连接层`self.fc3`，不再使用激活函数，因为输出是最终的分类结果。

```python
        return x
```
- 返回x，表示模型的输出。

```python
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```
- 这是一个辅助函数，用于计算展平后的特征数。`x.size()[1:]`获取x的所有维度，除了批次大小维度。然后通过循环计算所有维度的乘积，得到特征数。

```python
net = Net()
print(net)
```
- 创建一个`Net`类的实例`net`，并打印网络结构。

这个代码的目的是定义一个简单的卷积神经网络（CNN）来进行图像分类任务。每一层的功能如下：
1. 卷积层：提取特征
2. 激活函数：引入非线性
3. 池化层：减少特征图的尺寸
4. 全连接层：进行分类

希望这个解释能帮助你更好地理解每一行代码的作用！
"""
