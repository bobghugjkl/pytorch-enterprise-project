"""数据的加载和处理"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
"""
这段代码定义的 transform 变量包含了一个转换序列，这个序列首先将图像转换为张量，然后标准化每个通道的像素值，使其范围从 [0, 1] 转换到 [-1, 1]。
这种预处理通常在训练神经网络时使用，以加快收敛速度并提高模型性能。
"""
transet = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(transet, batch_size=4, shuffle=True)
testset = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)
"""保存加载"""
