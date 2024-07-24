import torch

if torch.cuda.is_available():
    print('CUDA is available')
else:
    print('CUDA is not available')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.rand(3, 3)
x_dev = x.to(device)
"""创建一个简单的模型"""
model = torch.nn.Linear(10, 1)
data = torch.randn(100, 10)
if torch.cuda.is_available():
    model = model.to(('cuda'))
    data = data.to(('cuda'))
