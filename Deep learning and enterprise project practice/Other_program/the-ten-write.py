import torch
import torchvision.transforms as transforms
import torch.optim as optim
from jinja2.compiler import F
from torch import nn
from torch.utils.data import DataLoader
from lib.datasets.MyDataset import MyDataset

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(0.09) # dropout 是一种正则化技术，用于防止神经网络过拟合。在训练过程中，dropout 会随机将一部分神经元的输出设置为零，从而迫使模型的其他神经元进行更好的学习。
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(self.fc3(x))
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
main_dir = r"data\Mnist_Image"
model_dir = r"modelpath\Linear.pth"
save_imagepath = r"pouput\train_loss.png"

train_set = MyDataset(main_dir, istrain = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1037,),(0.3081,))]))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
net = LeNet5().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_list = []
for epoch in range(10):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,start=0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("epoch %d, batch %d, loss %f" % (epoch + 1, batch_idx + 1, running_loss / 300))
            loss_list.append(running_loss / 300)
            running_loss = 0.0
torch.save(net.state_dict(), r"modelpath\Linear.pth")
print("finish")
import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.title("loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(save_imagepath)
plt.show()