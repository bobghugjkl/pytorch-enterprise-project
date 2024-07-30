import torchvision
from torch import nn
import numpy as np
import os
import json
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

"""训练过程会重复10次，每次使用整个数据集进行一次完整的训练，以便模型能够逐步优化和收敛。"""
epochs = 10
"""这是学习率（learning rate），用于控制模型在每次更新时步长的大小"""
lr = 0.03
"""这是每次训练迭代中使用的样本数量。"""
batch_size = 32
image_path = './garbage_data/data'
save_path = './garbage_chk/best_model.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    """data_transforms 是一个包含训练数据转换操作的字典。字典的键是 'train'，表示这些转换将应用于训练数据集。"""
    'train': transforms.Compose({
        # transforms.Compose 是 PyTorch 提供的一个类，用于将多个转换操作按顺序组合在一起。
        # 随机裁剪图像并调整大小，使其成为 224x224 的正方形。
        transforms.RandomResizedCrop(224),
        # 以 50% 的概率随机水平翻转图像。
        transforms.RandomHorizontalFlip(),
        # 将 PIL 图像或 numpy 数组转换为 PyTorch 的张量格式，并将像素值归一化到 [0, 1] 范围。
        transforms.ToTensor(),
        # 对图像数据进行标准化。第一个参数是每个通道的均值，第二个参数是每个通道的标准差。
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }),
}
"""训练集"""
train_dataset = datasets.ImageFolder(root=os.path.join(image_path), transform=data_transforms['train'])
"""迭代器"""
# 将数据集分成小批次，并在训练过程中按批次加载数据。设置 batch_size 为批次大小，True 表示每次迭代时打乱数据。
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
# 打印训练数据集中图像的总数量。
print('using {} images for training'.format(len(train_dataset)))
# train_dataset.class_to_idx 返回一个字典，其中键是类名，值是类对应的索引。接下来，
# 将这个字典的键值对互换，生成新的字典 class_dict，其中键是索引，值是类名。
cloth_list = train_dataset.class_to_idx
class_dict = {}
for key, val in cloth_list.items():
    class_dict[val] = key
# 使用 pickle 将 class_dict 字典保存到文件 class_dict.pk 中。
with open('class_dict.pk', 'wb') as f:
    pickle.dump(class_dict, f)

"""自定义损失函数"""


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, label):
        # 解释：对模型的预测值 pred 计算指数。
        # 作用：将预测值从对数几率（logits）转换为未归一化的概率。
        exp = torch.exp(pred)
        # 解释：从 exp 中提取每个样本对应正确类别的指数值。
        # 作用：将标签 label 作为索引，从 exp 中提取对应类别的指数值。
        tmp1 = exp.gather(1, label.unsqueeze(-1)).squeeze()
        # 解释：计算每个样本所有类别的指数值之和。
        # 作用：用于归一化计算，得到每个样本的总概率。
        tmp2 = exp.sum(1)
        # 解释：计算每个样本对应正确类别的softmax值。
        # 作用：得到正确类别的归一化概率。
        softmax = tmp1 / tmp2
        # 解释：计算每个样本正确类别的负对数概率。
        # 作用：用于计算交叉熵损失。
        log = -torch.log(softmax)
        # 解释：返回所有样本的平均损失值。
        # 作用：得到最终的损失值，用于优化模型。
        return log.mean()


model = torchvision.models.mnasnet1_0(
    weights=torchvision.models.MNASNet1_0_Weights.IMAGENET1K_V1
)
"""冻结模型参数"""
for param in model.parameters():
    param.requires_grad = False
"""修改最后一层全连接"""
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 12)
model = model.to('cpu')
"""使用自定义的损失函数"""
criterion = MyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
"""模型训练"""
best_acc = 0
best_model = None

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    epoch_acc = 0
    epoch_acc_count = 0
    train_count = 0
    # train_bar = tqdm(train_loader) 这行代码的作用是为训练数据加载器 train_loader 创建一个带进度条的迭代器，方便在训练过程中实时观察训练进度。
    train_bar = tqdm(train_loader)
    for data in train_bar:
        #解释：将当前批次的数据分解为图像和标签。
        #作用：分别获取输入图像和对应的标签。
        images, labels = data
        #解释：将模型的梯度缓存清零。
        #作用：防止梯度累积，因为 PyTorch 中的梯度默认是累加的。
        optimizer.zero_grad()
        #前向传播，预测输出
        outputs = model(images.to(device))
        #使用定义好的函数来计算损失值
        loss = criterion(outputs, labels.to(device))
        #反向传播，然后优化器更新参数
        loss.backward()
        optimizer.step()

        #解释：将当前批次的损失值累加到 running_loss 变量中。
        #作用：累积计算整个 epoch 的总损失，以便后续计算平均损失。
        running_loss += loss.item()
        #解释：更新 tqdm 进度条的描述信息，显示当前 epoch 和损失值。
        #作用：实时显示训练进度和当前损失值。
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        #计算当前批次准确预测的数量
        epoch_acc_count += (outputs.argmax(axis=1) == labels.view(-1)).sum()
        #计算当前批次预测总数
        train_count += len(images)

    epoch_acc = epoch_acc_count / train_count
    print("{EPOCH} %s" % str(epoch + 1))
    print("训练损失为%s" % str(running_loss))
    print("训练精度为%s" % (str(epoch_acc.item() * 100)[:5]) + '%')
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = model.state_dict()
    #如果是最后一个epoch，则将最好的模型保存
    if epoch == epochs - 1:
        torch.save(best_model, save_path)
print('Finished Training')
#加载类的信息，如索引
with open('class_dict.pk', 'rb') as f:
    class_dict = pickle.load(f)
data_transforms = transforms.Compose([
    #将图像缩放到 256 像素
    transforms.Resize(256),
    #从图像中心裁剪出 224x224 的区域。
    transforms.CenterCrop(224),
    #将图像转换为 PyTorch 张量，并将像素值归一化到 [0, 1] 范围。
    transforms.ToTensor(),
])
#加载与预处理
img_path = r'./garbage_data/text/shoes1750.jpg'
img = Image.open(img_path)
img = data_transforms(img)

plt.imshow(img.permute(1, 2, 0))
plt.show()

#增加一个批次维度，使图像张量形状变为 [1, 3, 224, 224]，符合模型输入格式。
img = torch.unsqueeze(img, 0)
#带入模型，获取预测结果中概率最高的类别索引，并获取其索引的名称
pred = class_dict[model(img).argmax(dim=1).item()]
print(pred)
