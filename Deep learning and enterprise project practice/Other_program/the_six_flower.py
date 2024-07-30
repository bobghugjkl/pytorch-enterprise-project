import torchvision
from torch import nn
import os
import pickle
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm
from PIL import Image

epochs = 10
lr = 0.03
batch_size = 32
image_path = './flowerdata/data'
save_path = './chk/flower_model.pkl'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose({
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                     transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                   transform=data_transforms['val'])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size,
                                           True)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size,True)
print('using {} images from training. '.format(len(train_dataset)))
print('using {} images from validation. '.format(len(val_dataset)))

cloth_list = train_dataset.class_to_idx
class_dict = {}
for key, val in cloth_list.items():
    class_dict[val] = key
with open('class_dict.pk', 'wb') as f:
    pickle.dump(class_dict, f)

model = torchvision.models.resnet152(
    weights=torchvision.models.Resnet152_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

best_acc = 0
best_model = None
for epoch in range(epochs):
    model.train()
    running_loss = 0
    epoch_acc = 0
    epoch_acc_count = 0
    train_count = 0
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        optimizer.zero_grad()
        output = model(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch: {} loss: {:.4f}".format(epoch + 1, epochs, loss)
        epoch_acc_count += (output.argmax(axis=1) == labels.view(-1)).sum()
        train_count += len(images)
    epoch_acc = epoch_acc_count / train_count
    print("EPOCH: %s" % str(epoch + 1))
    print("训练损失：%s" % str(running_loss))
    print("训练精度：%s" % (str(epoch_acc.item() * 100)[:5]) + '%')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = model.state_dict()
    if epoch == epochs - 1:
        torch.save(best_model, save_path)
print('Finished Training')
with open('class_dict.pk', 'rb') as f:
    class_dict = pickle.load(f)
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_path = r'./flowerdata/test/test01.jpg'
img = Image.open(img_path)
img = data_transform(img, dim=0)
pred = class_dict[model(img).argmax(axis=1).item()]
print("预测结果：%s" % pred)
