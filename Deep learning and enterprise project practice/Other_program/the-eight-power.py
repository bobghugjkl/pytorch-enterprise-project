import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

time_step = 1  # 利用多少时间窗口
batch_size = 32  # 批次大小
input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 3
epochs = 10
best_loss = 0
model_name = 'BiGRU'
save_path = './{}.pth'.format(model_name)

"""
这些参数是用来训练一个基于双向门控循环单元（BiGRU）神经网络模型的配置。让我逐一解释这些参数的含义：

1. **time_step = 1**:
   - 这个参数表示我们一次只使用一个时间窗口的数据来训练模型。这意味着每次输入到模型的数据都是独立的时间点。

2. **batch_size = 32**:
   - 批次大小是32，这表示每次训练时，模型会同时处理32个样本。批次大小决定了每次迭代中模型参数更新的频率。

3. **input_dim = 1**:
   - 输入维度是1，这意味着每个时间点输入到模型的数据只有一个特征。

4. **hidden_dim = 64**:
   - 隐藏层维度是64，这意味着每个GRU单元的隐藏层有64个神经元。这个参数决定了模型的复杂度和学习能力。

5. **output_dim = 1**:
   - 输出维度是1，这表示模型的输出只有一个特征。对于回归任务，这通常表示预测的值。

6. **num_layers = 3**:
   - 层数是3，这表示模型有3层GRU单元叠加在一起。增加层数可以使模型捕捉到更多复杂的模式。

7. **epochs = 10**:
   - 训练周期数是10，这表示模型会在所有训练数据上训练10次。每个周期都会遍历整个训练集一次。

8. **best_loss = 0**:
   - 最佳损失初始化为0，这个值用来存储训练过程中模型达到的最低损失值，以便保存最佳模型。

9. **model_name = 'BiGRU'**:
   - 模型名称是'BiGRU'，表示我们使用的是双向GRU模型。

10. **save_path = './{}.pth'.format(model_name)**:
    - 保存路径是'./BiGRU.pth'，表示训练好的模型会保存在当前目录下名为'BiGRU.pth'的文件中。

总结来说，这些参数配置是为了定义和训练一个双向GRU神经网络，用于处理时间序列数据。每个参数都有助于控制模型的结构和训练过程。
"""

df = pd.read_excel('./power_load_data.xlsx', header=None)
df = pd.DataFrame(df.values.reshape((-1, 1))[:10000])
scaler = StandardScaler()
scaler_model = StandardScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df.iloc[:, -1])).reshape((-1, 1))
"""
这些代码的作用是加载和预处理电力负载数据。具体步骤如下：

1. **读取数据**：从`power_load_data.xlsx`文件中读取数据。
2. **重塑数据**：将数据重塑为一列，并取前10000个数据点。
3. **标准化数据**：使用`StandardScaler`将数据标准化。

简洁解释：

1. 读取电力负载数据。
2. 重塑并截取前10000个数据点。
3. 标准化数据，使其均值为0，方差为1。
"""


def splot_data(data, timestep):
    dataX = []
    dataY = []
    for index in range(len(data) - timestep):
        dataX.append(data[index:index + timestep])
        dataY.append(data[index + timestep][0])
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    train_size = int(np.round(0.8 * dataX.shape[0]))
    x_train, x_test = dataX[: train_size, :].reshape(-1, timestep, 1), dataX[train_size:, :].reshape(-1, timestep, 1)
    y_train, y_test = dataY[: train_size], dataY[train_size:]
    return x_train, x_test, y_train, y_test


"""
这个函数的作用是将时间序列数据拆分为训练集和测试集。具体步骤如下：

1. **准备数据**：根据时间步长`timestep`，将数据分割成输入序列`dataX`和对应的目标值`dataY`。
2. **转换为数组**：将数据转换为NumPy数组。
3. **划分数据集**：将数据集按80%的比例划分为训练集和测试集。
4. **返回结果**：返回训练集和测试集的输入和输出数据。

简洁解释：

1. 按时间步长分割数据。
2. 转换为数组。
3. 按80%划分训练集和测试集。
4. 返回分割后的数据。
"""
# 获取训练集
x_train, x_test, y_train, y_test = splot_data(df, time_step)
# 转换为Tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
"""
在函数`split_data`中，`x`和`y`分别代表输入数据和目标数据。具体解释如下：

1. **`dataX` (即 `x`）**:
   - 这是输入数据，表示模型用来进行预测的数据序列。每个元素是一个时间窗口内的多个数据点。

2. **`dataY` (即 `y`）**:
   - 这是目标数据，表示模型需要预测的下一个时间点的值。

在训练和测试数据集中：

- **`x_train` 和 `x_test`**:
  - `x_train`：训练集的输入数据。
  - `x_test`：测试集的输入数据。

- **`y_train` 和 `y_test`**:
  - `y_train`：训练集的目标数据。
  - `y_test`：测试集的目标数据。

简洁解释：

- `x` 是模型用来预测的输入数据。
- `y` 是模型需要预测的目标数据。
"""
# 形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
# 迭代器
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# 一维卷积模块
class CNN(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 50, 1)
        self.maxpool1 = nn.AdaptiveAvgPool1d(output_size=100)
        self.conv2 = nn.Conv1d(50, 100, 1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(output_size=50)
        self.fc = nn.Linear(50 * 100, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = self.fc(x)
        return x


"""
这段代码定义了一个简单的一维卷积神经网络（CNN）用于处理时间序列数据。下面是每个部分的浅显易懂解释：

1. **定义CNN类**:
   - 这个类继承自PyTorch的`nn.Module`，用于构建神经网络模型。

2. **初始化方法 `__init__`**:
   - **`self.conv1`**: 一个一维卷积层，接收输入数据并输出50个特征。
   - **`self.maxpool1`**: 一个自适应平均池化层，将数据大小调整为100。
   - **`self.conv2`**: 另一个一维卷积层，接收50个特征并输出100个特征。
   - **`self.maxpool2`**: 一个自适应最大池化层，将数据大小调整为50。
   - **`self.fc`**: 一个全连接层，将卷积层和池化层处理后的数据转换为最终的输出。

3. **前向传播方法 `forward`**:
   - **`x = self.conv1(x)`**: 应用第一个卷积层。
   - **`x = self.maxpool1(x)`**: 应用第一个池化层。
   - **`x = self.conv2(x)`**: 应用第二个卷积层。
   - **`x = self.maxpool2(x)`**: 应用第二个池化层。
   - **`x = x.reshape(-1, x.shape[1] * x.shape[2])`**: 将数据展平成一个一维向量，以便输入全连接层。
   - **`x = self.fc(x)`**: 应用全连接层，生成最终输出。

简洁解释：

1. 这个CNN类用于处理一维时间序列数据。
2. 它包含两个卷积层和两个池化层，以及一个全连接层。
3. 前向传播方法中，数据依次通过卷积层、池化层、展平层和全连接层，生成最终的输出。
"""
model = CNN(output_dim, input_dim)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for data in train_bar:
        x_train, y_train = data
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch: [{}/{}] loss: {:.4f}".format(epoch + 1, epochs, loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), save_path)
print("Finished Training")

plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()).reshape(-1, 1)), "b")
plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)), "r")
plt.legend()
plt.show()

y_test_pred = model(x_test_tensor)
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((y_test_pred.detach().numpy())), "b")
plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)), "r")
plt.legend()
plt.show()
"""
# 创建CNN模型实例
model = CNN(output_dim, input_dim)

# 定义损失函数为均方误差损失
loss_function = nn.MSELoss()

# 定义优化器为Adam，并设置学习率为0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化运行损失
    train_bar = tqdm(train_loader)  # 显示进度条

    for data in train_bar:
        x_train, y_train = data  # 获取训练数据
        optimizer.zero_grad()  # 清空梯度
        y_train_pred = model(x_train)  # 预测输出
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()  # 累加损失
        train_bar.desc = "train epoch: [{}/{}] loss: {:.4f}".format(epoch + 1, epochs, loss)  # 更新进度条描述

    model.eval()  # 设置模型为评估模式
    test_loss = 0.0  # 初始化测试损失

    with torch.no_grad():  # 禁用梯度计算
        test_bar = tqdm(test_loader)  # 显示测试进度条

        for data in test_bar:
            x_test, y_test = data  # 获取测试数据
            y_test_pred = model(x_test)  # 预测输出
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))  # 计算测试损失

    if test_loss < best_loss:  # 如果测试损失更低
        best_loss = test_loss  # 更新最佳损失
        torch.save(model.state_dict(), save_path)  # 保存模型参数

print("Finished Training")  # 输出训练完成

# 绘制训练数据预测结果图
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()).reshape(-1, 1)), "b")  # 反向转换并绘制预测值
plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)), "r")  # 反向转换并绘制真实值
plt.legend()
plt.show()

# 绘制测试数据预测结果图
y_test_pred = model(x_test_tensor)
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((y_test_pred.detach().numpy())), "b")  # 反向转换并绘制预测值
plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)), "r")  # 反向转换并绘制真实值
plt.legend()
plt.show()

"""

