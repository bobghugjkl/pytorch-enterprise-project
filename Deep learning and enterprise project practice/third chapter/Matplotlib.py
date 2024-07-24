import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)
# plt.plot(x, y)
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y)
plt.show()
"""添加标题"""
import matplotlib

matplotlib.rcParams['font.family'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 20
a = np.arange(0.0, 5.0, 0.02)
plt.title("演示")
plt.xlabel("纵轴： 振幅")
plt.ylabel("横轴： 时间")
plt.plot(a, np.cos(2 * np.pi * a), "r--")
"""a: 这是一个包含从0到5之间数字的数组，每隔0.02取一个值（例如，0, 0.02, 0.04,...，直到5）。这个数组代表了曲线的横坐标（X轴）。

np.cos(2 * np.pi * a): 这是计算每个a对应的余弦值。np.cos是NumPy库中的一个函数，用来计算余弦值。2 * np.pi * a表示将数组a中的每个值都乘以2π，然后再计算这些值的余弦值。这样得到的数组代表了曲线的纵坐标（Y轴）。

"r--": 这是一个格式字符串，用来指定绘制曲线的颜色和样式。在这里，"r"表示红色（red），"--"表示用虚线绘制。"""
plt.show()
"""饼图"""
plt.rcParams['font.sans-serif'] = ['SimHei']
activies = ['工作', "吃", "睡", "玩"]
times = [8, 7, 3, 6]
color = ['r', 'g', 'b', 'c']
plt.pie(times, labels=activies, shadow=True, colors=color, explode=(0, 0.1, 0, 0), autopct='%.1f%%')
"""`plt.pie(times, labels=activies, shadow=True, colors=color, explode=(0, 0.1, 0, 0), autopct='%.1f%%')` 这行代码用来绘制一个饼图，具体解释如下：

1. **`times`**: 这是一个包含时间分配的数组（8, 7, 3, 6），分别代表工作、吃、睡、玩的时间（单位：小时）。

2. **`labels=activies`**: 这是为每个扇区添加标签。在这里，`activies` 是一个数组，包含了各个活动的名称（工作、吃、睡、玩）。

3. **`shadow=True`**: 这表示饼图的扇区会有阴影效果，使图表更立体。

4. **`colors=color`**: 这是为每个扇区指定颜色。`color`是一个数组，包含了红色（r）、绿色（g）、蓝色（b）和青色（c）。

5. **`explode=(0, 0.1, 0, 0)`**: 这是将某个扇区从饼图中稍微分离出来。数组中每个值对应一个扇区，值为0表示不分离，值为0.1表示分离0.1的距离。在这里，第二个扇区（“吃”）被分离出来了一些。

6. **`autopct='%.1f%%'`**: 这是在每个扇区上显示其所占比例。`'%.1f%%'` 表示显示小数点后一位的百分比。

综合起来，`plt.pie(times, labels=activies, shadow=True, colors=color, explode=(0, 0.1, 0, 0), autopct='%.1f%%')`这行代码的意思是：绘制一个饼图，
显示一天中各活动所占的时间比例，并在每个扇区上显示其百分比。图表的外观包括阴影效果、指定颜色、“吃”这个扇区被稍微分离出来。"""
plt.title('饼图')
plt.show()
"""散点图"""
n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
"""np.random.normal: 这是NumPy库中的一个函数，用来生成符合正态分布的随机数。

0: 这是正态分布的均值（平均值）。在这里，均值是0。

1: 这是正态分布的标准差（衡量数据分布的宽度）。在这里，标准差是1。

n: 这是生成的随机数的个数。

综合起来，y = np.random.normal(0, 1, n)这行代码的意思是：生成n个随机数，这些随机数的分布符合均值为0、标准差为1的正态分布，并将这些随机数存储在数组y中。"""
plt.scatter(x, y)
plt.title("散点图")
plt.show()
"""柱状图"""
xiaoming_score = [80, 75, 65, 58, 75, 80, 90]
subjects = ["语文", "数学", "英语", "物理", "化学", "生物", "体育"]
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.bar(subjects, xiaoming_score)
plt.show()
# 水平
plt.barh(y=np.arange(7),
         height=0.35,
         width=xiaoming_score,
         label='xiao ming',
         edgecolor='k',
         color='r',
         tick_label=subjects,
         linewidth=3, )
plt.legend()
plt.show()
