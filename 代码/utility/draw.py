import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

#用来画图的文件
#picture文件夹用来放图片
#plant_data文件中放的要用的txt文件

if not os.path.exists('../picture'):
    os.makedirs('../picture')

# 加载数据
with open('../plant_data/test.txt', 'r') as f1:
    data1lines = f1.readlines()  # 加载第一个文件的数据

with open('../plant_data/pFedLSD.txt', 'r') as f2:
    data2lines = f2.readlines()  # 加载第二个文件的数据

# with open('plant_data/fedper2-p88.72.txt', 'r') as f3:
#     data3 = f3.readlines()  # 加载第三个文件的数据
#
# with open('plant_data/fedvf2-p90.18.txt', 'r') as f4:
#     data4 = f4.readlines()  # 加载第四个文件的数据
#
# with open('plant_data/fedkd2-p92.68.txt', 'r') as f5:
#     data5 = f5.readlines()  # 加载第五个文件的数据

# with open('plant_data/2-84.68.txt', 'r') as f6:
#     data6 = f6.readlines()  # 加载第六个文件的数据

# 从每行数据中提取准确率值，分别保存到两个数组中
acc1 = [float(line.split()[-1]) for line in data1lines]
acc2 = [float(line.split()[-1]) for line in data2lines]
# acc3 = [float(line.split()[-1]) for line in data3][:100]
# acc4 = [float(line.split()[-1]) for line in data4][:100]
# acc5 = [float(line.split()[-1]) for line in data5][:100]
# acc6 = [float(line.split()[-1]) for line in data6][:100]

# # 创建一个表示轮数的列表（假设每行数据对应一轮训练）
# rounds = list(range(len(acc1)))

# 使用步长为10的range函数生成x轴坐标
rounds = list(range(1, 100, 10))

# 设置纵坐标范围为0到100
plt.ylim(0, 100)
y_ticks = [i * 10 for i in range(11)]
y_major_locator = FixedLocator(y_ticks)
plt.gca().yaxis.set_major_locator(y_major_locator)

# 绘制每个文件的准确率数据
plt.plot(rounds, acc1[::10], color='darkorange', marker='s', label='FedSD')  # Only every 10th data point is plotted
plt.plot(rounds, acc2[::10], color='springgreen', marker='o', label='FedLSD')
# plt.plot(rounds, acc3[::5], color='gold', marker='p', label='FedPer')
# plt.plot(rounds, acc4[::5], color='red', marker='*', label='FedVF')
# plt.plot(rounds, acc5[::5], color='cornflowerblue', marker='^', label='PDP-FD')


# 添加标签和图例
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy')
# plt.title('Training Results')
plt.legend()

# 保存图为svg格式，即矢量图格式
plt.savefig("../picture/acc.svg", dpi=300,format="svg")

# 显示图表
plt.show()
