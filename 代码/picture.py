import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# 加载数据
with open('pic/cifar10+sigma2.07+nls2.txt', 'r') as f1:
    data1 = f1.readlines()  # 加载第一个文件的数据

# with open('log/clip3/FedPer_partition_n_cls_local_ep_4_overlapping_classes_40_model_ResNet18_dataset_cifar100_local_bs_128_dp_clip_1_lr_0.01_epochs_100_split_ratio_0.9_num_users_10_DPLAC.txt', 'r') as f2:
#     data2 = f2.readlines()  # 加载第二个文件的数据
# #
# with open('log/clip3/FedPer_partition_n_cls_local_ep_4_overlapping_classes_4_model_ResNet18_dataset_svhn_local_bs_128_dp_clip_1_lr_0.01_epochs_100_split_ratio_0.9_num_users_10_固定值裁剪.txt', 'r') as f3:
#     data3 = f3.readlines()  # 加载第三个文件的数据
#
# with open('log/clip3/FedPer_partition_n_cls_local_ep_4_overlapping_classes_4_model_ResNet18_dataset_svhn_local_bs_128_dp_clip_1_lr_0.01_epochs_100_split_ratio_0.9_num_users_10——张.txt', 'r') as f4:
#     data4 = f4.readlines()  # 加载第四个文件的数据



# 从每行数据中提取准确率值，分别保存到两个数组中
acc1 = [float(line.split()[-1]) for line in data1 if "test" in line and "Best" not in line]
acc2 = [float(line.split()[-1]) for line in data1 if "test" in line and "Best" not in line]
acc3 = [float(line.split()[-1]) for line in data1 if "test" in line and "Best" not in line]
# acc4 = [float(line.split()[-1]) for line in data4 if "test" in line and "Best" not in line]
print(acc1)
# print(acc2)
# print(acc3)
# print(acc4)
# 使用步长为10的range函数生成x轴坐标
rounds = list(range(1, 100, 5))

# 设置纵坐标范围为0到100
plt.ylim(90, 100)
y_ticks = [i * 10 for i in range(11)]
y_major_locator = FixedLocator(y_ticks)
plt.gca().yaxis.set_major_locator(y_major_locator)

# 绘制每个文件的准确率数据
plt.plot(rounds, acc1[::5], color=(190/255, 42/255, 44/255), marker='*', label='ACDP')  # Only every 10th data point is plotted
plt.plot(rounds, acc2[::5], color=(53/255, 144/255, 58/255), marker='o', label='DPLAC')
plt.plot(rounds, acc3[::5], color=(228/255, 123/255, 38/255), marker='s', label='DP-pFedCKD')
plt.plot(rounds, acc4[::5], color=(32/255, 110/255, 158/255), marker='p', label='ULDP-FED')

# 绘制每个文件的准确率数据
# plt.plot(rounds, acc1, color='darkorange',  label='FedAvg')
# plt.plot(rounds, acc2, color='springgreen',  label='FedProx')
# plt.plot(rounds, acc3, color='gold', label='FedPer')
# plt.plot(rounds, acc4, color='red',  label='FedVF')
# plt.plot(rounds, acc5, color='cornflowerblue',  label='PDP-FD')
# plt.plot(rounds, acc6, color='violet',  label='dynamic_Positive>negative_--')
# 添加标签和图例
plt.xlabel('Round', fontsize='18')
plt.ylabel('Accuracy(%)', fontsize='18')
# plt.title('Training Results')
plt.legend(loc='best', fontsize='15')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(r"SVHN", fontsize='20')
# 保存图为svg格式，即矢量图格式
plt.savefig("picture/clip_SVHN.svg", dpi=300,format="svg")

# # 保存图为eps格式
# plt.savefig("picture/acc.eps", dpi=300, format="eps")

# 显示图表
plt.show()