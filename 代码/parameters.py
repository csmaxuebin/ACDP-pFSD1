import argparse
import json
import sys

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return v


def get_args():

	parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")

	parser.add_argument("--epochs", default=100, type=int,help="通信轮数")
	parser.add_argument("--num_users", default=100, type=int, help="所有客户端个数")
	parser.add_argument("--frac", default=0.1, type=float, help="每轮参与客户端个数")
	parser.add_argument("--local_ep", default=5, type=int, help="本地epoch数")
	parser.add_argument("--local_bs", default=0, type=int, help="batchsize大小")
	parser.add_argument("--bs", default=0, type=int, help="")
	parser.add_argument("--lr", default=0, type=float, help="学习率")
	parser.add_argument("--momentum", default=0, type=float, help="动量")
	parser.add_argument("--split_ratio", default=0, type=float,help="训练集占比")
	parser.add_argument("--overlapping_classes", default=0, type=int,help="每个客户端拥有的类别")
	parser.add_argument("--base_layers", default=0, type=int,help="上传到客户端的层数，resnet34一共有218层")
	parser.add_argument("--model", default='ResNet', type=str,help="模型")
	parser.add_argument("--dataset", default='cifar', type=str,help="数据集")

	parser.add_argument("--useProximal", default=False, type=str2bool, help="是否使用近端项")
	parser.add_argument("--iid", default='noniid', type=str,help="是否设置为独立同分布")
	parser.add_argument("--num_classes", default=0, type=int,help="数据集类别数")
	parser.add_argument("--gpu", default=0, type=int,help="gpu编号")
	parser.add_argument("--seed", default=0, type=int,help="随机种子")
	parser.add_argument("--finetune", default=False, type=str2bool,help="是否微调")
	parser.add_argument("--algo",default='pFedSD', type=str, help="FedPer,pFedSD,pFedLSD")
	parser.add_argument("--dirichlet_alpha", default=0, type=float, help="第雷克磊分布参数")

#Ditto相关设置
	parser.add_argument('--mu', type=float, default=0,help="Regularization weight")

	#知识蒸馏相关设置
	parser.add_argument(
		"--useKD",
		type=str,
		default="no",
		choices=["no","SD","DSD"],
		help="是否使用知识蒸馏")
	parser.add_argument("--KL_T", default=0, type=int, help="蒸馏温度")
	parser.add_argument("--alpha", default=0, type=float, help="蒸馏损失占比")
	parser.add_argument("--ALPHA", default=0, type=float, help="DKD目标损失占比")
	parser.add_argument("--BETA", default=0, type=float, help="DKD非目标损失占比")

	#DP相关设置
	parser.add_argument("--useDP", default=False, type=str2bool, help="是否有DP保护")
	parser.add_argument("--useAdaclip", default=False, type=str2bool, help="是否自适应裁剪")
	parser.add_argument("--c", default=0, type=float, help="裁剪阈值初始值")
	parser.add_argument("--decay_lamda_1", default=0, type=float, help="裁剪阈值衰减因子")
	parser.add_argument("--decay_lamda_2", default=0, type=float, help="裁剪阈值衰减因子")
	parser.add_argument("--clipmethod", type=str,default="fix",
		choices=["fix", "my1","my2","my","RL","ppo","per","zhang"], help="选择哪种裁剪方法")
	parser.add_argument("--percent", default=0, type=float, help="范数的百分位数")
	parser.add_argument(
		"--dp_mode",
		type=str,
		default="overhead",
		choices=["overhead", "bounded"],
		help="Using which mode to do private training. Options: overhead, bounded.",
	)
	parser.add_argument("--accountant", type=str, default="rdp", help="The dp accountant")
	parser.add_argument("--noise_multiplier", type=float, default=None, help="噪声因子z")
	parser.add_argument("--dp_epsilon", default=None, type=float, help="隐私预算")
	parser.add_argument("--dp_delta", default=0, type=float, help="违反差分隐私的概率")

	# parser.add_argument("--dp_epsilon", default=1, type=float, help="隐私预算")


	return parser.parse_args()

if __name__ == "__main__":
	args = get_args()
