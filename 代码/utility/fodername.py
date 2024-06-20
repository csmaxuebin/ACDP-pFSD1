import os
import datetime

from parameters import get_args


def create_dynamic_folder_name(args):
    if args.useDP:
        # 给定epsilon
        if args.dp_mode == "overhead":
            assert args.dp_epsilon is not None or args.noise_multiplier is not None
            if args.dp_epsilon is not None:
                dp_info = f"dpEpsilon{args.dp_epsilon}"
            else:
                dp_info = f"z{args.noise_multiplier}"
        elif args.dp_mode == "bounded":
            assert args.dp_epsilon is not None and args.noise_multiplier is not None
            dp_info = f"bounded_dpEpsilon{args.dp_epsilon}_z{args.noise_multiplier}"
        if args.clipmethod== 'fix':
            dp_info += f"_fixclip{args.c}"
        else:
            dp_info += f"clipmethod_{args.clipmethod}_adaclip{args.c}_p{args.percent}"
    else:
        dp_info = "{}"
    if args.useKD == "no":
        kd_info = "{}"
        if args.useKD == "no" and (args.algo == "FedPer" or args.algo == "FedPerGlobal"):
            kd_info = f"baselayers{args.base_layers}"
    elif args.useKD == "SD":
        kd_info = f"T{args.KL_T}"
    elif args.useKD == "DSD":
        kd_info = f"T{args.KL_T}_alpha{args.alpha}_ALPHA{args.ALPHA}_BETA{args.BETA}"
    # Taking hash of config values and using it as filename for storing model parameters and logs
    # 定义文件名称
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d%H%M%S")
    file_name = '{0}_clients_{1}_frac_{2}_lr_{3}_comm_rounds_{4}_epoch{5}_model{6}_dataset_{7}_time{8}_{9}_{10}_spilt{11}.txt'. \
        format(
        args.algo,
        args.num_users,
        args.frac,
        args.lr,
        args.epochs,
        args.local_ep,
        args.model,
        args.dataset,
        timestamp,
        kd_info,
        dp_info,
        args.split_ratio
    )
    return file_name

def create_dynamic_folder(args):
    if args.iid=='p':
        dataset_info=f"ncls{args.overlapping_classes}"
    if args.iid=='dir':
        dataset_info = f"dir{args.dirichlet_alpha}"
    folder_name='{0}/{1}/{2}/{3}'.format(args.algo,args.model,args.dataset,dataset_info)
    folder_path=os.path.join(f"./results/",folder_name)
    return folder_path

def create_dynamic_DP_folder(args):
    if args.iid=='p':
        dataset_info=f"ncls{args.overlapping_classes}"
    if args.iid=='dir':
        dataset_info = f"dir{args.dirichlet_alpha}"
    folder_name='{0}/{1}/{2}/{3}'.format(args.algo,'DP',args.dataset,dataset_info)
    DP_folder_path=os.path.join(f"./results/",folder_name)
    return DP_folder_path

def create_dynamic_norm_folder(args):
    folder_name='{0}/{1}'.format(args.algo,'norm')
    norm_folder_path=os.path.join(f"./results/",folder_name)
    return norm_folder_path

def create_dynamic_RL_folder(args):
    folder_name='{0}/{1}'.format(args.algo,'RL')
    RL_folder_path=os.path.join(f"./results/",folder_name)
    return RL_folder_path

def create_dynamic_norm_folder(args):
    folder_name='{0}/{1}'.format(args.algo,'norm')
    norm_folder_path=os.path.join(f"./results/",folder_name)
    return norm_folder_path

def create_dynamic_personalmodel_folder(args):
    if args.iid=='p':
        dataset_info=f"ncls{args.overlapping_classes}"
    if args.iid=='dir':
        dataset_info = f"dir{args.dirichlet_alpha}"
    folder_name='{0}/{1}/{2}/{3}/{4}'.format(args.algo,args.model,args.dataset,dataset_info,'Personal')
    norm_folder_path=os.path.join(f"./results/",folder_name)
    return norm_folder_path
