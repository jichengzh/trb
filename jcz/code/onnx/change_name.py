import os

# 指定需要重命名的目录所在的父文件夹路径
folder_path = '/home/featurize/Bench2Drive/1_eval_bench2drive220_3_vad_traj'  # 请替换为你的实际路径
prefix_to_remove = '1_jcz_bench2drive220_uniad_speedlimit_2_3_vad_traj_'

# 遍历该文件夹中的所有内容
for name in os.listdir(folder_path):
    old_path = os.path.join(folder_path, name)
    if os.path.isdir(old_path) and name.startswith(prefix_to_remove):
        new_name = name[len(prefix_to_remove):]  # 去掉前缀
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f'Renamed: {name} -> {new_name}')