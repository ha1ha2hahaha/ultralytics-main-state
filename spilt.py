import os
import random
import shutil

# 将文件分为 train 0.8 val 0.2

# 定义源文件夹、训练集文件夹和验证集文件夹路径
"""
src_folder = "/media/syb310/ubuntudata/Code/ultralytics-main/dataset/2113"
train_folder = "/media/syb310/ubuntudata/Code/ultralytics-main/dataset/images/train"
val_folder = "/media/syb310/ubuntudata/Code/ultralytics-main/dataset/images/val"
"""
print("开始spilt。。。。。。。。。。。。。。。。")
src_folder = "/home/data/2113"
train_folder = "/home/data/images/train"
val_folder = "/home/data/images/val"

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(val_folder):
    os.makedirs(val_folder)
# 获取源文件夹中所有jpg文件路径
jpg_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith(".jpg")]

# 随机打乱jpg文件顺序
random.shuffle(jpg_files)

# 计算训练集和验证集文件个数
num_train_files = int(len(jpg_files) * 0.9)
num_val_files = len(jpg_files) - num_train_files

# 遍历每个jpg文件
for i, jpg_file in enumerate(jpg_files):
    # 获取对应的json文件路径
    json_file = os.path.splitext(jpg_file)[0] + ".json"

    # 判断json文件是否存在
    if os.path.exists(json_file):
        # 如果存在，则将jpg和json文件分别拷贝到train或val文件夹下
        if i < num_train_files:
            train_jpg_file = os.path.join(train_folder, os.path.basename(jpg_file))
            train_json_file = os.path.join(train_folder, os.path.basename(json_file))
            shutil.copy(jpg_file, train_jpg_file)
            shutil.copy(json_file, train_json_file)
        else:
            val_jpg_file = os.path.join(val_folder, os.path.basename(jpg_file))
            val_json_file = os.path.join(val_folder, os.path.basename(json_file))
            shutil.copy(jpg_file, val_jpg_file)
            shutil.copy(json_file, val_json_file)