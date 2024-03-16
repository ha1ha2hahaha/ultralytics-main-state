import json
import os

"""
home
    data
        2113

        images

        labels

        train.json
        val.json
"""

name = "train"
print("开始merge。。。。。。。。。。。。。。。。")
train_path = "/home/data/images/train"
val_path = "/home/data/images/val"
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)
train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith(".json")]

res = []
for js in train_files:
    with open(js, "r", encoding="utf-8") as f:
        data = json.load(f)
        filename = os.path.basename(js)
        img = data['images']
        img[0]['file_name'] = filename[:-5] + ".jpg"
        res.append(data)

    # 将merged_data字典中的数据输出到新的json文件中
with open("/home/data/train.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False)

val_files = [os.path.join(val_path, f) for f in os.listdir(val_path) if f.endswith(".json")]

res2 = []

for js in val_files:
    with open(js, "r", encoding="utf-8") as f:
        data = json.load(f)
        filename = os.path.basename(js)
        img = data['images']
        img[0]['file_name'] = filename[:-5] + ".jpg"
        res2.append(data)

    # 将merged_data字典中的数据输出到新的json文件中
with open("/home/data/val.json", "w", encoding="utf-8") as f:
    json.dump(res2, f, ensure_ascii=False)