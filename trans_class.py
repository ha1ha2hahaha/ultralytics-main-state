# COCO 格式的数据集转化为 YOLO 格式的数据集
# --json_path 输入的json文件路径
# --save_path 保存的文件夹名字，默认为当前目录下的labels。

import os
import json
from tqdm import tqdm
import argparse

# name = "train"
parser = argparse.ArgumentParser()
# 这里根据自己的json文件位置，换成自己的就行
parser.add_argument('--json_path',
                    default=r'E:\pythonFile/dataset/annotations_trainval2017/annotations/person_keypoints_''2017.json', type=str,
                    help="input: coco format(json)")
# 这里设置.txt文件保存位置
parser.add_argument('--save_path', default=r'E:/pythonFile/dataset/labels/keypoint_''_2017', type=str,
                    help="specify where to save the output dir of labels")
arg = parser.parse_args()

name = "val"


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)


if __name__ == '__main__':
    #json_file = arg.json_path  # COCO Object Instance 类型的标注
    # name = "train"
    test = []
    json_file = "E:/pythonFile/ultralytics-main/datasets/data/"+name+".json"
    # ana_txt_save_path = arg.save_path  # 保存的路径
    ana_txt_save_path = "E:/pythonFile/ultralytics-main/datasets/data/labels/"+name

    with open(json_file, "r", encoding="utf-8") as f:
        datas = json.load(f)
    #data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {"stand":0, "sit": 1, "crouch": 2, "prostrate_sleep": 3, "sit_sleep": 4, "lie_sleep": 5}  # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for data in tqdm(datas):
            filename = data['imagePath']
            filename = filename[7:-4]
            # head, tail = os.path.splitext(filename)
            ana_txt_name = filename + ".txt"
            f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
            eyes = []
            width = data['imageWidth']
            height = data['imageHeight']
            for i, shape in enumerate(data['shapes']):
                # 两个眼睛
                #eye = []
                for point in shape['points']:
                    eyes.append(point)
                #eyes.append(eye)
            x1_center = (eyes[0][0]+eyes[1][0])/2
            x2_center = (eyes[2][0]+eyes[3][0])/2
            y1_center = (eyes[0][1]+eyes[1][1])/2
            y2_center = (eyes[2][1]+eyes[3][1])/2
            """
            width1 = eyes[1][0]-eyes[0][0]
            width2 = eyes[3][0]-eyes[2][0]
            high1 = eyes[1][1]-eyes[0][1]
            high2 = eyes[3][1]-eyes[2][1]
            """

            x1_center = x1_center/width
            y1_center = y1_center/height
            width1 = (eyes[1][0]-eyes[0][0])/width
            high1 = (eyes[3][0]-eyes[2][0])/height
            x2_center = x2_center/width
            y2_center = y2_center/height
            width2 = (eyes[1][1]-eyes[0][1])/width
            high2 = (eyes[3][1]-eyes[2][1])/height
            f_txt.write("%s %s %s %s %s" % (0, x1_center, y1_center, width1, high1))
            f_txt.write("\n")
            f_txt.write("%s %s %s %s %s" % (0, x2_center, y2_center, width2, high2))
            f_txt.close()
"""
            for ann in data['annotations']:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s" % (id_map[ann["category_name"]], box[0], box[1], box[2], box[3]))
                counter = 0
                for i in range(len(ann["keypoints"])):
                    if ann["keypoints"][i] == 2 or ann["keypoints"][i] == 1 or ann["keypoints"][i] == 0:
                        f_txt.write(" %s " % format(ann["keypoints"][i],'6f'))
                        counter = 0
                    else:
                        if counter == 0:
                            f_txt.write(" %s " % round((ann["keypoints"][i] / img_width), 6))
                        else:
                            f_txt.write(" %s " % round((ann["keypoints"][i] / img_height), 6))
                        counter+=1
            f_txt.write("\n")
            f_txt.close()
        test[0]
                # 将图片的路径写入train2017或val2017的路径
            # list_file.write('E:/pythonFile/dataset/2113/images/'+name+'/%s.jpg\n' % (head))
            # list_file.close()


"""



"""


if __name__ == '__main__':
    #json_file = arg.json_path  # COCO Object Instance 类型的标注
    name = "train"
    json_file = "E:/pythonFile/dataset/2113/train/"+name+".json"
    # ana_txt_save_path = arg.save_path  # 保存的路径
    ana_txt_save_path = "E:/pythonFile/dataset/2113/labels/"+name

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    #data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    # id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    # print(id_map)
    # 这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(ana_txt_save_path, 'train2017.txt'), 'w')
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
                counter=0
                for i in range(len(ann["keypoints"])):
                    if ann["keypoints"][i] == 2 or ann["keypoints"][i] == 1 or ann["keypoints"][i] == 0:
                        f_txt.write(" %s " % format(ann["keypoints"][i],'6f'))
                        counter=0
                    else:
                        if counter == 0:
                            f_txt.write(" %s " % round((ann["keypoints"][i] / img_width),6))
                        else:
                            f_txt.write(" %s " % round((ann["keypoints"][i] / img_height),6))
                        counter+=1
        f_txt.write("\n")
        f_txt.close()
        # 将图片的路径写入train2017或val2017的路径
        list_file.write('E:/pythonFile/dataset/images/'+name+'2017/%s.jpg\n' % (head))
    list_file.close()













import json
import os
from tqdm import tqdm
# COCO 2017 annotations file path
# coco_annotations_path = "E:\pythonFile/dataset/annotations_trainval2017/annotations/person_keypoints_train2017.json"
# Path to write YOLOv8 annotations file
# yolo_annotations_path = "E:\pythonFile/dataset/labels/person_keypoints_train2017.txt"

# COCO格式的输入文件路径
name = "val"
input_path = "E:\pythonFile/dataset/annotations_trainval2017/annotations/person_keypoints_"+name+"2017.json"

# YOLO格式的输出文件路径
output_path = "E:/pythonFile/dataset/labels/person_keypoints_"+name+"2017.txt"
coco_folder = "E:/pythonFile/dataset/annotations_trainval2017/annotations/"
yolov8_folder = "E:/pythonFile/dataset/labels/keypoint_"+name+"_2017"
json_file = os.path.join(coco_folder, 'person_keypoints_'+name+'2017.json')

with open(json_file, 'r') as f:
    annotations = json.load(f)
for image in tqdm(annotations['images']):
    # 获取图像信息
    width = image['width']
    height = image['height']
    image_id = image['id']
    file_name = image['file_name']

    # 待写入的 YOLOv8 标注文件路径
    txt_path = os.path.join(yolov8_folder, f'{os.path.splitext(file_name)[0]}.txt')

    # 遍历该图像对应的所有实例，提取位置和关键点信息
    instances = [instance for instance in annotations['annotations'] if instance['image_id']==image_id and instance['category_id']==1]  # 只提取人体实例
    for instance in instances:
        keypoints = instance['keypoints']
        bbox = instance['bbox']

        # 计算目标框位置和大小，并将关键点坐标转化为相对于目标框左上角的归一化坐标
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        cx /= width
        cy /= height
        w /= width
        h /= height

        kps_x = keypoints[0::3]
        kps_y = keypoints[1::3]
        kps_v = keypoints[2::3]
        kps_rel_x = [(x - bbox[0]) / bbox[2] for x, v in zip(kps_x, kps_v) if v > 0]
        kps_rel_y = [(y - bbox[1]) / bbox[3] for y, v in zip(kps_y, kps_v) if v > 0]

        # 将标注信息写入 YOLOv8 数据集标注文件中
        with open(txt_path, 'a') as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} ")
            f.write(" ".join([f"{kps_rel_x[i]:.6f} {kps_rel_y[i]:.6f}" for i in range(len(kps_rel_x))]) + "\n")
"""