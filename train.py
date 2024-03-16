from ultralytics import YOLO


"""
每组标签首位是0，代表人类类别，这里只有person一个类别。往后数4位是边界框坐标，再往后面的17*3位是关键点信息。每个关键点由x，y，v组成，v代表该点是否可见。
一组标注信息共(1+4+17*3)=56个数字。即每一行都有56个数字标注。
"""
import os
if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Load a model
    print("start train")
    model = YOLO('E:\\pythonFile\\ultralytics-main-state\\ultralytics\\models\\v8\\yolov8-state.yaml').load("yolov8n-pose.pt")
    # model = YOLO('ultralytics/models/v8/yolov8-sit.yaml')
    # model = YOLO('ultralytics/models/v8/yolov8-pose.yaml').load("yolov8n-pose.pt")
    # Train the model
    model.train(data='E:\\pythonFile\\ultralytics-main-state\\ultralytics\\datasets\\coco8-state.yaml', epochs=2, imgsz=640)
    # Predict with the model
    # results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    print("end train")

    """
    # Load a model
    #model = YOLO('/media/syb310/ubuntudata/Code/ultralytics-main/ultralytics/models/v8/yolov8-sit.yaml').load("/media/syb310/ubuntudata/Code/ultralytics-main/yolov8n-pose.pt")
    model = YOLO('/media/syb310/ubuntudata/Code/ultralytics-main/ultralytics/models/v8/yolov8-state.yaml').load("/media/syb310/ubuntudata/Code/ultralytics-main/yolov8n-pose.pt")
    # model = YOLO('ultralytics/models/v8/yolov8.yaml').load("yolov8n.pt")
    # Train the model
    model.train(data='coco8-state.yaml', epochs=20, imgsz=640)
    # Predict with the model
    # results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    """
