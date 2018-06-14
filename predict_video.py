from yolov3 import make_seq_yolov3_model
from utils.utils import WeightReader
from utils.utils import preprocess_input
from utils.utils import decode_netout
from utils.utils import correct_yolo_boxes
from utils.utils import do_nms
from utils.bbox import draw_boxes
from utils.utils import get_yolo_boxes

import numpy as np
import cv2
import os

net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

infer_model, seq_infer_model = make_seq_yolov3_model()

if not (os.path.exists('Weights/yolov3.h5')):
    weight_reader = WeightReader('yolov3.weights')
    weight_reader.load_weights(infer_model)
    infer_model.save_weights('Weights/yolov3.h5')
else :
    infer_model.load_weights('Weights/yolov3.h5',by_name=True)
    seq_infer_model.load_weights('Weights/yolov3.h5', by_name=True)


imgs_folder_path = 'Z:/dataset/KNU-Campus Dataset/images/20180312_172240/'
img_name = '20180312_172240_'

prev_feature=[]

isFirst=True
for img_num in range(50):
    img_num=str(img_num)
    while(len(img_num)<4):
        img_num='0'+img_num

    img = cv2.imread(imgs_folder_path+img_name+img_num+'.jpg')
    image_h, image_w, _ = img.shape
    
    process_image = preprocess_input(img, net_h, net_w)

    #yolos = infer_model.predict(process_image)

    if isFirst:
        yolos = infer_model.predict(process_image)
        prev_feature=yolos[3:]
        yolos=yolos[:3]
        isFirst=False
    else:
        yolos = seq_infer_model.predict([process_image,prev_feature[0],prev_feature[1],prev_feature[2]])
        prev_feature=yolos[3:]
        yolos=yolos[:3]


    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        yolo_anchors = anchors[(2 - i) * 6:(3 - i) * 6]
        boxes += decode_netout(yolos[i][0], yolo_anchors, obj_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)

    # draw bounding boxes on the image using labels
    draw_boxes(img, boxes, labels, obj_thresh)

    cv2.imwrite('outputs/'+img_name+img_num+'_seq_detected.jpg',cv2.resize(img,(1280,720)))
    cv2.imshow('video with bboxes', cv2.resize(img,(1280,720)))
    cv2.waitKey(9)