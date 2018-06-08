from yolov3 import make_yolov3_model
from utils.utils import WeightReader
from utils.utils import preprocess_input
from utils.utils import decode_netout
from utils.utils import decode_netout2
from utils.utils import decode_yolo
from utils.utils import correct_yolo_boxes
from utils.utils import do_nms
from utils.bbox import draw_boxes

import cv2
import os

net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
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

infer_model = make_yolov3_model()

if not (os.path.exists('Weights/yolov3.h5')):
    weight_reader = WeightReader('yolov3.weights')
    weight_reader.load_weights(infer_model)
    infer_model.save_weights('Weights/yolov3.h5')
else :
    infer_model.load_weights('Weights/yolov3.h5')

imgs_folder_path = 'Z:/dataset/KNU-Campus Dataset/images/20180312_171706/'
img_name = '20180312_171706_'

yolos_pred=[]
count=0
for img_num in range(20):
    img_num=str(img_num)
    while(len(img_num)<4):
        img_num='0'+img_num
    img = cv2.imread(imgs_folder_path+img_name+img_num+'.jpg')
    image_h, image_w, _ = img.shape
    new_image = preprocess_input(img, net_h, net_w)

    yolos = infer_model.predict(new_image)
    #yolos_pred.append(yolos)

    decode_yolos = [decode_yolo(yolos[0][0],obj_thresh),decode_yolo(yolos[1][0],obj_thresh),decode_yolo(yolos[2][0],obj_thresh)]


    # else:
    #     yolos_pred_sub = [yolos_pred[count-1][0]-yolos_pred[count-2][0],yolos_pred[count-1][1]-yolos_pred[count-2][1],yolos_pred[count-1][2]-yolos_pred[count-2][2]]
    #     yolos = [yolos_pred[count-1][0]+ yolos_pred_sub[0], yolos_pred[count-1][1] +yolos_pred_sub[1], yolos_pred[count-1][2] + yolos_pred_sub[2]]
    #     yolos_pred.append(yolos)


    boxes = []

    for i in range(len(decode_yolos)):
        # decode the output of the network
        boxes += decode_netout2(decode_yolos[i], anchors[i], obj_thresh, net_h, net_w)
        #boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, net_h, net_w)
    print(boxes)
    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)

    # draw bounding boxes on the image using labels
    draw_boxes(img, boxes, labels, obj_thresh)

    # write the image with bounding boxes to file
    cv2.imwrite('outputs/'+img_name+ img_num + '_detected.jpg', (img).astype('uint8'))
    count+=1