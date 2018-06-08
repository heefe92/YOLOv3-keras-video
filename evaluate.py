import os
from voc import parse_voc_annotation
from yolov3 import make_yolov3_model
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from utils.utils import WeightReader

model_labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
###############################
#   Create the validation generator
###############################
valid_ints, labels = parse_voc_annotation(
    'F:/DataSet/coco/Annotations/',
    'F:/DataSet/coco/JPEGImages/',
    'coco.pkl',
    model_labels
)
labels = labels.keys() if len(model_labels) == 0 else model_labels
labels = sorted(labels)

valid_generator = BatchGenerator(
    instances=valid_ints,
    anchors=anchors,
    labels=model_labels,
    downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
    max_box_per_image=0,
    batch_size=2,
    min_net_size=416,
    max_net_size=416,
    shuffle=False,
    jitter=0.0,
    norm=normalize
)
###############################
#   Load the model and do evaluation
###############################

infer_model = make_yolov3_model()

if not (os.path.exists('Weights/yolov3.h5')):
    weight_reader = WeightReader('yolov3.weights')
    weight_reader.load_weights(infer_model)
    infer_model.save_weights('Weights/yolov3.h5')
else :
    infer_model.load_weights('Weights/yolov3.h5')
# compute mAP for all the classes
average_precisions = evaluate(infer_model, valid_generator)
# print the score
for label, average_precision in average_precisions.items():
    print(labels[label] + ': {:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
