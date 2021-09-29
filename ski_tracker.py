from models import *
from utils import utils

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import cv2


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def clc_center_bbox(bbox):
    (x1, y1, box_w, box_h) = bbox
    x_center = int(x1 + box_w/2)
    y_center = int(y1 + box_h/2)
    return (x_center, y_center)


def show_frame(frame,found, bbox):
    if found:
        (x1, y1, box_w, box_h) = bbox
        (x_center, y_center) = clc_center_bbox(bbox)
        cv2.circle(frame, (x_center, y_center), 2, (128,0,128), 4)
        cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), (0,0,255), 4)
        # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
        # cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        exit(0)


def make_bbox(x1, y1, x2, y2, img_h, img_w, unpad_h, unpad_w):
    box_h = int(((y2 - y1) / unpad_h) * img_h)
    box_w = int(((x2 - x1) / unpad_w) * img_w)
    y1 = int(((y1 - pad_y // 2) / unpad_h) * img_h)
    x1 = int(((x1 - pad_x // 2) / unpad_w) * img_w)

    bbox = (x1, y1, box_w, box_h)
    return bbox



# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
output_ratio = 2

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
# model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor  # add cuda for GPU

videopath = 'videos/ski1.mp4'
vid = cv2.VideoCapture(videopath)

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Stream', (800,600))

ret,frame=vid.read()
if not ret:
    raise("couldn't read video")
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)

frames = 0
starttime = time.time()
gap2detect = 10

img_h = vh
img_w = vw
out_h = int(img_h/output_ratio)
out_w = int(img_w/output_ratio)
pad_x = max(img_h - img_w, 0) * (img_size / max(frame.shape))
pad_y = max(img_w - img_h, 0) * (img_size / max(frame.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_x

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(out_w,out_w))



bbox = None
InTrack = False
while(True): # go over all video
    found = False
    ret, frame = vid.read()
    #TODO end of video
    if not ret:
        raise ("couldn't read video")
    frames += 1
    print("frame: ", frames)
    pilimg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if InTrack:
        ok, bbox = tracker.update(frame)
        if ok:
            found = True
            print("on track !")
        else:
            InTrack = False

    if found == False:
        detections = detect_image(pilimg)
        if detections is None:
            print("couldn't find anything in image")
            #  bbox dont change
        else:
            detections_arr = detections.numpy()
            # go over all detection and stop at the first person
            for x1, y1, x2, y2, obj_id, cls_pred, cls_idx in detections_arr:
                if cls_idx != 0:
                    print("found a target, not person, ignored")
                    continue
                print("found a person")
                found = True
                bbox = make_bbox(x1, y1, x2, y2, img_h, img_w, unpad_h, unpad_w)
                (x_center, y_center) = clc_center_bbox(bbox)
                InTrack = True
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, bbox)
                break  # one person is enough

    show_frame(frame,found, bbox)

    outvideo.write(frame[0:out_w, 0:out_w])#y1:y2, x1:x2])
    if frames == 100:
        break
totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
outvideo.release()
cv2.destroyAllWindows()
