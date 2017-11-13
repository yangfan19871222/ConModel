#!/usr/bin/python
import os, sys
import numpy as np
import copy

cur_path = os.path.dirname(__file__)
model_path = os.path.join(cur_path, '../')
sys.path.insert(0, model_path)
cur_path = os.path.dirname(__file__)
model_path = os.path.join(cur_path, '../python')
sys.path.insert(0, model_path)

cv2_path = '/media/disk1/yangfan/opencv-2.4.13.2/lib'
sys.path.insert(0, cv2_path)
import cv2


import mxnet as mx
from mtcnn_detector import MtcnnDetector
from time import time
from nms.gpu_nms import *

from config import GPU_ID

if True:
    boxes = np.zeros((10, 5))
    boxes = boxes.astype('float32')
    pick = gpu_nms(boxes, float(0.7), GPU_ID)
    threshold = [0.5, 0.5, 0.6]
    ctx = mx.gpu(GPU_ID)
    detector = MtcnnDetector(model_folder='model', ctx=ctx, num_worker = 20, threshold = threshold, accurate_landmark = True, minsize = 40)
    detect_face = detector.detect_face
    age_dict = {}
    age_dict[0] = [0, 2]
    age_dict[1] = [3, 7]
    age_dict[2] = [8, 13]
    age_dict[3] = [14, 18]
    age_dict[4] = [19, 25]
    age_dict[5] = [26, 30]
    age_dict[6] = [31, 35]
    age_dict[7] = [36, 100]

def detect_face_impl(image):
#    start_time = time()
    if image == None:
        return []
    try:
        image = np.asarray(bytearray(image), dtype=np.uint8)
        #cv2.IMREAD_COLOR
        #image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_COLOR)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception, e:
        print str(e)
        print 'decode error!'
        return []
    
    image_width_org = image.shape[1]
    image_height_org = image.shape[0]
    img_tmp = image
   # start_time = time()
    image_width = 500
    image_height = 500
    scale = min(float(image_height) / float(img_tmp.shape[0]), float(image_width) / float(img_tmp.shape[1]))
       # print min(image_width, img_tmp.shape[1] * scale)
       # print min(image_height, img_tmp.shape[0] * scale)
        #while img_tmp.shape[0] > image_height or img_tmp.shape[1] > image_width:
    img_tmp = cv2.resize(img_tmp, (int(min(image_width, img_tmp.shape[1] * scale)), int(min(image_height, img_tmp.shape[0] * scale))))
        
       # print img_tmp.shape 
    img_tmp_shape_before_pad = img_tmp.shape
    img_tmp = np.lib.pad(img_tmp, ((0, image_height - img_tmp.shape[0]), (0, image_width - img_tmp.shape[1]), (0, 0)), 'constant')

       # print img_tmp.shape
        
    results, points = detect_face(img_tmp)

    max_index = 0
    max_value = 0
    if results is None:
        return []
        
    return_value = []
    for i in range(len(results)):
        point1 = results[i][0] * 1.0 / scale
        point2 = results[i][1] * 1.0 / scale
        width = (results[i][2] - results[i][0]) * 1.0 /scale
        height = (results[i][3] - results[i][1]) * 1.0 /scale
        if point1 > image_width_org or point2 > image_height_org:
             continue
        if point1 + width > image_width_org:
             width = image_width_org - point1
        if point2 + height > image_height_org:
             height = image_height_org - point2
        
        area = int(width) * int(height)
        try:
            ratio = float(area) / (image_width_org * image_height_org)
        except Exception,e :
            ratio = 0
        return_value.append([int(point1), int(point2), int(width), int(height), area, ratio, results[i][-12], results[i][-11], results[i][-10], int(results[i][-8]), results[i][-6], results[i][-5], results[i][-4], results[i][-3], results[i][-2], results[i][-1]])
        
        return_value[-1][-7] = copy.deepcopy(age_dict[return_value[-1][-7]])
        return_value[-1][-7].append(results[i][-9])
        return_value[-1][-7].append(results[i][-7])

    return return_value

if __name__ == '__main__':
    f = open('test1.jpg', 'rb')
    #out = open('test1_result.jpg', 'wb')
    image = f.read()
    img = cv2.imread('test1.jpg')
    result = detect_face_impl(image) 
    print result
    if len(result) > 0:
        for i in range(len(result)):
            x1 = result[i][0]
            y1 = result[i][1]
            width = result[i][2]
            height = result[i][3]
            female = result[i][4]
            beauty = result[i][5]
            smile = result[i][6]
            age = result[i][9][0]
            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (255, 255, 255))

            cv2.putText(img, str('FeMale%.4f'%(female)), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 
            cv2.putText(img, str('beauty%.4f'%(beauty)), (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 
            cv2.putText(img, str('smile%.4f'%(smile)), (x1, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 
            cv2.putText(img, str('age%.4f'%(age)), (x1, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 
    cv2.imwrite('test1_result.jpg', img)
