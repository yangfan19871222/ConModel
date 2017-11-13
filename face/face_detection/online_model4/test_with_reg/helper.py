# coding: utf-8
# YuanYang
import math
import cv2
import numpy as np
import mxnet as mx
from time import time
from nms.gpu_nms import *
has_reg = True

def nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    return pick

def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = out_data.transpose((2,0,1))
    out_data = np.expand_dims(out_data, 0)
    out_data = (out_data - 127.5)*0.0078125
    #out_data = out_data * 0.5 * 0.0078125
    return out_data

def generate_bbox(map, reg, scale, threshold):
     """
         generate bbox from feature map
     Parameters:
     ----------
         map: numpy array , n x m x 1
             detect score for each position
         reg: numpy array , n x m x 4
             bbox
         scale: float number
             scale of this detection
         threshold: float number
             detect threshold
     Returns:
     -------
         bbox array
     """
     stride = 2
     cellsize = 12

     t_index = np.where(map>threshold)

     # find nothing
     if t_index[0].size == 0:
         return np.array([])

     if has_reg == True:
        dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = map[t_index[0], t_index[1]]
   #  print score.shape
        boundingbox = np.vstack([np.round((stride*t_index[1]+1)/scale),
                              np.round((stride*t_index[0]+1)/scale),
                              np.round((stride*t_index[1]+1+cellsize)/scale),
                              np.round((stride*t_index[0]+1+cellsize)/scale),
                              score,
                              reg])

        return boundingbox.T
     else:
        score = map[t_index[0], t_index[1]]
   #  print score.shape
        boundingbox = np.vstack([np.round((stride*t_index[1]+1)/scale),
                              np.round((stride*t_index[0]+1)/scale),
                              np.round((stride*t_index[1]+1+cellsize)/scale),
                              np.round((stride*t_index[0]+1+cellsize)/scale),
                              score])

        return boundingbox.T

real_executors = []
real_scales = []

def init_executor(scales, executors):
    for i in range(len(scales)):
        real_executors.append(executors[i])

        real_scales.append(scales[i])

def detect_first_stage(img, index, threshold, ctx):
#    return None
    """
        run PNet for first stage
    
    Parameters:
    ----------
        img: numpy array, bgr order
            input image
        scale: float number
            how much should the input image scale
        net: PNet
            worker
    Returns:
    -------
        total_boxes : bboxes
    """
 #   print index
    scale = real_scales[index]
    height, width, _ = img.shape
    hs = int(height * scale)
    ws = int(width * scale)
    
   # img = mx.nd.array(img)
   # im_data = mx.image.imresize(img, hs, ws)
    im_data = cv2.resize(img, (ws,hs))
    
    # adjust for the network input
    input_buf = adjust_input(im_data)
#    print 'prepare data:%.4f'%(end_time - start_time)
   # print input_buf.shape
  #  output = net.predict(input_buf)
    
    #net.forward(data = mx.nd.array(input_buf))
    
   # start_time = time() 
   # data_shape = [("data", input_buf.shape)]
   # input_shapes = dict(data_shape)
   # executor = net.simple_bind(ctx = ctx, **input_shapes)
   # for key in executor.arg_dict.keys():
   #     if key in arg_params:
   #         arg_params[key].copyto(executor.arg_dict[key])


    #root_path = '/media/disk1/yangfan/wider_faces/mtcnn_data/'

  #  end_time = time()
  #  print 'binding parameters: %.2f'%(end_time - start_time)
    
   # start_time = time()
   # data_shape = [("data", input_buf.shape)]
   # input_shapes = dict(data_shape)
   # executor = executor.reshape(allow_up_sizing = True, **input_shapes)
   # end_time = time()

    #print 'reshape time %.4f'%(end_time - start_time)
    
    real_executors[index].forward(is_train = False, data = input_buf)
    output = real_executors[index].outputs[0].asnumpy()
    if has_reg == True:
        reg = real_executors[index].outputs[1].asnumpy()
  #  print 'test1'
  #  print output.shape
  #  print 'scale:%.2f, time:%.4f'%(scale, end_time - start_time)
    output_hs = ((hs - 2) / 2) - 2 - 2
    output_ws = ((ws - 2) / 2) - 2 - 2

  #  print output_hs
  #  print output_ws
  #  for i in range(output.shape[1]):
  #      for j in range(output.shape[2]):
  #          for k in range(output.shape[3]):
  #              if output[0][i][j][k] > 0.9:
  #                  print '%d, %d, %d' %(i, j, k)
    #result =  np.where(output[0] > 0.9)
    #result[0]
    #output = np.transpose(output, (0, 3, 1, 2))
   # output = output.reshape((1, output_hs, output_ws, 2))
  #  print output[0, 1, :, :]
    if has_reg == True:
        boxes = generate_bbox(output[0][1,:,:], reg, scale, threshold)
    else:
        boxes = generate_bbox(output[0][1,:,:], output[0], scale, threshold)
#    print 'generated bbox: %d'%(len(boxes))
    if boxes.size == 0:
        return None
  #  print 'test2'
    # nms
    #print 'generating box time: %.4f'%(end_time - start_time)
    #print 'generating box:%d'%(boxes.shape[0])
    #pick = nms(boxes[:,0:5], 0.5, mode='Union')
   # boxes.dtype = 'float32'
    boxes = boxes.astype('float32')
    pick = gpu_nms(boxes[:, 0:5], 0.5, 5)
    #print pick
 #   print 'nms:' + str(len(pick))
    boxes = boxes[pick]
    #print 'nms time: %.4f'%(end_time - start_time)
    return boxes

def detect_first_stage_warpper( args ):
    return detect_first_stage(*args)
