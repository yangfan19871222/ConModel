#!/usr/bin/python
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random
import os
import logging
logging.basicConfig(level = logging.DEBUG)


def get_net():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name = 'conv1', dilate = (1, 1))
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=64, name = 'conv2', dilate = (1, 1))
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=64, name = 'conv3', dilate = (1, 1))
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 128, name = 'conv4', dilate = (1, 1))
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    
    relu4 = mx.symbol.Flatten(data = relu4)
    
    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 256, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
 #   relu4 = mx.symbol.Dropout(data = relu4, p = 0.5, name = 'dropout1')

    conv6_1 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_1')

    loss1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax')
    return loss1

if __name__ == '__main__':
    batch_size = 128
    _, arg_params, __ = mx.model.load_checkpoint("mtcnn", 2)
    #_, arg_params, __ = mx.model.load_checkpoint("./org_model/det3", 1)
    data_shape = [("data", (batch_size, 3, 48, 48))]
    input_shapes = dict(data_shape)
    sym = get_net()
    executor = sym.simple_bind(ctx = mx.gpu(6), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])

    root_path = '/media/disk1/yangfan/wider_faces/mtcnn_data/'
    imglist = os.path.join(root_path, 'imglist_stage3/val.txt')
    f = open(imglist)
    
    test_begin = 00
    test_count = 25600
    for i in range(test_begin):
        f.readline()

    count = 0
    err_count = 0
    face_count = 0
    face_recall_count = 0
    for i in range(test_count / batch_size):
        if i % 100 == 0:
            print 'processing batch:' +  str(i)
        imgs = []
        labels = []
        j = 0
        while j < batch_size:
            line = f.readline()
            ll = line.split(' ')
        #    if 'part' in ll[0]:
        #        continue
            cur_file = os.path.join(root_path, 'images_stage3/', ll[0])
            image = cv2.imread(cur_file)
            if image == None:
                continue
            if image.shape[1] < 24 or image.shape[0] < 24:
                continue
            image = cv2.resize(image, (48, 48))
            image = image.transpose((2, 0, 1))
            image = np.multiply(image - 127.5, 1.0 / 127.5)
            imgs.append(image)
            ll[1] == ll[1].strip('\n')
            labels.append(ll[1])
            if ll[1] == '1':
                face_count += 1
            j += 1

        executor.forward(is_train = False, data = mx.nd.array(imgs))
        probs = executor.outputs[0].asnumpy()
        
        for k in range(batch_size):
            if int(labels[k]) != np.argmax(probs[k]) and int(labels[k]) == 0:
               err_count += 1
            
            if np.argmax(probs[k]) == 1:
                count += 1
            
            if int(labels[k]) == 1 and np.argmax(probs[k]) == 1:
                
             #  print probs[k]
               face_recall_count += 1
                

    print count
    print err_count
    print float(err_count) / float(count)

    print face_count
    print face_recall_count
    print float(face_recall_count) / float(face_count)
               
       # result =  np.argmax(probs[i][41:65])+41
