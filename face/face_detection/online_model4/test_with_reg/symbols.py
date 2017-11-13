#!/usr/bin/python

import os,sys
import mxnet as mx
import random
import numpy as np

first_multi_task = True
second_multi_task = True

landmarks_task = True


def get_stage1_symbol():
    
    data = mx.symbol.Variable('data')
    
    if first_multi_task == True:
        outside_weights = mx.symbol.Variable('outside_weights')
        bbox_targets = mx.symbol.Variable('bbox_targets')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=10, name = 'conv1')
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu", name = 'PReLU1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                              kernel=(2, 2), stride=(2,2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=16, name = 'conv2')
    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu", name = 'PReLU2')

    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')
    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32, name = 'conv3')
    #relu3 = mx.symbol.Activation(data=conv3, act_type="relu", name = 'PReLU3')
    
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')
    
   # relu3 = mx.symbol.Flatten(data = relu3)
   # conv4_1 = mx.symbol.FullyConnected(data = relu3, num_hidden = 2)
    conv4_1 = mx.symbol.Convolution(data = relu3, kernel = (1, 1), num_filter = 2, name = 'conv4_1')
    output = mx.symbol.SoftmaxActivation(data = conv4_1, mode = 'channel', name = 'softmax')
    if first_multi_task == True:

        conv4_2 = mx.symbol.Convolution(data = relu3, kernel = (1, 1), num_filter = 4, name = 'conv4_2')
    

        return mx.symbol.Group([output, conv4_2])

    return output

def get_stage2_symbol():
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=28, name = 'conv1', dilate = (1, 1))
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=48, name = 'conv2', dilate = (1, 1))
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(2, 2), num_filter=64, name = 'conv3', dilate = (1, 1))
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')
    
    relu3 = mx.symbol.Flatten(data = relu3)
    
    conv4 = mx.symbol.FullyConnected(data = relu3, num_hidden = 128, name = 'conv4')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
   # relu4 = mx.symbol.Dropout(data = relu4, p = 0.5, name = 'dropout1')

    conv5_1 = mx.symbol.FullyConnected(data=relu4, num_hidden = 2, name = 'conv5_1')

    output = mx.symbol.SoftmaxActivation(data = conv5_1, name = 'softmax')

    if second_multi_task == True:
        conv5_2 = mx.symbol.FullyConnected(data=relu4, num_hidden = 4, name = 'conv5_2')
        
        mtcnn = mx.symbol.Group([output, conv5_2])
        
        return mtcnn

    return output
   # conv4_2 = mx.symbol.Convolution(data = relu3, kernel = (1, 1), num_filter = 4)

    #conv4_1 = mx.symbol.Reshape(data = conv4_1, target_shape = (0, ))
    
def get_stage3_symbol():
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

    output1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax')
        
    output2 = mx.symbol.FullyConnected(data=relu5, num_hidden = 4, name = 'conv6_2')
    
    if landmarks_task == True:
        output3 = mx.symbol.FullyConnected(data=relu5, num_hidden = 10, name = 'conv6_3')

        mtcnn = mx.symbol.Group([output1, output2, output3, relu2])
        
        return mtcnn

    return mtcnn



def get_stage4_symbol():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=16, name = 'conv1', dilate = (1, 1))
    #bn1 = mx.symbol.BatchNorm(data = conv1, name = 'batchnorm0')
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=32, name = 'conv2', dilate = (1, 1))
    #bn2 = mx.symbol.BatchNorm(data = conv2, name = 'batchnorm1')
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=32, name = 'conv3', dilate = (1, 1))
    #bn3 = mx.symbol.BatchNorm(data = conv3, name = 'batchnorm2')
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

#    conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 64, name = 'conv4', dilate = (1, 1))
    #bn4 = mx.symbol.BatchNorm(data = conv4, name = 'batchnorm3')
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
 #   relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    
    relu4 = mx.symbol.Flatten(data = pool3)
    
#    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 64, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
#    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
    #relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')

    conv6_1 = mx.symbol.FullyConnected(data=relu4, num_hidden = 2, name = 'conv6_1')

    output1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax1')
    
    conv6_2 = mx.symbol.FullyConnected(data=relu4, num_hidden = 2, name = 'conv6_2')
    
    output2 = mx.symbol.SoftmaxActivation(data=conv6_2,  name='softmax2')

    conv6_3 = mx.symbol.FullyConnected(data=relu4, num_hidden = 2, name = 'conv6_3')

    output3 = mx.symbol.SoftmaxActivation(data=conv6_3,  name='softmax3')

#    conv6_4 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_4')

#    output4 = mx.symbol.SoftmaxActivation(data=conv6_4,  name='softmax4')

    mtcnn = mx.symbol.Group([output1, output2, output3])

    return mtcnn

def get_vgg16_net():
    data = mx.symbol.Variable('data')
    #gender_label = mx.symbol.Variable('gender_label')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter = 64, name = 'conv1_1', dilate = (1, 1))
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(data=relu1, kernel=(3, 3), num_filter = 64, name = 'conv1_2', dilate = (1, 1))
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool1')
    # second conv
    conv3 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter = 128, name = 'conv2_1', dilate = (1, 1))
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), num_filter = 128, name = 'conv2_2', dilate = (1, 1))
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu4, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    #elu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    conv5 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter = 256, name = 'conv3_1', dilate = (1, 1))
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    conv6 = mx.symbol.Convolution(data=relu5, kernel=(3, 3), num_filter = 256, name = 'conv3_2', dilate = (1, 1))
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu")
    conv7 = mx.symbol.Convolution(data=relu6, kernel=(3, 3), num_filter = 256, name = 'conv3_3', dilate = (1, 1))
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu7, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv8 = mx.symbol.Convolution(data=pool3, kernel=(3, 3), num_filter = 512, name = 'conv4_1', dilate = (1, 1))
    relu8 = mx.symbol.Activation(data=conv8, act_type="relu")
    conv9 = mx.symbol.Convolution(data=relu8, kernel=(3, 3), num_filter = 512, name = 'conv4_2', dilate = (1, 1))
    relu9 = mx.symbol.Activation(data=conv9, act_type="relu")
    conv10 = mx.symbol.Convolution(data=relu9, kernel=(3, 3), num_filter = 512, name = 'conv4_3')
    relu10 = mx.symbol.Activation(data=conv10, act_type="relu")
    pool4 = mx.symbol.Pooling(data=relu10, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool4')

    conv11 = mx.symbol.Convolution(data=pool4, kernel=(3, 3), num_filter = 512, name = 'conv5_1')
    relu11 = mx.symbol.Activation(data=conv11, act_type="relu")
    conv12 = mx.symbol.Convolution(data=relu11, kernel=(3, 3), num_filter = 512, name = 'conv5_2')
    relu12 = mx.symbol.Activation(data=conv12, act_type="relu")
    conv13 = mx.symbol.Convolution(data=relu12, kernel=(3, 3), num_filter = 512, name = 'conv5_3')
    relu13 = mx.symbol.Activation(data=conv13, act_type="relu")
    pool5 = mx.symbol.Pooling(data=relu13, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool5')

    pool5 = mx.symbol.Flatten(data = pool5)
    
    fc14 = mx.symbol.FullyConnected(data = pool5, num_hidden = 4096, name = 'fc6_tmp')
    relu14 = mx.symbol.Activation(data=fc14, act_type="relu")
    relu14 = mx.symbol.Dropout(data = relu14, p = 0.5, name = 'dropout1')
   # relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')
    fc15 = mx.symbol.FullyConnected(data = relu14, num_hidden = 512, name = 'fc7_tmp')
    relu15 = mx.symbol.Activation(data=fc15, act_type="relu")
    #relu15 = mx.symbol.Dropout(data = relu15, p = 0.5, name = 'dropout2')
    fc16 = mx.symbol.FullyConnected(data=relu15, num_hidden = 2, name = 'fc8_101')
    fc16_1 = mx.symbol.FullyConnected(data=relu15, num_hidden = 2, name = 'fc8_101_1')
    fc16_2 = mx.symbol.FullyConnected(data=relu15, num_hidden = 2, name = 'fc8_101_2')

    output1 = mx.symbol.SoftmaxActivation(data=fc16,  name='softmax')
    output2 = mx.symbol.SoftmaxActivation(data=fc16_1,  name='softmax_tmp')
    output3 = mx.symbol.SoftmaxActivation(data=fc16_2,  name='softmax_tmp2')
    #loss1 = mx.symbol.SoftmaxOutput(data=fc16, label = gender_label, name='softmax', grad_scale = 1.0)

    #loss1 = mx.symbol.MakeLoss(data = loss1_, name = 'softmax_loss', grad_scale = 10.0, normalization = 'valid')
            
    mtcnn = mx.symbol.Group([output1, output2, output3])
    return mtcnn


def get_stage5_symbol():
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter = 16, name = 'conv1', dilate = (1, 1))
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter= 32, name = 'conv2', dilate = (1, 1))
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter = 32, name = 'conv3', dilate = (1, 1))
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 64, name = 'conv4', dilate = (1, 1))
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    relu4 = mx.symbol.Flatten(data = relu4)
    
    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 128, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
   # relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')

    conv6_1 = mx.symbol.FullyConnected(data=relu5, num_hidden = 8, name = 'conv6_1')
    output1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax1')

    mtcnn = mx.symbol.Group([output1])
    return mtcnn

def get_attractive_symbol():
    
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=64, name = 'conv1', dilate = (1, 1))
    #bn1 = mx.symbol.BatchNorm(data = conv1, name = 'batchnorm0')
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=128, name = 'conv2', dilate = (1, 1))
    #bn2 = mx.symbol.BatchNorm(data = conv2, name = 'batchnorm1')
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=128, name = 'conv3', dilate = (1, 1))
    #bn3 = mx.symbol.BatchNorm(data = conv3, name = 'batchnorm2')
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 256, name = 'conv4', dilate = (1, 1))
    #bn4 = mx.symbol.BatchNorm(data = conv4, name = 'batchnorm3')
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    
    relu4 = mx.symbol.Flatten(data = relu4)
    
    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 256, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
    #relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')

    conv6_1 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_1')

    output1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax1')
    
#    conv6_4 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_4')

#    output4 = mx.symbol.SoftmaxActivation(data=conv6_4,  name='softmax4')

    mtcnn = mx.symbol.Group([output1])
    
    return mtcnn

def get_attractive_small_symbol():
    
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=64, name = 'conv1', dilate = (1, 1))
    #bn1 = mx.symbol.BatchNorm(data = conv1, name = 'batchnorm0')
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=128, name = 'conv2', dilate = (1, 1))
    #bn2 = mx.symbol.BatchNorm(data = conv2, name = 'batchnorm1')
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=128, name = 'conv3', dilate = (1, 1))
    #bn3 = mx.symbol.BatchNorm(data = conv3, name = 'batchnorm2')
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    #pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

   # conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 128, name = 'conv4', dilate = (1, 1))
    #bn4 = mx.symbol.BatchNorm(data = conv4, name = 'batchnorm3')
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
   # relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    
    relu4 = mx.symbol.Flatten(data = relu3)
    
    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 256, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
    #relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')

    conv6_1 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_1')

    output1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax1')

    mtcnn = mx.symbol.Group([output1])
    
    return mtcnn
    
#    conv6_4 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_4')
def get_rotation_symbol():
    
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter = 64, name = 'conv1', dilate = (1, 1))
    #bn1 = mx.symbol.BatchNorm(data = conv1)
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter= 128, name = 'conv2', dilate = (1, 1))
    #bn2 = mx.symbol.BatchNorm(data = conv2)
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter = 128, name = 'conv3', dilate = (1, 1))
    #bn3 = mx.symbol.BatchNorm(data = conv3)
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 256, name = 'conv4', dilate = (1, 1))
    #bn4 = mx.symbol.BatchNorm(data = conv4)
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    relu4 = mx.symbol.Flatten(data = relu4)
    
    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 16, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')

    conv6_1 = mx.symbol.FullyConnected(data=relu5, num_hidden = 1, name = 'conv6_1')

    conv6_2 = mx.symbol.FullyConnected(data=relu5, num_hidden = 1, name = 'conv6_2')
            
    conv6_3 = mx.symbol.FullyConnected(data=relu5, num_hidden = 1, name = 'conv6_3')

    mtcnn = mx.symbol.Group([conv6_1, conv6_2, conv6_3])
    return mtcnn

def get_true_symbol():
    
    data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter = 32, name = 'conv1', dilate = (1, 1), pad = (1, 1))

    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter= 64, name = 'conv2', dilate = (1, 1), pad = (1, 1))

    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')
    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter = 64, name = 'conv3', dilate = (1, 1), pad = (1, 1))
  #  bn3 = mx.symbol.BatchNorm(data = conv3)
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv4 = mx.symbol.Convolution(data = pool3, kernel = (3, 3), num_filter = 128, name = 'conv4', dilate = (1, 1), pad = (1, 1))
    #relu4 = mx.symbol.relu(data=conv4, act_type="prelu", name = 'prelu4')
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    pool4 = mx.symbol.Pooling(data = relu4, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool4')

    relu5 = mx.symbol.Flatten(data = pool4)
    
    conv6 = mx.symbol.FullyConnected(data = relu5, num_hidden = 256, name = 'conv6')

    relu6 = mx.symbol.LeakyReLU(data=conv6, act_type="prelu", name = 'prelu6')
    #relu6 = mx.symbol.Dropout(data = relu6, p = 0.5, name = 'dropout1')

    conv6_1 = mx.symbol.FullyConnected(data=relu6, num_hidden = 2, name = 'conv6_1')

    output1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax')

    mtcnn = mx.symbol.Group([output1])
    
    return mtcnn

def get_clear_symbol():

    data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter = 32, name = 'conv1', dilate = (1, 1))
#    bn1 = mx.symbol.BatchNorm(data = conv1)
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter= 64, name = 'conv2', dilate = (1, 1))

 #   bn2 = mx.symbol.BatchNorm(data = conv2)
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter = 64, name = 'conv3', dilate = (1, 1))
  #  bn3 = mx.symbol.BatchNorm(data = conv3)
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 128, name = 'conv4', dilate = (1, 1))
   # bn4 = mx.symbol.BatchNorm(data = conv4)
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')

    #pool4 = mx.symbol.Pooling(data = relu4, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool4')

    relu4 = mx.symbol.Flatten(data = relu4)
    
    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 256, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
    #relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')

    conv6_1 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_1')

    output1 = mx.symbol.SoftmaxActivation(data=conv6_1,  name='softmax1')

    mtcnn = mx.symbol.Group([output1])
    
    return mtcnn


def get_PNet():
    
    cur_path = os.path.dirname(__file__)
    #model_path = os.path.join(cur_path, '../stage1/mtcnn')
    model_path = os.path.join(cur_path, '../stage1/mtcnn_online_new')
    #model_path = os.path.join(cur_path, '../stage1/wider_faces_based_model/mtcnn')

    _, arg_params, __ = mx.model.load_checkpoint(model_path, 10)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage1/org_model/det1", 1)
    sym = get_stage1_symbol()
    return sym, arg_params

    #data_shape = [("data", (1, 3, 693, 512))]
    #input_shapes = dict(data_shape)

    #executor = sym.simple_bind(ctx = mx.cpu())
    #for key in executor.arg_dict.keys():
    #    if key in arg_params:
    #        arg_params[key].copyto(executor.arg_dict[key])

   # return executor
def get_RNet():
    cur_path = os.path.dirname(__file__)
    #model_path = os.path.join(cur_path, '../stage2/mtcnn')
    model_path = os.path.join(cur_path, '../stage2/mtcnn_online_new')
    #model_path = os.path.join(cur_path, '../stage2/wider_faces_based_model/mtcnn')

    _, arg_params, __ = mx.model.load_checkpoint(model_path, 2)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage2/org_model/det2", 1)
    sym = get_stage2_symbol()
    return sym, arg_params

def get_ONet():
    cur_path = os.path.dirname(__file__)
    #model_path = os.path.join(cur_path, '../stage3/mtcnn_online')
    #model_path = os.path.join(cur_path, '../stage3/mtcnn_online_new')
    #model_path = os.path.join(cur_path, '../stage3/wider_faces_based_model/mtcnn')
    model_path = os.path.join(cur_path, '../stage3/mtcnn_wider_online_with_reg_grad1000_ratio1_new_320w')
    _, arg_params, __ = mx.model.load_checkpoint(model_path, 3)
   # _, arg_params, __ = mx.model.load_checkpoint("../stage3/org_model/det3", 1)
    sym = get_stage3_symbol()
    return sym, arg_params

def get_gender_attractive_Net():
    
    cur_path = os.path.dirname(__file__)
    model_path = os.path.join(cur_path, '../stage4_gender_attractive/mtcnn_online')
#   model_path = os.path.join(cur_path, '../stage4_gender_attractive/mtcnn_online_new')

    _, arg_params, __ = mx.model.load_checkpoint(model_path, 10)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage4_smile/mtcnn-celeba", 10)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage4_gender/mtcnn", 3)
   # _, arg_params, __ = mx.model.load_checkpoint("../stage4/wider_faces_based_model/mtcnn", 2)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage4/mtcnn", 10)

    #_, arg_params, __ = mx.model.load_checkpoint("../stage4/ord_model/mtcnn", 1)
    sym = get_stage4_symbol()
    #sym = get_vgg16_net()
    return sym, arg_params

def get_smile_Net():
    cur_path = os.path.dirname(__file__)
    #model_path = os.path.join(cur_path, '../stage4_smile/mtcnn_online')
    #_, arg_params, __ = mx.model.load_checkpoint(model_path, 10)
    model_path = os.path.join(cur_path, '../stage4_smile/mtcnn_online_with_google')
    _, arg_params, __ = mx.model.load_checkpoint(model_path, 100)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage4_smile/mtcnn-celeba", 10)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage4_gender/mtcnn", 3)
   # _, arg_params, __ = mx.model.load_checkpoint("../stage4/wider_faces_based_model/mtcnn", 2)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage4/mtcnn", 10)
    #_, arg_params, __ = mx.model.load_checkpoint("../stage4/ord_model/mtcnn", 1)
    sym = get_stage4_symbol()
    #sym = get_vgg16_net()
    return sym, arg_params

def get_attractive_Net():
    cur_path = os.path.dirname(__file__)
    model_path = os.path.join(cur_path, '../stage4_attractive/mtcnn_online_output2')
    #odel_path = os.path.join(cur_path, '../stage4_attractive/mtcnn_online_new')
    _, arg_params, __ = mx.model.load_checkpoint(model_path, 2)
    sym = get_attractive_symbol()
    #sym = get_vgg16_net()
    return sym, arg_params

def get_attractive_small_Net():
    cur_path = os.path.dirname(__file__)
    model_path = os.path.join(cur_path, '../stage4_attractive_small/mtcnn_online_output2')
    _, arg_params, __ = mx.model.load_checkpoint(model_path, 2)
    sym = get_attractive_small_symbol()
    #sym = get_vgg16_net()
    return sym, arg_params


def get_QNet():

    cur_path = os.path.dirname(__file__)
#    model_path = os.path.join(cur_path, '../stage5/mtcnn_output7')
    model_path = os.path.join(cur_path, '../stage5/mtcnn_output8_balance')
    _, arg_params, __ = mx.model.load_checkpoint(model_path, 20)
    sym = get_stage5_symbol()
    return sym, arg_params

def get_rotation_Net():
    cur_path = os.path.dirname(__file__)
    
    model_path = os.path.join(cur_path, '../stage4_rotation/mtcnn_online_three_grad100')
    _, arg_params, __ = mx.model.load_checkpoint(model_path, 100)
    sym = get_rotation_symbol()
    #sym = get_vgg16_net()
    return sym, arg_params

def get_glass_Net():
    cur_path = os.path.dirname(__file__)
    
    model_path = os.path.join(cur_path, '../stage4_glass/mtcnn_online')

    _, arg_params, aux_params = mx.model.load_checkpoint(model_path, 10)

    sym = get_stage4_symbol()
    
    return sym, arg_params


def get_true_Net():

    cur_path = os.path.dirname(__file__)
    
    model_path = os.path.join(cur_path, '../face_true/face_true')
    #model_path = os.path.join(cur_path, '../face_true/mtcnn_face_or_not')

    _, arg_params, aux_params = mx.model.load_checkpoint(model_path, 1)

    sym = get_true_symbol()
    
    return sym, arg_params
    
 
def get_clear_Net():
    cur_path = os.path.dirname(__file__)
    
    model_path = os.path.join(cur_path, '../face_clear/face_clear')
    #model_path = os.path.join(cur_path, '../face_clear/mtcnn_clear_or_not')
    _, arg_params, aux_params = mx.model.load_checkpoint(model_path, 1)
    sym = get_clear_symbol()
    
    return sym, arg_params

