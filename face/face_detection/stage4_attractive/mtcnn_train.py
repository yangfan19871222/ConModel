"""
LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
Gradient-based learning applied to document recognition.
Proceedings of the IEEE (1998)
"""
import mxnet as mx
import logging
logging.basicConfig(level = logging.DEBUG)

import os, sys
sys.path.insert(0, '../../python')
import cv2 
import numpy as np

multi_task = False
landmarks_task = False

attractive_task = True
smile_task = False
#glass_task = True

class FaceBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class FaceDataIter(mx.io.DataIter):
    def __init__(self, batchsize, data_root, data_shape, max_iter, begin = 0):
        super(FaceDataIter, self).__init__()
        self.batchsize = batchsize
        self.batch_size = batchsize
        self.data_root = data_root
        self.data_shape = data_shape
        self.max_iter = max_iter
        self.cur_iter = 0
        self.data = []
        self.label = []
        self.outside_weights = []
        self.landmarks_weights = []
        self.landmarks = []
        self.bbox_targets = []
        self.total_label = []
        self.total_label_tmp = []
        self.index = 0
        self.f = open(data_root)
        self.begin = begin
        self.total_data = []
        for i in range(begin):
            self.f.readline()
       

    @property
    def provide_data(self):
        return [('data', (self.batchsize, self.data_shape[0], self.data_shape[1], self.data_shape[2]))]

    @property
    def provide_label(self):
       
        return  [('attractive_label', (self.batchsize, ))]
                 #('smile_label',  (self.batchsize, ))]# \
                 #('glass_label',  (self.batchsize, ))]

        if landmarks_task == True:
            return  [('softmax_label', (self.batchsize, )),\
                     ('outside_weights', (self.batchsize, 4)),\
                     ('bbox_targets',  (self.batchsize, 4)), \
                     ('landmarks_weights', (self.batchsize, 10)),\
                     ('landmarks', (self.batchsize, 10))]

        if multi_task == True:
            return  [('softmax_label', (self.batchsize, )),\
                     ('outside_weights', (self.batchsize, 4)),\
                     ('bbox_targets',  (self.batchsize, 4))]
            
        return [('softmax_label', (self.batchsize, ))]
            
        
    def __iter__(self):
        while self.cur_iter < self.max_iter:
            self.get_train_data()
            self.cur_iter += self.batch_size
            
            yield FaceBatch(['data'], [self.data], ['attractive_label'], self.total_label_tmp) 
            #print 'test'
           # if landmarks_task == True:
           #     yield FaceBatch(['data'], [self.data], ['softmax_label', 'outside_weights', 'bbox_targets', 'landmarks_weights', 'landmarks'], self.total_label) 

           # elif multi_task == True:
            #    yield FaceBatch(['data'], [self.data], ['softmax_label', 'outside_weights', 'bbox_targets'], self.total_label) 
            #else:
            #    yield FaceBatch(['data'], [self.data], ['softmax_label'], [self.label]) 

    def next(self):
        if self.cur_iter <= self.max_iter:
            
            print 'test111'
            self.get_train_data()
            #print self.data.size
            self.cur_iter += self.batchsize
            return mx.io.DataBatch(data = [self.data],\
                                   label = [self.label],\
                                   provide_data = self.provide_data,\
                                   provide_label = self.provide_label)
                 #                  index = self.cur_iter / self.batch_size, \
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    
    def reset(self):
        self.cur_iter = 0
        self.f.seek(0,0)
        for i in range(self.begin):
            self.f.readline()

    def get_train_data(self):
        self.data = []
        self.gender_label = []
        self.attractive_label = []
        self.smile_label = []
        self.glass_label = []
        self.outside_weights = []
        self.bbox_targets = []
        self.total_label_tmp = []
        self.landmarks = []
        self.landmarks_weights = []
        if len(self.total_data) == self.max_iter / self.batch_size:
            self.data = self.total_data[self.cur_iter / self.batch_size]
            self.total_label_tmp = self.total_label[self.cur_iter / self.batch_size]
            return
      #  self.label = np.random.randint(0, 3, [self.batch_size,])
      #  self.data = np.random.uniform(-1, 1, [self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
      #  self.data = mx.nd.array(self.data)
      #  self.label = mx.nd.array(self.label)
      #  return
        i = 0
#        if self.cur_iter % 200000 == 0:
#            self.f.seek(0, 0)
       # for i in range(self.batchsize):
        non_face_rand_male = 0
        non_face_rand_female = 0
        non_face_rand_kid = 0
        while i < self.batchsize:

            cur_file = self.f.readline()
            cur_file_list = cur_file.split(' ')
            cur_file_list[-1] = cur_file_list[-1].strip('\n')
            cur_file_list = [cur_file_list[j] for j in range(len(cur_file_list)) if cur_file_list[j]]
            try:
                if float(cur_file_list[2]) < 45 and float(cur_file_list[2]) > 30 and cur_file_list[1] == 'male' and float(cur_file_list[3]) > 10:
                    #non_face_rand_male += 1
                    #if non_face_rand_male % 5 != 0:
                    continue
                if float(cur_file_list[2]) < 55 and float(cur_file_list[2]) > 35  and cur_file_list[1] == 'female' and float(cur_file_list[3]) > 10:
                    #non_face_rand_female += 1
                    #if non_face_rand_female % 2 == 0:
                    continue
                if float(cur_file_list[2]) < 40 and float(cur_file_list[2]) > 20 and float(cur_file_list[3]) < 10:
                    continue
                   # non_face_rand_kid += 1
                   # if non_face_rand_kid % 10 != 0:
                   #     continue
            except Exception:
                print cur_file
                continue
           # if 'part' in cur_file_list[0]:
           #     continue
            
            file_path = os.path.join('/media/mfs/fordata/td_hg2_web_server/yangfan/online_data_200w/', cur_file_list[0])
            #file_path = os.path.join(os.path.dirname(self.data_root) + '/../images_stage4/', cur_file_list[0])
            #print file_path
            try:
                if not os.path.exists(file_path):
                    print 'file not exist'
                    continue
                #int(cuf_file_list[1].strip('\n'))
            except Exception:
                print file_path
                continue
            #print file_path  
            image = cv2.imread(file_path)
            if image == None:
             #   print 'file is none'
                continue
            if image.shape[1] < 24 or image.shape[0] < 24:
                print 'file too small'
                continue
            #print file_path
            height = image.shape[0]
            width = image.shape[1]
#            image = image[int(0.1 * height): int(0.9 * height), int(0.1 * width): int(0.9 * width)]
            image = cv2.resize(image, (48, 48))
            image = np.multiply(image - 127.5, 1.0 / 127.5)
            image = image.transpose((2, 0, 1))
            self.data.append(image)
           # if int(cur_file_list[21]) == -1:
           #     self.gender_label.append(0)
           # if int(cur_file_list[21]) == 1:
           #     self.gender_label.append(1)
           # if int(cur_file_list[3]) == -1:
           #     self.attractive_label.append(0)
           # if int(cur_file_list[3]) == 1:
           #     self.attractive_label.append(1)
#            if int(cur_file_list[32]) == -1:
#                self.smile_label.append(0)
#            if int(cur_file_list[32]) == 1:
#                self.smile_label.append(1)
#             
#            if int(cur_file_list[16]) == -1:
#                self.glass_label.append(0)
#            if int(cur_file_list[16]) == 1:
#                self.glass_label.append(1)

            if cur_file_list[1] == 'female':
                self.gender_label.append(0)
            else:
                self.gender_label.append(1)
           
            attr = float(cur_file_list[2])
            if float(cur_file_list[3]) < 10:
                if attr >= 40:
                    attr = 1
                elif attr <= 20:
                    attr = 0
            else:
                if cur_file_list[1] == 'female':

                    if attr >= 55:
                        attr = 1
                    elif attr <= 35:
                        attr = 0
                elif cur_file_list[1] == 'male':
                    if attr >= 45:
                        attr = 1
                    elif attr <= 30:
                        attr = 0
                
            self.attractive_label.append(attr)
            
           # self.glass_label.append(int(cur_file_list[4]))
           # self.smile_label.append(int(cur_file_list[5]))

            
            i += 1

#            self.label.append(int(cur_file_list[1].strip('\n')))
#            if len(cur_file_list) == 6:
#                self.bbox_targets.append([float(cur_file_list[2]), float(cur_file_list[3]), float(cur_file_list[4]), float(cur_file_list[5].strip('\n'))])
#                self.outside_weights.append([1, 1, 1, 1])
#                self.landmarks_weights.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#                self.landmarks.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#            if len(cur_file_list) < 6:
#                self.bbox_targets.append([0, 0, 0, 0])
#                self.outside_weights.append([0, 0, 0, 0])
#                self.landmarks_weights.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#                self.landmarks.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#            if len(cur_file_list) == 16:
#                self.bbox_targets.append([float(cur_file_list[2]), float(cur_file_list[3]), float(cur_file_list[4]), float(cur_file_list[5])])
#                self.outside_weights.append([0, 0, 0, 0])
#                self.landmarks_weights.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#                self.landmarks.append([float(cur_file_list[6]), float(cur_file_list[7]), float(cur_file_list[8]), float(cur_file_list[9]),\
 #                   float(cur_file_list[10]), float(cur_file_list[11]), float(cur_file_list[12]), float(cur_file_list[13]), \
 #                   float(cur_file_list[14]), float(cur_file_list[15].strip('\n'))])
            
        self.data = mx.nd.array(self.data)
        self.gender_label = mx.nd.array(self.gender_label)
        self.attractive_label = mx.nd.array(self.attractive_label)
        self.smile_label = mx.nd.array(self.smile_label)
#        self.glass_label = mx.nd.array(self.glass_label)
#        self.bbox_targets = mx.nd.array(self.bbox_targets)
#        self.outside_weights = mx.nd.array(self.outside_weights)
#       self.landmarks_weights = mx.nd.array(self.landmarks_weights)
#        self.landmarks = mx.nd.array(self.landmarks)
#        if landmarks_task == True:
#            self.total_label = [self.label, self.outside_weights, self.bbox_targets, self.landmarks_weights, self.landmarks]
#        else:
#            self.total_label = [self.label, self.outside_weights, self.bbox_targets]
        self.total_label.append([self.attractive_label])
        self.total_data.append(self.data)
        self.total_label_tmp = [self.attractive_label]
         

def get_symbol():
    data = mx.symbol.Variable('data')
    #gender_label = mx.symbol.Variable('gender_label')
    attractive_label = mx.symbol.Variable('attractive_label')
   # smile_label = mx.symbol.Variable('smile_label')
    #glass_label = mx.symbol.Variable('glass_label')

    if landmarks_task ==True:
       landmarks_weights = mx.symbol.Variable('landmarks_weights')
       landmarks = mx.symbol.Variable('landmarks')
        
    if multi_task == True: 
       outside_weights = mx.symbol.Variable('outside_weights')
       bbox_targets = mx.symbol.Variable('bbox_targets')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter = 20, name = 'conv1', dilate = (1, 1))
    #bn1 = mx.symbol.BatchNorm(data = conv1)
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter= 40, name = 'conv2', dilate = (1, 1))
    #bn2 = mx.symbol.BatchNorm(data = conv2)
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter = 40, name = 'conv3', dilate = (1, 1))
    #bn3 = mx.symbol.BatchNorm(data = conv3)
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    #conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 64, name = 'conv4', dilate = (1, 1))
    #bn4 = mx.symbol.BatchNorm(data = conv4)
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    #relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    relu4 = mx.symbol.Flatten(data = pool3)
    
    #conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 64, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
   # relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
   # relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')

    #conv6_1 = mx.symbol.FullyConnected(data=relu4, num_hidden = 2, name = 'conv6_1')
    #loss1 = mx.symbol.SoftmaxOutput(data=conv6_1, label = gender_label, name='softmax1')
    #return loss1

   # if attractive_task == True:
    conv6_1 = mx.symbol.FullyConnected(data=relu4, num_hidden = 2, name = 'conv6_1')
    loss1 = mx.symbol.SoftmaxOutput(data=conv6_1, label = attractive_label, name='softmax1')
    
    if smile_task == True:
        conv6_3 = mx.symbol.FullyConnected(data=relu4, num_hidden = 2, name = 'conv6_3')
        loss3 = mx.symbol.SoftmaxOutput(data=conv6_3, label = smile_label, name='softmax3')
     
   # if glass_task == True:
   #     conv6_4 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_4')
   #     loss4 = mx.symbol.SoftmaxOutput(data=conv6_4, label = glass_label, name='softmax4')
        
    mtcnn = mx.symbol.Group([loss1])

    return mtcnn

    if multi_task == True:
        
        conv6_2 = mx.symbol.FullyConnected(data=relu5, num_hidden = 4, name = 'conv6_2')

        loss2_ = mx.symbol.square(data = outside_weights * (conv6_2 - bbox_targets))

        loss2_ = mx.symbol.sum(data = loss2_, axis = (1))

        loss2 = mx.symbol.MakeLoss(data = loss2_, name = 'bbox_loss', grad_scale = 10.0, normalization = 'valid')
            
        mtcnn = mx.symbol.Group([loss1, loss2])

        if landmarks_task == True:
            
            conv6_3 = mx.symbol.FullyConnected(data=relu5, num_hidden = 10, name = 'conv6_3')

            loss3_ = mx.symbol.square(data = landmarks_weights * (conv6_3 - landmarks))

            loss3_ = mx.symbol.sum(data = loss3_, axis = (1))

            loss3 = mx.symbol.MakeLoss(data = loss3_, name = 'landmark_loss', grad_scale = 10.0, normalization = 'valid')
            
            mtcnn = mx.symbol.Group([loss1, loss2, loss3])
            
            return mtcnn

        return mtcnn

    return loss1

def get_vgg16_net():
    data = mx.symbol.Variable('data')
    gender_label = mx.symbol.Variable('gender_label')
    attractive_label = mx.symbol.Variable('attractive_label')
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

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
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
    relu15 = mx.symbol.Dropout(data = relu15, p = 0.5, name = 'dropout2')

    fc16 = mx.symbol.FullyConnected(data=relu15, num_hidden = 2, name = 'fc8_101')

    loss1 = mx.symbol.SoftmaxOutput(data=fc16, label = gender_label, name='softmax', grad_scale = 1.0)

    fc17 = mx.symbol.FullyConnected(data=relu15, num_hidden = 2, name = 'fc8_102')

    loss2 = mx.symbol.SoftmaxOutput(data=fc17, label = attractive_label, name='softmax1', grad_scale = 1.0)
    #loss1 = mx.symbol.MakeLoss(data = loss1_, name = 'softmax_loss', grad_scale = 10.0, normalization = 'valid')
    
    return mx.symbol.Group([loss1, loss2])

class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        super(Multi_Accuracy, self).__init__('multi-accuracy', num)

    def update(self, labels, preds):
        #mx.metric.check_label_shapes(labels, preds)

        if self.num != None:
            assert len(labels) == self.num

        for i in range(len(labels)):
#            if i == 1 or i == 3:
#                continue
#
#            if i == 2:
#                pred = preds[i - 1].asnumpy().astype('float32')
#                label = labels[i - 1].asnumpy().astype('float32')
#               # print pred.shape
#               # print label.shape
#                self.sum_metric[i] += np.sum(pred)
#                for t in range(len(label)):
#                    if label[t, 0] != 0:
#                        self.num_inst[i] += 1
#                continue
#            
#            if i == 4:
#                pred = preds[i - 2].asnumpy().astype('float32')
#                label = labels[i - 1].asnumpy().astype('float32')
#                self.sum_metric[i] += np.sum(pred)
#                for t in range(len(label)):
#                    if label[t, 0] != 0:
#                        self.num_inst[i] += 1
#
#            if i == 0:
             pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
             label = labels[i].asnumpy().astype('int32')
             mx.metric.check_label_shapes(label, pred_label)
             

             if i == None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
             else:
                 self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                 self.num_inst[i] += len(pred_label.flat)
       
        #print float(self.sum_metric[0]) / float(self.num_inst[0])

if __name__ == '__main__':

    #root_path = '/media/disk1/yangfan/online_data_more/mtcnn_training/'
    #root_path = '/media/disk1/yangfan/celeba/mtcnn_data/'
    root_path = '/media/disk1/yangfan/online_data_200w/mtcnn_training/'
    network = get_symbol()
    #network = get_vgg16_net()
    batch_size = 256
    train = FaceDataIter(batch_size, os.path.join(root_path, './imglist_stage4/train.txt'), (3, 48, 48), 512000, begin = 0)
    val = FaceDataIter(batch_size, os.path.join(root_path, './imglist_stage4/train'), (3, 48, 48), 12800, begin = 2500000)
    device = mx.gpu(1)
    lr = 0.01

    #params = 'mtcnn_smaller-0010.params'
   
    params = 'mtcnn_online_output2-0002.params'
    #params = 'org_model/det3-0001.params'
    model = mx.model.FeedForward(
        ctx                = device,
        symbol             = network,
        num_epoch          = 2,
        learning_rate      = lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        #initializer        = mx.init.Load(params, default_init = mx.init.Xavier(factor_type="in", magnitude=2.34))
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34)
        )
    
   # if landmarks_task == True:
  #      label_num = 5
  #  elif multi_task == True:
  #      label_num = 3
  #  else:
  #      label_num = 1
   
    label_num = 1

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = Multi_Accuracy(num = label_num),
        batch_end_callback = mx.callback.Speedometer(batch_size, 50))

    model.save('mtcnn_online_output2_40')
