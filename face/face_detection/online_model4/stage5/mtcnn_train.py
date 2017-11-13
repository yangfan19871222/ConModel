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

age_task = True

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
        self.index = 0
        self.f = open(data_root)
        self.begin = begin
        self.total_data = []
        self.total_label = []
        for i in range(begin):
            self.f.readline()
       

    @property
    def provide_data(self):
        return [('data', (self.batchsize, self.data_shape[0], self.data_shape[1], self.data_shape[2]))]

    @property
    def provide_label(self):
       
        return  [('age_label', (self.batchsize, ))]

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
            
            yield FaceBatch(['data'], [self.data], ['age_label'], [self.age_label]) 
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
   #     for i in range(self.begin):
   #         self.f.readline()

    def get_train_data(self):
        self.data = []
        self.gender_label = []
        self.attractive_label = []
        self.smile_label = []
        self.age_label = []
        self.outside_weights = []
       # self.bbox_targets = []
       # self.total_label = []
        self.landmarks = []
        self.landmarks_weights = []
        if len(self.total_data) == self.max_iter / self.batch_size:
            self.data = self.total_data[self.cur_iter / self.batch_size]
            self.age_label = self.total_label[self.cur_iter / self.batch_size]
            return

            
     #   if self.first_time == 0 and len(self.total_data) > 0:
            
      #  self.label = np.random.randint(0, 3, [self.batch_size,])
      #  self.data = np.random.uniform(-1, 1, [self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
      #  self.data = mx.nd.array(self.data)
      #  self.label = mx.nd.array(self.label)
      #  return
        i = 0
#        if self.cur_iter % 200000 == 0:
#            self.f.seek(0, 0)
       # for i in range(self.batchsize):
        face_rand = 0
        while i < self.batchsize:

            cur_file = self.f.readline()
            cur_file_list = cur_file.split(' ')
            cur_file_list[-1] = cur_file_list[-1].strip('\n')
            cur_file_list = [cur_file_list[j] for j in range(len(cur_file_list)) if cur_file_list[j]]
            try:
                if cur_file_list[1].strip('\n') == '1' and len(cur_file_list) > 6:
                    face_rand += 1
                   # if face_rand % 4 != 0:
                   #     continue
            except Exception:
                print cur_file
                continue
           # if 'part' in cur_file_list[0]:
            file_path = os.path.join('/media/mfs/fordata/td_hg2_web_server/yangfan/online_data_200w/', cur_file_list[0])
            try:
                if not os.path.exists(file_path):
                    print 'file not exist'
                    continue
                #int(cuf_file_list[1].strip('\n'))
            except Exception:
                print file_path
                continue
            
            image = cv2.imread(file_path)
            if image == None:
            #    print 'file is none'
                continue
            if image.shape[1] < 24 or image.shape[0] < 24:
                print 'file too small'
                continue
            #print file_path
            image = cv2.resize(image, (48, 48))
            image = np.multiply(image - 127.5, 1.0 / 127.5)
            image = image.transpose((2, 0, 1))
            self.data.append(image)
            age = float(cur_file_list[3])
            
            age = int(age)
            if age <= 10:
                age = 0
            elif age >= 35:
                age = 6
            else:
                age = (age  - 10) / 5 + 1
            
           # age_label = []
           # if age >= 40:
           #    age = 2
           # elif age < 40  and age > 20:
           #    age = 1
           # else:
           #    age = 0
#            for t in range(101):
#                if t < age:
#                    age_label.append(1)
#                else:
#                    age_label.append(0)
            self.age_label.append(age)
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
        self.age_label = mx.nd.array(self.age_label)
        self.total_data.append(self.data)
        self.total_label.append(self.age_label)
#        self.bbox_targets = mx.nd.array(self.bbox_targets)
#        self.outside_weights = mx.nd.array(self.outside_weights)
#       self.landmarks_weights = mx.nd.array(self.landmarks_weights)
#        self.landmarks = mx.nd.array(self.landmarks)
#        if landmarks_task == True:
#            self.total_label = [self.label, self.outside_weights, self.bbox_targets, self.landmarks_weights, self.landmarks]
#        else:
#            self.total_label = [self.label, self.outside_weights, self.bbox_targets]
       # self.total_label = [self.age_label]

def get_symbol():
    data = mx.symbol.Variable('data')
    age_label = mx.symbol.Variable('age_label')
    if landmarks_task ==True:
       landmarks_weights = mx.symbol.Variable('landmarks_weights')
       landmarks = mx.symbol.Variable('landmarks')
        
    if multi_task == True: 
       outside_weights = mx.symbol.Variable('outside_weights')
       bbox_targets = mx.symbol.Variable('bbox_targets')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter = 32, name = 'conv1', dilate = (1, 1))
    #relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    relu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name = 'prelu1')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name = 'pool1')
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter= 64, name = 'conv2', dilate = (1, 1))
    relu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name = 'prelu2')

    #relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name = 'pool2')

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter = 64, name = 'conv3', dilate = (1, 1))
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name = 'prelu3')

    pool3 = mx.symbol.Pooling(data = relu3, pool_type = "max", kernel=(2, 2), stride=(2, 2), name = 'pool3')

    conv4 = mx.symbol.Convolution(data = pool3, kernel = (2, 2), num_filter = 128, name = 'conv4', dilate = (1, 1))
   # relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    relu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name = 'prelu4')
    relu4 = mx.symbol.Flatten(data = relu4)
    
    conv5 = mx.symbol.FullyConnected(data = relu4, num_hidden = 128, name = 'conv5')
    #relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    relu5 = mx.symbol.LeakyReLU(data=conv5, act_type="prelu", name = 'prelu5')
    relu5 = mx.symbol.Dropout(data = relu5, p = 0.5, name = 'dropout1')

    conv6 = mx.symbol.FullyConnected(data=relu5, num_hidden = 7, name = 'conv6_1')
   # relu6 = mx.symbol.LeakyReLU(data=conv6, act_type="prelu", name = 'prelu6')

   # conv7 = mx.symbol.FullyConnected(data=relu6, num_hidden = 1, name = 'conv7_1_tmp')

   # loss1_ = mx.symbol.square(data = (conv7 - age_label))
    
   # loss1 = mx.symbol.sum(data = loss1_, axis = (1))

   # loss1 = mx.symbol.MakeLoss(data = loss1,  grad_scale = 0.01, normalization = 'valid')
            
   # return loss1

    loss1 = mx.symbol.SoftmaxOutput(data=conv6, label = age_label, name='softmax1', grad_scale = 1.0)

    #loss1 = mx.symbol.MakeLoss(data = loss1_, name = 'softmax_loss', grad_scale = 10.0, normalization = 'valid')
            
    return loss1

    if attractive_task == True:
        conv6_2 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_2')
        loss2 = mx.symbol.SoftmaxOutput(data=conv6_2, label = attractive_label, name='softmax2')
    
    if smile_task == True:
        conv6_3 = mx.symbol.FullyConnected(data=relu5, num_hidden = 2, name = 'conv6_3')
        loss3 = mx.symbol.SoftmaxOutput(data=conv6_3, label = smile_label, name='softmax3')
    mtcnn = mx.symbol.Group([loss1, loss2, loss3])

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
    age_label = mx.symbol.Variable('age_label')
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
    fc15 = mx.symbol.FullyConnected(data = relu14, num_hidden = 4096, name = 'fc7_tmp')
    relu15 = mx.symbol.Activation(data=fc15, act_type="relu")
    relu15 = mx.symbol.Dropout(data = relu15, p = 0.5, name = 'dropout2')

    fc16 = mx.symbol.FullyConnected(data=relu15, num_hidden = 7, name = 'fc8_101')

    loss1 = mx.symbol.SoftmaxOutput(data=fc16, label = age_label, name='softmax', grad_scale = 1.0)

    #loss1 = mx.symbol.MakeLoss(data = loss1_, name = 'softmax_loss', grad_scale = 10.0, normalization = 'valid')
            
    return loss1


class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        super(Multi_Accuracy, self).__init__('multi-accuracy', num)

    def update(self, labels, preds):
        #mx.metric.check_label_shapes(labels, preds)

        if self.num != None:
            assert len(labels) == self.num

        for i in range(len(labels)):
#             self.sum_metric[i] += 1.0#(label[:, pred_label] == label_check).sum()
#             self.num_inst[i] += 16#len(pred_label.flat)
#             break
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
#             print pred_label
             label = labels[i].asnumpy().astype('int32')
            # pred_label = preds[i].asnumpy()
#             print pred_label
             #print label
             mx.metric.check_label_shapes(label, pred_label)
             #print label.shape
            # pred_label = np.reshape(pred_label, (256, 1))
                
            # print pred_label.shape
            # print label.shape
             if i == None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
             else:
                 #self.sum_metric[i] += (pred_label.flat == label.flat).sum()
#                 for t in range(len(pred_label)):
#                    if label[t, pred_label[t]] == 1:
#                        self.sum_metric[i] += 1
                 self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                 #print (pred_label - label).sum()
                 #self.sum_metric[i] += np.square(pred_label -  label).sum()
                 self.num_inst[i] += label.shape[0]
       
        #print float(self.sum_metric[0]) / float(self.num_inst[0])

if __name__ == '__main__':

    #root_path = '/media/disk1/yangfan/imdb_data/mtcnn_training/'
    root_path = '/media/disk1/yangfan/online_data_200w/mtcnn_training/'
    network = get_symbol()
    #network = get_vgg16_net()
    batch_size = 256
    train = FaceDataIter(batch_size, os.path.join(root_path, './imglist_stage4/train.txt'), (3, 48, 48), 512000, begin = 0)
    val = FaceDataIter(batch_size, os.path.join(root_path, './imglist_stage4/train.txt'), (3, 48, 48), 12800, begin = 600000)
    device = mx.gpu(0)
    lr = 0.0001

    #params = 'mtcnn-0002.params'
    params = 'mtcnn_output7-0050.params'
    #params = 'vgg16-0001.params'
    #params = 'org_model/det3-0001.params'
    model = mx.model.FeedForward(
        ctx                = device,
        symbol             = network,
        num_epoch          = 50,
        learning_rate      = lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Load(params, default_init = mx.init.Xavier(factor_type="in", magnitude=2.34))
        #initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34)
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

    model.save('mtcnn_output7')
