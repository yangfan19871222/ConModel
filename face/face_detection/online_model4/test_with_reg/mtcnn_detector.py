# coding: utf-8
import os
import mxnet as mx
import numpy as np
import math
import cv2
from multiprocessing import Pool
from itertools import repeat
from itertools import izip
from symbols import get_PNet, get_RNet, get_ONet, get_gender_attractive_Net, get_smile_Net, get_QNet, get_attractive_Net, get_attractive_small_Net, get_rotation_Net, get_glass_Net, get_true_Net, get_clear_Net
from time import time
from helper import nms, adjust_input, generate_bbox, detect_first_stage, detect_first_stage_warpper, init_executor
import threading
from nms.gpu_nms import *
from config import GPU_ID 

first_has_reg = True
has_reg = True
has_landmark = True
mx.Context(mx.gpu(GPU_ID))

class MyThread(threading.Thread):
    def __init__(self, arg):
        super(MyThread, self).__init__()
        self.arg = arg
        self.return_boxes = []
    def run(self):
        self.return_boxes = detect_first_stage_warpper(self.arg)

class MtcnnDetector(object):
    """
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a mxnet version
    """
    def __init__(self,
                 model_folder='.',
                 minsize = 20,
                 threshold = [0.6, 0.7, 0.8],
                 factor = 0.709,
                 num_worker = 1,
                 accurate_landmark = True,
                 ctx=mx.cpu()):
        """
            Initialize the detector

            Parameters:
            ----------
                model_folder : string
                    path for the models
                minsize : float number
                    minimal face to detect
                threshold : float number
                    detect threshold for 3 stages
                factor: float number
                    scale factor for image pyramid
                num_worker: int number
                    number of processes we use for first stage
                accurate_landmark: bool
                    use accurate landmark localization or not

        """
        self.num_worker = num_worker
        self.accurate_landmark = accurate_landmark

        # load 4 models from folder
        #models = ['det1', 'det2', 'det3','det4']
        #models = [ os.path.join(model_folder, f) for f in models]
        
       # models = []
       # models.append(os.path.join('model', 'mtcnn'))
       # models.append(os.path.join('../stage2', 'mtcnn'))
       # models.append(os.path.join('../stage3', 'mtcnn'))
        #self.PNets = []
        #for i in range(num_worker):
        #    workner_net = mx.model.FeedForward.load(models[0], 1, ctx=ctx)
        #    self.PNets.append(workner_net)

       # self.PNet = mx.model.FeedForward.load(models[0], 1, ctx=ctx)
        self.PNet, self.arg_params = get_PNet()
        self.RNet, self.arg_params2 = get_RNet() # mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.ONet, self.arg_params3 = get_ONet() # mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.LNet_0, self.arg_params4_0 = get_rotation_Net()
        self.LNet_1, self.arg_params4_1 = get_gender_attractive_Net() # mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.LNet_2, self.arg_params4_2 = get_smile_Net() # mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.LNet_3, self.arg_params4_3 = get_attractive_Net() # mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        #self.LNet_4, self.arg_params4_4 = get_attractive_small_Net() # mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.LNet_5, self.arg_params4_5 = get_glass_Net()

        self.LNet_true, self.arg_params_true = get_true_Net()

        self.LNet_clear, self.arg_params_clear = get_clear_Net()

        self.QNet, self.arg_params5 = get_QNet() # mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        #self.ONet = mx.model.FeedForward.load(models[2], 10, ctx=ctx)
        #self.LNet = mx.model.FeedForward.load(models[2], 1, ctx=ctx)

        self.minsize   = float(minsize)
        self.factor    = float(factor)
        self.threshold = threshold
        self.ctx = ctx

        self.second_stage_num = 200
        self.third_stage_num = 20
        self.fourth_stage_num = 20
        self.five_stage_num = 10
        # only works for color image
        # detected boxes
        MIN_DET_SIZE = 12
        self.height = 500
        self.width = 500
        minl = min(self.height, self.width)

        # get all the valid scales
        self.scales = []
        m = MIN_DET_SIZE/self.minsize
        minl *= m
        factor_count = 0
        while minl > MIN_DET_SIZE:
            self.scales.append(m*self.factor**factor_count)
            minl *= self.factor
            factor_count += 1
        
        #self.Pool = Pool(len(self.scales))
        i = 0
        self.executor1 = []
        for scale in self.scales:
            data_shape = [("data", (1, 3, int(self.height * scale), int(self.width * scale)))]
            input_shapes = dict(data_shape)
            self.executor1.append(self.PNet.simple_bind(ctx = self.ctx, **input_shapes))
            for key in self.executor1[i].arg_dict.keys():
                if key in self.arg_params:
                    self.arg_params[key].copyto(self.executor1[i].arg_dict[key])
            i += 1

        init_executor(self.scales, self.executor1)
        #self.Pool = Pool(i)

        data_shape = [("data", (self.second_stage_num, 3, 24, 24))]
        input_shapes = dict(data_shape)
        self.executor2 = self.RNet.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor2.arg_dict.keys():
            if key in self.arg_params2:
                self.arg_params2[key].copyto(self.executor2.arg_dict[key])

        data_shape = [("data", (self.third_stage_num, 3, 48, 48))]
        input_shapes = dict(data_shape)
        self.executor3 = self.ONet.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor3.arg_dict.keys():
            if key in self.arg_params3:
                self.arg_params3[key].copyto(self.executor3.arg_dict[key])

        data_shape = [("data", (self.fourth_stage_num, 3, 48, 48))]
        input_shapes = dict(data_shape)
        self.executor4_0 = self.LNet_0.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor4_0.arg_dict.keys():
            if key in self.arg_params4_0:
                self.arg_params4_0[key].copyto(self.executor4_0.arg_dict[key])


        data_shape = [("data", (self.fourth_stage_num, 3, 48, 48))]
        input_shapes = dict(data_shape)
        self.executor4_1 = self.LNet_1.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor4_1.arg_dict.keys():
            if key in self.arg_params4_1:
                self.arg_params4_1[key].copyto(self.executor4_1.arg_dict[key])

        data_shape = [("data", (self.fourth_stage_num, 3, 48, 48))]
        input_shapes = dict(data_shape)
        self.executor4_2 = self.LNet_2.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor4_2.arg_dict.keys():
            if key in self.arg_params4_2:
                self.arg_params4_2[key].copyto(self.executor4_2.arg_dict[key])

        
        data_shape = [("data", (self.fourth_stage_num, 3, 48, 48))]
        input_shapes = dict(data_shape)
        self.executor4_3 = self.LNet_3.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor4_3.arg_dict.keys():
            if key in self.arg_params4_3:
                self.arg_params4_3[key].copyto(self.executor4_3.arg_dict[key])

      #  data_shape = [("data", (self.fourth_stage_num, 3, 24, 24))]
      #  input_shapes = dict(data_shape)
      #  self.executor4_4 = self.LNet_4.simple_bind(ctx = self.ctx, **input_shapes)
      #  for key in self.executor4_4.arg_dict.keys():
      #      if key in self.arg_params4_4:
      #          self.arg_params4_4[key].copyto(self.executor4_4.arg_dict[key])

        data_shape = [("data", (self.fourth_stage_num, 3, 48, 48))]
        input_shapes = dict(data_shape)
        self.executor4_5 = self.LNet_5.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor4_5.arg_dict.keys():
            if key in self.arg_params4_5:
                self.arg_params4_5[key].copyto(self.executor4_5.arg_dict[key])

        data_shape = [("data", (self.fourth_stage_num, 3, 48, 48))]
        input_shapes = dict(data_shape)
        self.executor5 = self.QNet.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor5.arg_dict.keys():
            if key in self.arg_params5:
                self.arg_params5[key].copyto(self.executor5.arg_dict[key])

        data_shape = [("data", (self.fourth_stage_num, 3, 64, 64))]
        input_shapes = dict(data_shape)
        self.executor_true = self.LNet_true.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor_true.arg_dict.keys():
            if key in self.arg_params_true:
                self.arg_params_true[key].copyto(self.executor_true.arg_dict[key])

       # for key in self.executor_true.aux_dict.keys():
       #     if key in self.aux_params_true:
       #         self.aux_params_true[key].copyto(self.executor_true.aux_dict[key])

        data_shape = [("data", (self.fourth_stage_num, 3, 96, 96))]
        input_shapes = dict(data_shape)
        self.executor_clear = self.LNet_clear.simple_bind(ctx = self.ctx, **input_shapes)
        for key in self.executor_clear.arg_dict.keys():
            if key in self.arg_params_clear:
                self.arg_params_clear[key].copyto(self.executor_clear.arg_dict[key])

        #for key in self.executor_clear.aux_dict.keys():
        #    if key in self.aux_params_clear:
        #        self.aux_params_clear[key].copyto(self.executor_clear.aux_dict[key])


    def convert_to_square(self, bbox):
        """
            convert bbox to square

        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox

        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h,w)
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes

        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxex adjustment

        Returns:
        -------
            bboxes after refinement

        """
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox[:, 0:4] = bbox[:, 0:4] + aug
        return bbox

 
    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it

        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------s
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox

        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1,  bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx , dy= np.zeros((num_box, )), np.zeros((num_box, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]
        return  return_list

    def slice_index(self, number):
        """
            slice the index into (n,n,m), m < n
        Parameters:
        ----------
            number: int number
                number
        """
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]
        num_list = range(number)
        return list(chunks(num_list, self.num_worker))
        

    def detect_face(self, img):
        """
            detect face over img
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
        Retures:
        -------
            bboxes: numpy array, n x 5 (x1,y2,x2,y2,score)
                bboxes
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
                landmarks
        """

        # check input
        
        global_start_time = time()
        global_first_start_time = time()

        if img is None:
            return None

        # only works for color image
        if len(img.shape) != 3:
            return None

        # detected boxes
#        total_boxes = []

#        height, width, _ = img.shape
#        minl = min( height, width)

        # get all the valid scales
#        scales = []
#        m = MIN_DET_SIZE/self.minsize
#        minl *= m
#        factor_count = 0
#        while minl > MIN_DET_SIZE:
#            scales.append(m*self.factor**factor_count)
#            minl *= self.factor
#            factor_count += 1

        #############################################
        # first stage
        #############################################
        total_boxes = []
        i = 0
        self.index = []
        self.t = []
        for scale in self.scales:
            return_boxes = detect_first_stage(img, i, self.threshold[0], self.ctx)
            if return_boxes is not None:
                total_boxes.append(return_boxes)
            i += 1
  #          return_boxes = self.Pool.apply_async(detect_first_stage_warpper, (img, i, self.threshold[0], self.ctx))
           # self.index.append(i)
           # return_boxes = self.Pool.map(detect_first_stage_warpper, \
           #         izip(repeat(img), [i]))
           # start_time1 = time()
            #self.t.append(MyThread((img, self.executor1[i], scale, self.threshold[0], self.ctx)))
            #self.t[i].start()
           # i += 1

       # for j in range(i):
       #     self.t[j].join()
       #     return_boxes = self.t[j].return_boxes
           # if return_boxes is not None:
           #     total_boxes.append(return_boxes)


           # end_time1 = time()
            #print 'append time: %.4f'%(end_time1 - start_time1)
        
       # self.Pool.close()
       # self.Pool.join() 
#        print 'first stage time:%.4f'%(end_time - start_time)
        #print 'first stage end'
#        sliced_index = self.slice_index(len(scales))
#        total_boxes = []
#        for batch in sliced_index:
#            local_boxes = self.Pool.map( detect_first_stage_warpper, \
#                    izip(repeat(img), self.PNets[:len(batch)], [scales[i] for i in batch], repeat(self.threshold[0])) )
#            total_boxes.extend(local_boxes)
        
        # remove the Nones 
        total_boxes = [ i for i in total_boxes if i is not None]

        if len(total_boxes) == 0:
            if has_landmark == True:
                return None, None
            else:
                return None
            return None
        
        #print 'before'
        #print len(total_boxes)
        total_boxes = np.vstack(total_boxes)
        
        #print 'after'
        #print total_boxes.shape
        if total_boxes.size == 0:
            if has_landmark == True:
                return None, None
            else:
                return None
            return None

        # merge the detection from first stage
        #print 'global nms:'  + str(total_boxes.shape[0])
        total_boxes.dtype = 'float32'
        pick = gpu_nms(total_boxes[:, 0:5], float(0.7), GPU_ID)
        #pick = nms(total_boxes[:, 0:5], 0.7, 'Union')
        total_boxes = total_boxes[pick]
        #print 'global nms time:%.4f'%(end_time - start_time)
        
        # refine the bboxes
        if first_has_reg == True:
            bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
            bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        #    total_boxes = np.vstack([total_boxes[:, 0]+total_boxes[:, 5] * bbw,
        #                             total_boxes[:, 1]+total_boxes[:, 6] * bbh,
        #                             total_boxes[:, 2]+total_boxes[:, 7] * bbw,
        #                             total_boxes[:, 3]+total_boxes[:, 8] * bbh,
        #                             total_boxes[:, 4]
        #                             ])

         #   total_boxes = total_boxes.T
        total_boxes = self.convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        #return total_boxes
        #############################################
        # second stage
        #############################################
        num_box = total_boxes.shape[0]
        print 'first stage num: %d'%(num_box)

        #return total_boxes
        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, self.width, self.height)
        # (3, 24, 24) is the input shape for RNet
        input_buf = np.zeros((self.second_stage_num, 3, 24, 24), dtype=np.float32)

        #print 'global_first time;%.4f'%(global_first_end_time - global_first_start_time)

        for i in range(num_box):
            if i >= self.second_stage_num:
                break
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
           # tmp = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (24, 24)))
        #    input_buf[i, :, :, :] = adjust_input(mx.image.imresize(tmp, 24, 24).asnumpy())
        #print 'prepare data: %.4f'%(end_time - start_time)

        if len(input_buf) < self.second_stage_num:
            input_buf = np.lib.pad(input_buf, ((self.second_stage_num - len(input_buf), 0), (0,0), (0, 0), (0, 0)), 'constant')
        #print 'first stage :' + str(num_box)
        
        
        if True:
           # start_time = time()
           # data_shape = [("data", input_buf.shape)]
           # input_shapes = dict(data_shape)
           # self.executor2 = self.executor2.reshape(allow_up_sizing = True, **input_shapes)
           # end_time = time()
           # print 'reshape time: %.4f'%(end_time - start_time)
            #executor = self.RNet.simple_bind(ctx = self.ctx, **input_shapes)
            #for key in executor.arg_dict.keys():
            #    if key in self.arg_params2:
            #        self.arg_params2[key].copyto(executor.arg_dict[key])


    #root_path = '/media/disk1/yangfan/wider_faces/mtcnn_data/'

            start_time = time() 
            self.executor2.forward(is_train = False, data = input_buf)
            output1 = self.executor2.outputs[0].asnumpy()
            output2 = self.executor2.outputs[1].asnumpy()
  #  print 'test1'
            end_time = time()
        #    print 'second stage time: %.4f'%(end_time - start_time)
  #  print output.shape
   # print end_time - start_time
        #output = self.RNet.predict(input_buf)
       # print output[:,:]

        # filter the total_boxes with threshold
        if has_reg == True:
            passed = np.where(output1[:, 1] > self.threshold[1])
        else:
        #    print output.shape
            passed = np.where(output[:, 1] > self.threshold[1])
        
        #print output1[:, :]
        total_boxes = total_boxes[passed]


        if total_boxes.size == 0:
            if has_landmark == True:
                return None, None
            else:
                return None
       # print output2
        if has_reg == True:
            total_boxes[:, 4] = output1[passed, 1].reshape((-1,))
            reg = output2[passed]
        else:
            total_boxes[:, 4] = output[passed, 1].reshape((-1,))
            
        # nms
        pick = gpu_nms(total_boxes, 0.7, GPU_ID)
        total_boxes = total_boxes[pick]
        if has_reg == True:
            total_boxes = self.calibrate_box(total_boxes, reg[pick])
        total_boxes = self.convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])
        #print 'second nms:%.4f'%(end_time -start_time)

        #############################################
        # third stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, self.width, self.height)
        # (3, 48, 48) is the input shape for ONet
        input_buf = np.zeros((self.third_stage_num, 3, 48, 48), dtype=np.float32)

        #global_second_end_time = time()
        #print 'global second time:%.4f'%(global_second_end_time - global_second_start_time)

        #global_third_start_time = time()
        
        #start_time = time()
        for i in range(num_box):
            if i >= self.third_stage_num:
                break
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (48, 48)))
        
        if len(input_buf) < self.third_stage_num:
            input_buf = np.lib.pad(input_buf, (self.third_stage_num - len(input_buf, 0), (0, 0), (0, 0), (0, 0)), 'constant')
        print 'second stage :' + str(num_box)
        #end_time = time()
        #print 'prepare data third stage:%.4f'%(end_time - start_time)
        #return total_boxes
        if True:
           # data_shape = [("data", input_buf.shape)]
           # input_shapes = dict(data_shape)
           # executor = self.ONet.simple_bind(ctx = self.ctx, **input_shapes)
           # for key in executor.arg_dict.keys():
           #     if key in self.arg_params3:
           #         self.arg_params3[key].copyto(executor.arg_dict[key])


    #root_path = '/media/disk1/yangfan/wider_faces/mtcnn_data/'
            
           # start_time = time()
           # data_shape = [("data", input_buf.shape)]
           # input_shapes = dict(data_shape)
           # self.executor3 = self.executor3.reshape(allow_up_sizing = True, **input_shapes)
           # end_time = time()

           # print 'reshape time: %.4f'%(end_time - start_time)
         #   start_time = time()
            self.executor3.forward(is_train = False, data = input_buf)
            output1 = self.executor3.outputs[0].asnumpy()
            output2 = self.executor3.outputs[1].asnumpy()
            output3 = self.executor3.outputs[2].asnumpy()
            output3_1 = self.executor3.outputs[3].asnumpy()
            print output3_1.shape
  #  print 'test1'
          #  end_time = time()
        #    print 'third stage time: %.4f'%(end_time - start_time)
  #  print output.shape
   # print end_time - start_time
        #output = self.RNet.predict(input_buf)
        #output = self.ONet.predict(input_buf)

      #  print output
        # filter the total_boxes with threshold
        passed = np.where(output1[:, 1] > self.threshold[2])
        total_boxes = total_boxes[passed]


        if total_boxes.size == 0:
            if has_landmark == True:
                return None, None
            else:
                return None

        total_boxes[:, 4] = output1[passed, 1].reshape((-1,))
        if has_reg == True:
            reg = output2[passed]
        if has_landmark == True:
            points = output3[passed]

        # compute landmark points
        if has_landmark == True:
            bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
            bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
            #for i in range(len(points)):
            for t in range(10):
                if t % 2 == 0:
                    points[:, t] = points[:, t] * bbw + total_boxes[:, 0]
                else:
                    points[:, t] = points[:, t] * bbh + total_boxes[:, 1]
            #points[:, 0:5] = np.expand_dims(total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
            #points[:, 5:10] = np.expand_dims(total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
   #     start_time = time()
        if has_reg == True:
            total_boxes = self.calibrate_box(total_boxes, reg)

        pick = nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[pick]
        if has_landmark == True:
            points = points[pick]
      #  global_end_time = time()
      #  print 'third time %.4f'%(global_end_time - start_time)
      #  print 'global time %.4f'%(global_end_time - global_start_time)
      #  print 'global third time: %.4f'%(global_end_time - global_third_start_time)
        if not self.accurate_landmark:
            if has_landmark == True:
                return total_boxes, points
            else:
                return total_boxes

        #############################################
        # extended stage
        #############################################
        num_box = total_boxes.shape[0]
       # patchw = np.maximum(total_boxes[:, 2]-total_boxes[:, 0]+1, total_boxes[:, 3]-total_boxes[:, 1]+1)
       # patchw = np.round(patchw*0.25)

        # make it even
       # patchw[np.where(np.mod(patchw,2) == 1)] += 1

      #  input_buf = np.zeros((num_box, 15, 24, 24), dtype=np.float32)
      #  for i in range(5):
      #      x, y = points[:, i], points[:, i+5]
      #      x, y = np.round(x-0.5*patchw), np.round(y-0.5*patchw)
      #      [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(np.vstack([x, y, x+patchw-1, y+patchw-1]).T,
                                                          #          width,
                                                          #          height)
      #      for j in range(num_box):
      #          tmpim = np.zeros((tmpw[j], tmpw[j], 3), dtype=np.float32)
      #          tmpim[dy[j]:edy[j]+1, dx[j]:edx[j]+1, :] = img[y[j]:ey[j]+1, x[j]:ex[j]+1, :]
      #          input_buf[j, i*3:i*3+3, :, :] = adjust_input(cv2.resize(tmpim, (24, 24)))
        
        total_boxes_tmp = self.convert_to_square(total_boxes)
        #total_boxes_tmp = total_boxes.copy()
        
        total_boxes_tmp[:, 0:4] = np.round(total_boxes_tmp[:, 0:4])
        if False:
            width = total_boxes_tmp[:, 2] - total_boxes_tmp[:, 0]
            height = total_boxes_tmp[:, 3] - total_boxes_tmp[:, 1]
            total_boxes_tmp[:, 0] += np.round(0.1 * (width)) 
           # index = np.where(total_boxes_tmp[:, 0] < 0)
           # total_boxes_tmp[index, 0] = 0

            total_boxes_tmp[:, 1] += np.round(0.1 * (height)) 
           # index = np.where(total_boxes_tmp[:, 1] < 0)
           # total_boxes_tmp[index, 1] = 0

            total_boxes_tmp[:, 2] -= np.round(0.1 * (width)) 
      #      index = np.where(total_boxes_tmp[:, 2] >= self.width)
      #      total_boxes_tmp[index, 2] = self.width - 1
        
            total_boxes_tmp[:, 3] -= np.round(0.1 * (height)) 
       #     index = np.where(total_boxes_tmp[:, 3] >= self.height)
       #     total_boxes_tmp[index, 3] = self.height - 1

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes_tmp, self.width, self.height)
        input_buf = np.zeros((self.fourth_stage_num, 3, 48, 48), dtype=np.float32)
        input_buf2 = np.zeros((self.fourth_stage_num, 3, 48, 48), dtype=np.float32)
        input_buf3 = np.zeros((self.fourth_stage_num, 3, 64, 64), dtype=np.float32)
        input_buf4 = np.zeros((self.fourth_stage_num, 3, 96, 96), dtype=np.float32)
        #input_buf_rotate = np.zeros((self.fourth_stage_num, 3, 48, 48), dtype=np.float32)

        num_box = len(total_boxes_tmp)
        index = np.zeros((self.fourth_stage_num), dtype = np.uint8)
        for i in range(num_box):
            if i >= self.fourth_stage_num:
                break
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            if tmph[i] > 100 or tmpw[i] > 100:
                index[i] = 1
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i]+1, :]
        #    tmp = img[y[i]: ey[i] + 1, x[i]: ex[i] + 1, :]
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (48, 48)))
           # height = tmp.shape[0]
           # width = tmp.shape[1]
           # if height > 80 or width > 80:
           #     tmp = cv2.resize(tmp, (height / 8, width / 8))
            input_buf2[i, :, :, :] = adjust_input(cv2.resize(tmp, (48, 48)))
            input_buf3[i, :, :, :] = adjust_input(cv2.resize(tmp, (64, 64)))
            input_buf4[i, :, :, :] = adjust_input(cv2.resize(tmp, (96, 96)))


            #input_buf_rotate[i] = input_buf[i].copy()
        
        if len(input_buf) < self.fourth_stage_num:
            input_buf = np.lib.pad(input_buf, (0, self.fourth_stage_num - len(input_buf)), (0, 0), (0, 0), (0, 0), 'constant')
            input_buf2 = np.lib.pad(input_buf2, (0, self.fourth_stage_num - len(input_buf2)), (0, 0), (0, 0), (0, 0), 'constant')
            input_buf3 = np.lib.pad(input_buf3, (0, self.fourth_stage_num - len(input_buf3)), (0, 0), (0, 0), (0, 0), 'constant')
            input_buf4 = np.lib.pad(input_buf4, (0, self.fourth_stage_num - len(input_buf4)), (0, 0), (0, 0), (0, 0), 'constant')
        #print 'third stage :' + str(num_box)
        
        #print 'prepare data fourth stage: %.4f'%(end_time - start_time) 

        self.executor4_0.forward(is_train = False, data = input_buf)
        output0_0 = self.executor4_0.outputs[0].asnumpy()
        output0_1 = self.executor4_0.outputs[1].asnumpy()
        output0_2 = self.executor4_0.outputs[2].asnumpy()
        output0_0 *= 90.
        output0_1 *= 90.
        output0_2 *= 90.
        #for t in range(input_buf_rotate.shape[0]):
        #     if output0_2[t] > 15 or output0_2[t] < -15:
        #         tmp_img = input_buf_rotate[t].transpose((1, 2, 0))
        #         tmp_img = tmp_img / 0.0078125 + 127.5
        #         angle = output0_2[t]
        #         scale = 0.9
        #         rotateMat = cv2.getRotationMatrix2D((48 / 2, 48 / 2), angle, scale)
        #         rotateImg = cv2.warpAffine(tmp_img, rotateMat, (48, 48))
                
        #         rotateImg = rotateImg.transpose((2, 0, 1))
        #         rotateImg = (rotateImg - 127.5) * 0.007812
        #         input_buf_rotate[t, :, :, :] = rotateImg

        self.executor4_1.forward(is_train = False, data = input_buf)
        self.executor4_3.forward(is_train = False, data = input_buf)
       # self.executor4_4.forward(is_train = False, data = input_buf2)

        output1 = self.executor4_1.outputs[0].asnumpy()
        output2 = self.executor4_3.outputs[0].asnumpy()
       # output2_1 = self.executor4_4.outputs[0].asnumpy()


       # pick = np.argmax(output2, axis = 1)
       # pick = (pick * 10 + 5) / 100.0
       # pick = np.reshape(pick, (pick.shape[0], 1))
       # output4 = self.executor4.outputs[3].asnumpy()

        #print 'cnn fourth stage: %.4f'%(end_time - start_time) 
       # output = self.LNet.predict(input_buf)
        
        if num_box > self.fourth_stage_num:
            num_box = self.fourth_stage_num

       # for tt in range(num_box):
       #     if index[tt] == 0:
       #         output2[tt, :] = output2_1[tt, :]


        total_boxes = np.hstack([total_boxes_tmp[0: num_box], output1[0:num_box, 0:1], output2[0:num_box, 1:2]]) 

        #return total_boxes[0:num_box], points[0: num_box]
        self.executor4_2.forward(is_train = False, data = input_buf)
        output3 = self.executor4_2.outputs[0].asnumpy()
        
        self.executor4_5.forward(is_train = False, data = input_buf)
        output4 = self.executor4_5.outputs[0].asnumpy()
       # print 'cnn fifth stage: %.4f'%(end_time - start_time) 

 #       for i in range(101):
 #           output1[0:num_box, 0] += i * output1[0:num_box, i]
        #pick = np.argmax(output1, axis = 1)

        
        #pick = pick * 10
        #pick = np.reshape(pick, (pick.shape[0], 1))
        total_boxes = np.hstack([total_boxes[0: num_box], output3[0:num_box, 1:2]])
        
       # total_boxes[0:num_box, 5] = output1[:, 0]

        self.executor5.forward(is_train = False, data = input_buf2)
        output1 = self.executor5.outputs[0].asnumpy()
        
        age = np.zeros((num_box, 1), dtype = np.float32)
        for i in range(num_box):
            age[i] = output1[i][0] * 1.0 + output1[i][1] * 5.0 + output1[i][2] * 11 + output1[i][3] * 16 + output1[i][4] * 23 + output1[i][5] * 28 + output1[i][6] * 33 + output1[i][7] * 40

        pick = np.argmax(output1, axis = 1)
        #pick = (pick - 1) * 5 + 10
        pick = np.reshape(pick, (pick.shape[0], 1))


        output1 = np.max(output1, axis = 1)
        output1 = np.reshape(output1, (output1.shape[0], 1))

        total_boxes = np.hstack([total_boxes[0: num_box], output1[0:num_box], pick[0:num_box], age, output4[0:num_box,1:2], output0_0[0:num_box], output0_1[0:num_box], output0_2[0:num_box]])

        self.executor_true.forward(is_train = False, data = input_buf3)
        output1 = self.executor_true.outputs[0].asnumpy()

        self.executor_clear.forward(is_train = False, data = input_buf4)
        output2 = self.executor_clear.outputs[0].asnumpy()
        

        total_boxes = np.hstack([total_boxes[0: num_box], output1[0: num_box, 1:2], output2[0: num_box, 1:2]])

        return total_boxes[0: num_box], points[0: num_box]
        
        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))

        for k in range(5):
            # do not make a large movement
            tmp_index = np.where(np.abs(output[k]-0.5) > 0.35)
            output[k][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(points[:, k] - 0.5*patchw) + output[k][:, 0]*patchw
            pointy[:, k] = np.round(points[:, k+5] - 0.5*patchw) + output[k][:, 1]*patchw

        points = np.hstack([pointx, pointy])
        points = points.astype(np.int32)

        return total_boxes, points

    def list2colmatrix(self, pts_list):
        """
            convert list to column matrix
        Parameters:
        ----------
            pts_list:
                input list
        Retures:
        -------
            colMat: 

        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        colMat = np.matrix(colMat).transpose()
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        """
            find transform between shapes
        Parameters:
        ----------
            from_shape: 
            to_shape: 
        Retures:
        -------
            tran_m:
            tran_b:
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0]/2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0]/2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(self, img, points, desired_size=256, padding=0):
        """
            crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
            desired_size: default 256
            padding: default 0
        Retures:
        -------
            crop_imgs: list, n
                cropped and aligned faces 
        """
        crop_imgs = []
        for p in points:
            shape  =[]
            for k in range(len(p)/2):
                shape.append(p[k])
                shape.append(p[k+5])

            if padding > 0:
                padding = padding
            else:
                padding = 0
            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

            from_points = []
            to_points = []

            for i in range(len(shape)/2):
                x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
                y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
                to_points.append([x, y])
                from_points.append([shape[2*i], shape[2*i+1]])

            # convert the points to Mat
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)

            # compute the similar transfrom
            tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

            from_center = [(shape[0]+shape[2])/2.0, (shape[1]+shape[3])/2.0]
            to_center = [0, 0]
            to_center[1] = desired_size * 0.4
            to_center[0] = desired_size * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1*angle, scale)
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
            crop_imgs.append(chips)

        return crop_imgs

