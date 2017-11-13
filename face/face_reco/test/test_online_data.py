# coding:utf-8
# FACE verifycation by this model

import os
import math
import numpy as np

import sys
sys.path.insert(0, '/media/disk1/yangfan/face_reco/caffe-face/python')
sys.path.insert(0, '/media/disk1/yangfan/face_reco/caffe-face/lib')
sys.path.insert(0, '/media/disk1/yangfan/opencv-2.4.13.2/lib/')
import cv2
import caffe

#import matplotlib.pyplot as plt
import scipy 
import time
    
from detect_landmark import get_executor  

import mxnet as mx

executor = get_executor()

sys.path.insert(0, '/home/yangfan/face_detection/mxnet/example/mtcnn/implement/online_model4/test_with_reg')
from get_id_ip import detect_face_impl



if False:
    le_prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/le/face_deploy.prototxt"
    le_caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/le/le_iter_40000.caffemodel"
    le_net = caffe.Net(le_prototxt, le_caffemodel, caffe.TEST)

    re_prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/re/face_deploy.prototxt"
    re_caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/re/re_iter_500000.caffemodel"
    re_net = caffe.Net(re_prototxt, re_caffemodel, caffe.TEST)
    
    no_prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/no/face_deploy.prototxt"
    no_caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/no/no_iter_500000.caffemodel"
    no_net = caffe.Net(no_prototxt, no_caffemodel, caffe.TEST)

    lm_prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/lm/face_deploy.prototxt"
    lm_caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/lm/lm_iter_500000.caffemodel"
    lm_net = caffe.Net(lm_prototxt, lm_caffemodel, caffe.TEST)

    rm_prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/rm/face_deploy.prototxt"
    rm_caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/multi_patch/rm/rm_iter_500000.caffemodel"
    rm_net = caffe.Net(rm_prototxt, rm_caffemodel, caffe.TEST)

    disc_prototxt = r"/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/feature_selection/face_deploy.prototxt"
    disc_caffemodel = r"/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/feature_selection/face_train_test_new_iter_300000.caffemodel"
    disc_net = caffe.Net(disc_prototxt, disc_caffemodel, caffe.TEST)


def get_le_feature(le_inputs):
    
    img_blobinp = np.array(le_inputs)
    le_net.blobs['data'].reshape(*img_blobinp.shape)
    le_net.blobs['data'].data[...] = img_blobinp
    le_net.blobs['data'].data.shape
    le_net.forward()  #go through the LCNN network
    return le_net.blobs['fc5'].data
        #feature = net.blobs['fc6_new'].data    #feature is from eltwise_fc1 layer

def get_re_feature(re_inputs):
    img_blobinp = np.array(re_inputs)
    re_net.blobs['data'].reshape(*img_blobinp.shape)
    re_net.blobs['data'].data[...] = img_blobinp
    re_net.blobs['data'].data.shape
    re_net.forward()  #go through the LCNN network
    return re_net.blobs['fc5'].data
    

def get_no_feature(no_inputs):
    
    img_blobinp = np.array(no_inputs)
    no_net.blobs['data'].reshape(*img_blobinp.shape)
    no_net.blobs['data'].data[...] = img_blobinp
    no_net.blobs['data'].data.shape
    no_net.forward()  #go through the LCNN network
    return no_net.blobs['fc5'].data

def get_lm_feature(lm_inputs):
    img_blobinp = np.array(lm_inputs)
    lm_net.blobs['data'].reshape(*img_blobinp.shape)
    lm_net.blobs['data'].data[...] = img_blobinp
    lm_net.blobs['data'].data.shape
    lm_net.forward()  #go through the LCNN network
    return lm_net.blobs['fc5'].data
    

def get_rm_feature(rm_inputs):
    img_blobinp = np.array(rm_inputs)
    rm_net.blobs['data'].reshape(*img_blobinp.shape)
    rm_net.blobs['data'].data[...] = img_blobinp
    rm_net.blobs['data'].data.shape
    rm_net.forward()  #go through the LCNN network
    return rm_net.blobs['fc5'].data

def get_disc_feature(inputs):
    img_blobinp = np.array(inputs)
    disc_net.blobs['data'].reshape(*img_blobinp.shape)
    disc_net.blobs['data'].data[...] = img_blobinp
    disc_net.blobs['data'].data.shape
    disc_net.forward()  #go through the LCNN network
    return disc_net.blobs['fc6_new'].data
    

def test_centerloss():
    
    total_time = 0
    prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/face_deploy.prototxt"
    #prototxt = r'/media/disk1/yangfan/face_reco/caffe-face/face_example/light_cnn/LCNN_deploy.prototxt'

    #prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/inception_v2/test.prototxt"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/backup/face_train_test_new_iter_50000.caffemodel"
    #caffemodel = r'/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_inception_v2_iter_100000.caffemodel'
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_without_centerloss_iter_60000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_14w_norm_iter_50000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_14w_norm_0.1centerloss_0.1triplet-centerloss_iter_10000.caffemodel"

#    caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_1600W_without_centerloss_iter_60000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_14w_norm_1.0centerloss_0.1triplet-centerloss_iter_20000.caffemodel"

    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_14w_with_centerloss_iter_40000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_4w_256_with_norm_0.5centerlossnorm_iter_10000.caffemodel"
#    caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_42_with_norm_0.1centerlossnorm_0.001centerlosswithoutnorm_iter_30000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_4w_with_norm_0.1centerlossnorm_iter_10000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_14w_with_0.002centerloss_iter_30000.caffemodel"
    #face_train_test_new_4w_256_with_norm_0.5centerlossnorm_iter_10000.caffemodel

    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_14w_with_0.005centerloss_0.005tripletloss_iter_30000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_4w_with_0.008_0.05centerloss_iter_30000.caffemodel"
    caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_video_20w_with_0.002centerloss_iter_160000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_video_with_0.002centerloss_iter_40000.caffemodel"
    #caffemodel = r'/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_17w_256_without_centerloss_iter_10000.caffemodel'
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_4w_with_0.008_0.05centerloss_iter_10000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_14w_with_0.01centerloss_iter_30000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_14w_with_centerloss_iter_30000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_4w_with_0.008centerloss_0.008tripletloss_iter_30000.caffemodel"

    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_without_centerloss_iter_60000.caffemodel"
   # caffemodel = r'/media/disk1/yangfan/face_reco/caffe-face/face_example/light_cnn/face_iter_2300000.caffemodel'
   # caffemodel = r'/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/face_train_test_new_org_iter_10000.caffemodel'
    #caffemodel = r'/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/face_train_test_new_org_iter_1000.caffemodel'
    th = 0.35

    if not os.path.isfile(caffemodel):
        print ("caffemodel not found!")
    
    caffe.set_mode_gpu()
    caffe.set_device(7)

    net = caffe.Net(prototxt,caffemodel,caffe.TEST)

#print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
#root_path = '/media/mfs/fordata/td_hg2_web_server/yangfan/face_reco/online_data/2133393359'
#data_path = '/media/disk1/yangfan/lfw/lfw-deepfunneled'

    #data_path = '/media/mfs/fordata/td_hg2_web_server/yangfan/face_reco/online_data/users'
    data_path = '/home/yangfan/face_reco/online_data/centerloss_model_aligned/pair_files'

    feats = np.zeros((2, 512),dtype=np.float32)
    #feats = np.zeros((2, 512),dtype=np.float32)
    #feats = np.zeros((2,  1024),dtype=np.float32)

    right_count = 0
    total_count = 0
    file_path = 'pairs.txt'
   # file_path = 'pairs_anotate.txt'
    f = open(file_path)
    
    output_f = open('feature1.txt', 'w')
    output_f1 = open('online_face_result1.txt', 'w')
    output_f2 = open('check_result.txt', 'w')
    lines = f.readlines()
    file_list = []
    for line in lines:
        s = line.strip('\n').split('\t ')
        if total_count != 0:
            print right_count / float(total_count) 
        label = 2
        if len(s) == 3:
            dirname = s[0]
            filename1 = os.path.join(data_path, dirname, s[1])
            filename2 = os.path.join(data_path, dirname, s[2])
            label = 1
        elif len(s) == 2:
            filename1 = os.path.join(data_path, s[0])
            filename2 = os.path.join(data_path, s[1])
            label = 0

        if label == 2:
            continue
      
        if not os.path.exists(filename1) or not os.path.exists(filename2):
            continue
            
        input1 = cv2.imread(filename1)   #read face image
        input2 = cv2.imread(filename2)
        #img_org1 = cv2.imread(filename1)
        #img_org2 = cv2.imread(filename2)
        inputs = [] 
        le_inputs = [] 
        re_inputs = [] 
        no_inputs = [] 
        lm_inputs = [] 
        rm_inputs = [] 
        for cur_input in [input1, input2]:
            height = cur_input.shape[0]
            width = cur_input.shape[1]
            input_org = cur_input.copy()
            face_img = cv2.resize(cur_input, (64, 64))
            face_img = face_img.transpose((2, 0, 1))
            face_img = np.multiply(face_img - 127.5, 1.0 / 128.0)
            executor.forward(is_train = False, data = mx.nd.array([face_img]))
            points = executor.outputs[0].asnumpy()[0]
            angle = math.atan((points[3] - points[1]) / (points[2] - points[0])) * 180. / 3.14159
                
            point_le = np.array([[points[0] * width], [points[1] * height], [1]])
            point_re = np.array([[points[2] * width], [points[3] * height], [1]])
            point_no = np.array([[points[4] * width], [points[5] * height], [1]])
            point_lm = np.array([[points[6] * width], [points[7] * height], [1]])
            point_rm = np.array([[points[8] * width], [points[9] * height], [1]])

            img_tmp = input_org.copy()
            if angle > 30 or angle < -30:
                center_x = points[4] * width
                center_y = points[5] * height
                scale = 1.0
                rotateMat = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)

                img_tmp = cv2.warpAffine(img_tmp, rotateMat, (width, height), borderValue = 0)

                point_le = np.dot(rotateMat, point_le)
                point_re = np.dot(rotateMat, point_re)
                point_no = np.dot(rotateMat, point_no)
                point_lm = np.dot(rotateMat, point_lm)
                point_rm = np.dot(rotateMat, point_rm)

            scale_x = float(45) / (point_re[0][0] - point_le[0][0])
            width_new = int(width * scale_x)
            height_new = int(height * scale_x)

            img_tmp = cv2.resize(img_tmp, (width_new, height_new))
                
            face_img = np.zeros((128, 128, 3), dtype = np.uint8)
            point_le = point_le * scale_x
            point_re = point_re * scale_x
            point_no = point_no * scale_x
            point_lm = point_lm * scale_x
            point_rm = point_rm * scale_x

            point_le = point_le.astype(np.int32)
            point_re = point_re.astype(np.int32)
            point_no = point_no.astype(np.int32)
            point_lm = point_lm.astype(np.int32)
            point_rm = point_rm.astype(np.int32)

            if point_le[0][0] <= 0 or point_re[0][0] <= 0 or point_le[0][0] >= img_tmp.shape[1] or point_re[0][0] >= img_tmp.shape[1] or point_le[1][0] >= img_tmp.shape[0] or point_re[1][0] >= img_tmp.shape[1] or point_le[0][0] > img_tmp.shape[1] or point_le[1][0] <= 0 or point_re[1][0] <= 0:
                print filename1
                print filename2
                break
            src_start_x = 0
            src_start_y = 0
            src_end_x = 128
            src_end_y = 128

            start_x = point_le[0][0] - 42
                
            if start_x < 0:
                src_start_x = 42 - point_le[0][0]
                start_x = 0

            start_y = point_le[1][0] - 50
            if start_y < 0:
                src_start_y = 50 - point_le[1][0]
                start_y = 0
                
            end_x = point_le[0][0] + 86 

            if end_x >= img_tmp.shape[1]:
                src_end_x = 128 - end_x + img_tmp.shape[1] - 1
                end_x = img_tmp.shape[1] - 1

            end_y = point_le[1][0] + 78
            if end_y >= img_tmp.shape[0]:
                src_end_y = 128 - end_y + img_tmp.shape[0] - 1
                end_y = img_tmp.shape[0] - 1

            face_img[src_start_y: src_end_y, src_start_x: src_end_x] = img_tmp[start_y: end_y, start_x: end_x]

            #face_img = cv2.resize(face_img[10:-10, 10:-10, :], (128, 128))
            
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)


            #cv2.imwrite(os.path.join('result', os.path.basename(filename1).strip('.jpg') + '%d.jpg'%(len(inputs))), face_img)
            
            le_x = 42
            le_y = 50
            re_x = 87 
            re_y = 50
            no_x = 42 - point_le[0][0] + point_no[0][0] 
            no_y = 50 - point_le[1][0] + point_no[1][0]
            lm_x = 42 - point_le[0][0] + point_lm[0][0] 
            lm_y = 50 - point_le[1][0] + point_lm[1][0] 
            rm_x = 42 - point_le[0][0] + point_rm[0][0] 
            rm_y = 50 - point_le[1][0] + point_rm[1][0] 
            
            pad = 24
            top_le_x = le_x - 24
            top_le_y = le_y - 24
            top_re_x = re_x - 24
            top_re_y = re_y - 24

            top_no_x = no_x - 24
            top_no_y = no_y - 24
            top_lm_x = lm_x - 24
            top_lm_y = lm_y - 24
            top_rm_x = rm_x - 24
            top_rm_y = rm_y - 24

            try:
                le_img = face_img.copy()[top_le_y: top_le_y + 48, top_le_x: top_le_x + 48]
                le_img = cv2.resize(le_img, (64, 64))
                re_img = face_img.copy()[top_re_y: top_re_y + 48, top_re_x: top_re_x + 48]
                re_img = cv2.resize(re_img, (64, 64))
                no_img = face_img.copy()[top_no_y: top_no_y + 48, top_no_x: top_no_x + 48]
                no_img = cv2.resize(no_img, (64, 64))
                lm_img = face_img.copy()[top_lm_y: top_lm_y + 48, top_lm_x: top_lm_x + 48]
                lm_img = cv2.resize(lm_img, (64, 64))
                rm_img = face_img.copy()[top_rm_y: top_rm_y + 48, top_rm_x: top_rm_x + 48]
                rm_img = cv2.resize(rm_img, (64, 64))
            except Exception,e:
                print str(e)
                continue
            

            if False:
                if not os.path.exists('le'):
                    os.makedirs('le')
                cv2.imwrite('le/le_%d.jpg'%(total_count), le_img)

                if not os.path.exists('re'):
                    os.makedirs('re')
                cv2.imwrite('re/re_%d.jpg'%(total_count), re_img)


                if not os.path.exists('no'):
                    os.makedirs('no')
                cv2.imwrite('no/no_%d.jpg'%(total_count), no_img)


                if not os.path.exists('lm'):
                    os.makedirs('lm')
                cv2.imwrite('lm/lm_%d.jpg'%(total_count), lm_img)

                if not os.path.exists('rm'):
                    os.makedirs('rm')
                cv2.imwrite('rm/rm_%d.jpg'%(total_count), rm_img)

            le_inputs.append((le_img[np.newaxis, :, :] - 127.5) / 128.) 
            re_inputs.append((re_img[np.newaxis, :, :] - 127.5) / 128.) 
            no_inputs.append((no_img[np.newaxis, :, :] - 127.5) / 128.) 
            lm_inputs.append((lm_img[np.newaxis, :, :] - 127.5) / 128.) 
            rm_inputs.append((rm_img[np.newaxis, :, :] - 127.5) / 128.) 
            #face_img = face_img.resize(face_img)
            #face_img = cv2.resize(face_img, (224, 224))


            inputs.append((face_img[np.newaxis, :, :] - 127.5) / 128.)
            #face_img = face_img.transpose((2, 0, 1))
            #inputs.append((face_img - 127.5) / 128.)
        
        if len(inputs) != 2:
            continue
        start = time.time()
        #img_blobinp = np.array([(input1[np.newaxis, :, :] - 127.5) / 128.0, (input2[np.newaxis, :, :] - 127.5) / 128.0])    #divide 255.0 ,make input is between 0-1
        img_blobinp = np.array(inputs)
        net.blobs['data'].reshape(*img_blobinp.shape)
        net.blobs['data'].data[...] = img_blobinp
        net.blobs['data'].data.shape
        start = time.time()
        net.forward()  #go through the LCNN network
        end = time.time()
        total_time += end - start
        #feature = net.blobs['fc6_new'].data    #feature is from eltwise_fc1 layer
        #feature = net.blobs['norm1'].data    #feature is from eltwise_fc1 layer
        feature = net.blobs['fc5'].data    #feature is from eltwise_fc1 layer
        #feature = net.blobs['eltwise_fc1'].data
        #feature = net.blobs[r'pool5/7x7_s1'].data
        last_feature = feature

        #middle_feature = net.blobs['res5_6'].data

        if False:

            le_feature = get_le_feature(le_inputs)
            re_feature = get_re_feature(re_inputs)
            no_feature = get_no_feature(no_inputs)
            lm_feature = get_lm_feature(lm_inputs)
            rm_feature = get_rm_feature(rm_inputs)

            all_features = np.concatenate((feature, le_feature, re_feature, no_feature, lm_feature, rm_feature), axis = 1)

            last_feature = get_disc_feature(all_features)

            print 'test'

        #feats[0,:] = feature.reshape((512))[0:256]
  #      feats[0,:] = feature.reshape((1024))[0:512]
  #      feats[1,:] = feature.reshape((1024))[512:1024]
        #feature = net.blobs['fc'].data    #feature is from eltwise_fc1 layer
        #feats[0,:] = feature[0].reshape((512))


        if filename1 not in file_list:
             output_f.write(filename1 + ' \n')
             for i in range(512):
                 output_f.write(str(last_feature[0][i]) + ' ')
##             
##             for i in range(256):
##                output_f.write(str(feature[1][i]) + ' ')
##            
             output_f.write('\n')
             file_list.append(filename1)
##
#
        if filename2 not in file_list:
             output_f.write(filename2 + ' \n')
             for i in range(512):
                 output_f.write(str(last_feature[1][i]) + ' ')
##
##             for i in range(256):
##                output_f.write(str(feature[1][i]) + ' ')
##
             output_f.write('\n')
             file_list.append(filename2)

#        for i in range(feature1.shape[1]):
#            if feature1[0][i] > 6.0:
#                
#                if not os.path.exists(os.path.join('feature_result', '%d'%(i))):
#                    os.makedirs(os.path.join('feature_result', '%d'%(i)))
#                
#                save_image = os.path.join('feature_result', '%d'%(i), '%d.jpg'%(len(os.listdir(os.path.join('feature_result', '%d'%(i))))))
#
#                cv2.imwrite(save_image, img_org)
#
#            if feature1[0][i] < -6.0:
#                
#                if not os.path.exists(os.path.join('feature_result1', '%d'%(i))):
#                    os.makedirs(os.path.join('feature_result1', '%d'%(i)))
#                
#                save_image = os.path.join('feature_result1', '%d'%(i), '%d.jpg'%(len(os.listdir(os.path.join('feature_result1', '%d'%(i))))))
#                
#                cv2.imwrite(save_image, img_org)
#

        #input2_tmp = np.zeros((input2.shape[0], input2.shape[1], 1),dtype=np.uint8)
        #cv2.cvtColor(input2, input2_tmp, cv2.COLOR_BGR2GRAY)
        #input2 = input2_tmp

#        for i in range(feature2.shape[1]):
#            if feature2[0][i] > 6.0:
#                
#                if not os.path.exists(os.path.join('feature_result', '%d'%(i))):
#                    os.makedirs(os.path.join('feature_result', '%d'%(i)))
#                
#                save_image = os.path.join('feature_result', '%d'%(i), '%d.jpg'%(len(os.listdir(os.path.join('feature_result', '%d'%(i))))))
#                
#                cv2.imwrite(save_image, img_org)
#
#            if feature2[0][i] < -6.0:
#                
#                if not os.path.exists(os.path.join('feature_result1', '%d'%(i))):
#                    os.makedirs(os.path.join('feature_result1', '%d'%(i)))
#                
#                save_image = os.path.join('feature_result1', '%d'%(i), '%d.jpg'%(len(os.listdir(os.path.join('feature_result1', '%d'%(i))))))
#                
#                cv2.imwrite(save_image, img_org)
        
   #     similarity = 1 - scipy.spatial.distance.cosine(feats[0, :], feats[1, :])
        #similarity = np.sqrt(np.sum(np.square(feats[0, :] - feats[1, :])))
#        similarity = np.sqrt(np.sum(np.square(last_feature[0, :] - last_feature[1, :])))
#        last_feature[0, :][last_feature[0, :] < 0] = 0
#        last_feature[1, :][last_feature[1, :] < 0] = 0
        similarity = 1 - scipy.spatial.distance.cosine(last_feature[0, :], last_feature[1: ])

        th = 0.28 
        if similarity > th and label == 1:
             right_count += 1
           #  print np.sum(np.square(last_feature[0, :]))
           #  print np.sum(np.square(last_feature[1, :]))

        elif similarity <= th and label == 0:
             right_count += 1
           #  print np.sum(np.square(last_feature[0, :]))
           #  print np.sum(np.square(last_feature[1, :]))

        else:
            
             print filename1
             print filename2
             print similarity
#             print np.sum(np.square(last_feature[0, :]))
 #            print np.sum(np.square(last_feature[1, :]))
            # print middle_feature
    #    if  True:
            
             #last_feature[0, :] = last_feature[0, :] / np.sqrt(np.sum(np.square(last_feature[0, :])))

             #last_feature[1, :] = last_feature[1, :] / np.sqrt(np.sum(np.square(last_feature[1, :])))

             #output_f2.write(filename1 + ' ' + filename2 + '\n')

            # last_feature[0, :] = last_feature[0, :] * last_feature[1, :]

            # last_feature[0, :][last_feature[0, :] < 0] = 0

            # for i in range(512):
 #               output_f2.write(str(last_feature[0, i]) + ' ')
#
  #           output_f2.write('\n')
   #          output_f2.write(str(similarity) + '\n')

            # for i in range(512):
            #    output_f2.write(str(last_feature[1, i]) + ' ')

   #          output_f2.write('\n\n')
            
            
        total_count += 1
        output_f1.write(str(label) + ' ' + str(similarity) + '\n')

    print right_count / float(total_count)
    print total_count
    print right_count
    print total_time

def test_centerloss_17w():
    prototxt = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/face_deploy.prototxt"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_1600W_without_centerloss_iter_40000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_14w_with_centerloss_iter_10000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_14w_with_centerloss_iter_30000.caffemodel"
    #caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_video_with_0.002centerloss_iter_20000.caffemodel"
    caffemodel = r"/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_video_20w_with_0.002centerloss_iter_160000.caffemodel"
#    prototxt = r'/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/face_deploy.prototxt'
#caffemodel = r"models/_iter_3560000.caffemodel"
#caffemodel = r'models/LightenedCNN_B.caffemodel'
   # caffemodel = r'/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/face_train_test_new_org_iter_10000.caffemodel'
    #caffemodel = r'/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/face_train_test_new_org_iter_1000.caffemodel'
    th = 0.32

    if not os.path.isfile(caffemodel):
        print ("caffemodel not found!")
    caffe.set_mode_gpu()
    caffe.set_device(3)
    net = caffe.Net(prototxt,caffemodel,caffe.TEST)
    feats = np.zeros((2, 512),dtype=np.float32)

    data_path = '/media/mfs/fordata/td_hg2_web_server/yangfan/face_reco/online_data_for_anotation/videos_one_person'
    f = open('pairs_anotate.txt')
    output_f = open('feature.txt', 'w')
    output_f1 = open('online_face_result_17w_1.txt', 'w')
    lines = f.readlines()
    f.close()
    total_count = 0
    right_count = 0
    file_list = []
    for line in lines:
        if total_count != 0:
            print float(right_count) / total_count
        s = line.strip('\n').split(' ')
        label = 2
        if len(s) == 3:
            filename1 = os.path.join(data_path, s[0], s[1])
            filename2 = os.path.join(data_path, s[0], s[2])
            label = 1
        if len(s) == 4:
            filename1 = os.path.join(data_path, s[0], s[1])
            filename2 = os.path.join(data_path, s[2], s[3])
            label = 0
        
        if not os.path.exists(filename1):
            print filename1 + ' no exist'
            continue
        if not os.path.exists(filename2):
            print filename2 + ' no exists'
            continue
        f1 = open(filename1)
        f2 = open(filename2)
        inputs = []
        for filename in [filename1, filename2]:
            f = open(filename)
            try:
                image = f.read()
                img = cv2.imread(filename)
            except Exception,e :
                print str(e)
                print filename + 'test'
                break

            height_org = img.shape[0]
            width_org = img.shape[1]
            try:
                results = detect_face_impl(image)
            except Exception, e:
                print str(e)
                print filename
                break
            if results == None:
                print filename
                print 'None Face'
                break
            if len(results) != 1:
                break

            i = 0 

            if True:
                result = results[0]
                
               # if abs(result[-1]) > 45 or abs(result[-2]) > 45 or abs(result[-3]) > 45:
               #     break
                print result
                img_tmp = img.copy()
                width = result[2]
                height = result[3]
                x = result[0] -  width * 0.02
                y = result[1] - height * 0.02

                width += width * 0.05
                height += height * 0.05
                
                x = int(x)
                y = int(y)
                width = int(width)
                height = int(height)

                if x < 0: 
                    x = 0

                if y < 0:
                    y = 0
                
                if width + x >= width_org:
                    width = width_org - x

                if height + y >= height_org:
                    height = height_org - y
                
                #if abs(height - width) > 10:
                #    continue

                face_img = img[y: y + height, x: x + width]

                face_img = cv2.resize(face_img, (64, 64))
                face_img = face_img.transpose((2, 0, 1))
                face_img = np.multiply(face_img - 127.5, 1.0 / 128.0)
                executor.forward(is_train = False, data = mx.nd.array([face_img]))
                points = executor.outputs[0].asnumpy()[0]
                angle = math.atan((points[3] - points[1]) / (points[2] - points[0])) * 180. / 3.14159
                
                point_le = np.array([[points[0] * width + x], [points[1] * height + y], [1]])
                point_re = np.array([[points[2] * width + x], [points[3] * height + y], [1]])

                if angle > 3 or angle < -3:

                    center_x = points[4] * width + x
                    center_y = points[5] * height + y
                    scale = 1.0
                    rotateMat = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
                    img_tmp = cv2.warpAffine(img_tmp, rotateMat, (width_org, height_org), borderValue = 0)

                    point_le = np.dot(rotateMat, point_le)
                    point_re = np.dot(rotateMat, point_re)

                scale_x = float(45) / (point_re[0][0] - point_le[0][0])
                width_new = int(width_org * scale_x)
                height_new = int(height_org * scale_x)

                img_tmp = cv2.resize(img_tmp, (width_new, height_new))
                
                face_img = np.zeros((128, 128, 3), dtype = np.uint8)

                point_le = point_le * scale_x
                point_re = point_re * scale_x

                point_le = point_le.astype(np.int32)
                point_re = point_re.astype(np.int32)

                if point_le[0][0] <= 0 or point_re[0][0] <= 0 or point_le[0][0] >= img_tmp.shape[1] or point_re[0][0] >= img_tmp.shape[1] or point_le[1][0] >= img_tmp.shape[0] or point_re[1][0] >= img_tmp.shape[1] or point_le[0][0] > img_tmp.shape[1] or point_le[1][0] <= 0 or point_re[1][0] <= 0:
                    break
                src_start_x = 0
                src_start_y = 0
                src_end_x = 128
                src_end_y = 128

                start_x = point_le[0][0] - 42
                
                if start_x < 0:
                    src_start_x = 42 - point_le[0][0]
                    start_x = 0

                start_y = point_le[1][0] - 50
                if start_y < 0:
                    src_start_y = 50 - point_le[1][0]
                    start_y = 0
                
                end_x = point_le[0][0] + 86 

                if end_x >= img_tmp.shape[1]:
                    src_end_x = 128 - end_x + img_tmp.shape[1] - 1
                    end_x = img_tmp.shape[1] - 1

                end_y = point_le[1][0] + 78
                if end_y >= img_tmp.shape[0]:
                    src_end_y = 128 - end_y + img_tmp.shape[0] - 1
                    end_y = img_tmp.shape[0] - 1

                face_img[src_start_y: src_end_y, src_start_x: src_end_x] = img_tmp[start_y: end_y, start_x: end_x]
                
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        #        cv2.imwrite(os.path.join('result', os.path.basename(filename1).strip('.jpg') + '%d.jpg'%(len(inputs))), face_img)
                inputs.append((face_img[np.newaxis, :, :] - 127.5) / 128.)
         
        if len(inputs) != 2:
            continue
        #img_blobinp = np.array([(input1[np.newaxis, :, :] - 127.5) / 128.0, (input2[np.newaxis, :, :] - 127.5) / 128.0])    #divide 255.0 ,make input is between 0-1
        img_blobinp = np.array(inputs)
        net.blobs['data'].reshape(*img_blobinp.shape)
        net.blobs['data'].data[...] = img_blobinp
        net.blobs['data'].data.shape
        net.forward()  #go through the LCNN network
        #feature = net.blobs['fc6_new'].data    #feature is from eltwise_fc1 layer
        #feature = net.blobs['norm1'].data    #feature is from eltwise_fc1 layer
        feature = net.blobs['fc5'].data    #feature is from eltwise_fc1 layer
        #feats[0,:] = feature.reshape((512))[0:256]
        feats[0,:] = feature.reshape((1024))[0:512]
        feats[1,:] = feature.reshape((1024))[512:1024]
        #feature = net.blobs['fc'].data    #feature is from eltwise_fc1 layer
        #feats[0,:] = feature[0].reshape((512))

        if filename1 not in file_list:
             output_f.write(filename1 + ' \n')
             for i in range(512):
                 output_f.write(str(feats[0][i]) + ' ')
##             
##             for i in range(256):
##                output_f.write(str(feature[1][i]) + ' ')
##            
             output_f.write('\n')
             file_list.append(filename1)
##
#
        if filename2 not in file_list:
             output_f.write(filename2 + ' \n')
             for i in range(512):
                 output_f.write(str(feats[1][i]) + ' ')

             output_f.write('\n')
             file_list.append(filename2)
##
##             for i in range(256):
##                output_f.write(str(feature[1][i]) + ' ')
##
        similarity = 1 - scipy.spatial.distance.cosine(feats[0, :], feats[1, :])

        th = 0.16
        if similarity > th and label == 1:
            right_count += 1

        elif similarity <= th and label == 0:
            right_count += 1

        else:
            print filename1
            print filename2
            print similarity
        total_count += 1
        output_f1.write(str(label) + ' ' + str(similarity) + '\n')

    print right_count / float(total_count)
    print total_count
    print right_count


def test_centerloss_has_feature():
    
    data_path = '/media/mfs/fordata/td_hg2_web_server/yangfan/face_reco/online_data_for_anotation/videos_one_person_feature'
    
    f = open('pairs_anotate.txt')
    output_f = open('feature.txt', 'w')

    lines = f.readlines()
    
    features = np.zeros((2, 512), dtype = np.float32)
    right_count = 0
    count = 0
    th = 0.41
    for line in lines:
        if count != 0:
            print float(right_count) / count
        s = line.strip('\n').split(' ')
        if len(s) == 3:
            filename1 = os.path.join(data_path, s[0], s[1].strip('.jpg'))
            filename2 = os.path.join(data_path, s[0], s[2].strip('.jpg'))
        
        if len(s) == 4:
            filename1 = os.path.join(data_path, s[0], s[1].strip('.jpg'))
            filename2 = os.path.join(data_path, s[2], s[3].strip('.jpg'))


        if not os.path.exists(filename1):
            print filename1 + ' no exist'
            continue
        if not os.path.exists(filename2):
            print filename2 + ' no exists'
            continue
        f1 = open(filename1)
        f2 = open(filename2)
        line1 = f1.readline()
        line2 = f2.readline()
        
        s1 = line1.strip('\n').split(' ')
        s2 = line2.strip('\n').split(' ')
        
        output_f.write(filename1 + '\n')
        output_f.write(line1.strip(' \n') + '\n')
        output_f.write(filename2 + '\n')
        output_f.write(line2.strip(' \n') + '\n')

        for i in range(512):
            features[0, i] = float(s1[i])
            
            features[1, i] = float(s2[i])    
            
        similarity = 1 - scipy.spatial.distance.cosine(features[0, :], features[1, :])
        print str(similarity) + ' ' + str(len(s))

        if similarity > th and len(s) == 3:
            right_count += 1

        elif similarity <= th and len(s) == 4:
            right_count += 1

        else:
            print filename1
            print filename2
            print similarity

        count += 1

def get_bad_case_feature():
            
    f = open('bad_case.txt')
    line = f.readline()

    case = []
    while line:
        if 'media' in line:
            s = line.strip('\n')
            case.append(s)

        line = f.readline()
            
    f = open('feature_after_pca.txt')
    line = f.readline()
    feature = {}
    print 'get base case over'
    count = 0
    while line:
        if 'media' in line:
            filename = line.strip(' \n')
            if filename in case:
                line = f.readline().strip(' \n')
                s = line.split(' ')
                feature[filename] = []
                for i in range(len(s)):

                    feature[filename].append(float(s[i]))

                feature[filename] = np.array(feature[filename])
           #     feature[filename][feature[filename] < 0] = 0

        line = f.readline()
        count += 1
        if count % 10000 == 0:
            print count

    #f = open('bad_case_feature.txt', 'w')
    #for fi in case:
    #    f.write(fi)
    #    f.write('\n')
    #    for i in range(feature[fi].shape[0]):

    #        f.write('%f '%(feature[fi][i]))

    #    f.write('\n')
                

    f = open('bad_case.txt')
    line = f.readline()
    while line:
        if 'media' in line:
            filename1 = line.strip('\n')
            if filename1 not in feature.keys():
                line = f.readline()
                line = f.readline()

                continue
            feature1 = feature[filename1]
            line = f.readline()
            filename2 = line.strip('\n')
            if filename2 not in feature.keys():
                line = f.readline()
                continue
            feature2 = feature[filename2]

            feature1 = feature1 / np.sqrt(np.sum(np.square(feature1)))
            feature2 = feature2 / np.sqrt(np.sum(np.square(feature2)))

            feature1[feature1[0, :] < 0] = 0
            feature2[feature2[0, :] < 0] = 0

            print filename1
            print filename2
            similarity = np.dot(feature1, feature2)
            

            
            #similarity = 1 - scipy.spatial.distance.cosine(feature1, feature2)
            print similarity
        line = f.readline()

def test_pair():
    
    prototxt = r'/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/face_deploy.prototxt'
    #caffemodel = r'/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_with_centerloss_iter_10000.caffemodel'
    caffemodel = r'/media/disk1/yangfan/face_reco/caffe-face/face_example/online_face/face_train_test_new_14w_norm_0.1centerloss_0.1triplet-centerloss_iter_40000.caffemodel'
    #caffemodel = r'/media/disk1/yangfan/py-faster-rcnn/caffe-fast-rcnn/examples/face_reco/face_train_test_new_org_iter_1000.caffemodel'
    th = 0.35

    if not os.path.isfile(caffemodel):
        print ("caffemodel not found!")
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(prototxt,caffemodel,caffe.TEST)

    #data_path = '/media/mfs/fordata/td_hg2_web_server/yangfan/face_reco/online_data/users'
    #data_path = '/media/disk1/yangfan/zhanggang/pair_result/'

    data_path = '/media/mfs/fordata/td_hg2_web_server/zhangzhiwei/data_100w/'

    feats = np.zeros((2, 512),dtype=np.float32)
    #feats = np.zeros((2, 256),dtype=np.float32)

    right_count = 0
    total_count = 0

    f = open('test_pair.txt')

    lines = f.readlines()

    for line in lines:
        s = line.strip('\n').split(' ')

        if len(s) == 3:
            dirname = s[0]
            filename1 = os.path.join(data_path, dirname, s[1])
            filename2 = os.path.join(data_path, dirname, s[2])
        elif len(s) == 2:
            filename1 = os.path.join(data_path, s[0])
            filename2 = os.path.join(data_path, s[1])
      
        print filename1
        print filename2
        input1 = cv2.imread(filename1, 0)   #read face image
        input2 = cv2.imread(filename2, 0)   #read face image

        input1 = cv2.resize(input1, (128,128), interpolation = cv2.INTER_CUBIC)   #we just need to resize the face to (128,128) 
        input2 = cv2.resize(input2, (128,128), interpolation = cv2.INTER_CUBIC)   #we just need to resize the face to (128,128) 

        start = time.time()
        img_blobinp = np.array([(input1[np.newaxis, :, :] - 127.5) / 128.0, (input2[np.newaxis, :, :] - 127.5) / 128.0])    #divide 255.0 ,make input is between 0-1
        net.blobs['data'].reshape(*img_blobinp.shape)
        net.blobs['data'].data[...] = img_blobinp
        net.blobs['data'].data.shape
        net.forward()  #go through the LCNN network
        feature = net.blobs['fc5'].data    #feature is from eltwise_fc1 layer
        feature = net.blobs['fc5'].data    #feature is from eltwise_fc1 layer
        #feats[0,:] = feature.reshape((512))[0:256]
        feats[0,:] = feature.reshape((1024))[0:512]
        feats[1, :] = feature.reshape((1024))[512:1024]
        #feature = net.blobs['fc'].data    #feature is from eltwise_fc1 layer
        #feats[0,:] = feature[0].reshape((512))
    
        similarity = 1 - scipy.spatial.distance.cosine(feats[0, :], feats[1, :])
        print similarity

if __name__ == '__main__':
    test_centerloss()
#    test_centerloss_17w()
    #get_bad_case_feature()

    #test_pair()

