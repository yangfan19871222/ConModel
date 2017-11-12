#!/usr/bin/python
import os, sys

import numpy as np
#import cv2
import scipy
from scipy import spatial
th = 0.3

root_path = os.getcwd()
#data_path = '/media/disk1/yangfan/lfw/lfw_aligned_for_centerloss/pair_files'
#data_path = '/media/mfs/fordata/td_hg2_web_server/yangfan/face_reco/online_data/users' 
#data_path = '/media/mfs/fordata/td_hg2_web_server/yangfan/face_reco/online_data_for_anotation/videos_one_person/' 
data_path = '/home/yangfan/face_reco/online_data/centerloss_model_aligned/pair_files'

#
if __name__ == '__main__':
#    file_path = os.path.join(root_path, 'pairs_anotate.txt')
    file_path = os.path.join(root_path, 'pairs.txt')
    f = open('feature_after_pca.txt')

    lines = f.readlines()

    features = {}
    for line in lines:
        s = line.strip(' \n').split(' ')

        if len(s) < 10:
           cur_filename = line.strip('\n ') 

        else:
            x = []
            for i in range(len(s)):
                x.append(float(s[i]))
            print cur_filename
            features[cur_filename] = x
     
    f.close()

    f = open(file_path)
    output_f = open('online_face_result1.txt', 'w')
    
    f.readline()
    lines = f.readlines()
    total_count = 0
    right_count = 0
    for line in lines:
        s = line.strip('\n').split('\t ')
        #s = line.strip('\n').split(' ')
        print s
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
            #filename1 = os.path.join(data_path, s[0], s[1])
            #filename2 = os.path.join(data_path, s[2], s[3])
            label = 0

       # filename1 = filename1.strip('.jpg')
       # filename2 = filename2.strip('.jpg')
        print filename1
        print filename2
        if label == 2:
            continue
      
       
        if not os.path.exists(filename1) or not os.path.exists(filename2):
            print filename1
            print filename2
            print 'dddd'
            continue
            
        if not  filename1 in features.keys():
            continue
        if filename2 not in features.keys():
            continue

        feature1 = features[filename1]
        feature2 = features[filename2]

        similarity = 1 - spatial.distance.cosine(feature2, feature1)
        
        if similarity > th and label == 1:
            right_count += 1
            
        if similarity <= th and label == 0:
            right_count += 1

        print similarity
        output_f.write(str(label) + ' ' + str(similarity) + '\n')
        total_count += 1

