#!/usr/bin/python

import os, sys
import numpy as np

if __name__ == '__main__':
    
    f = open('online_face_result1.txt')
    
    lines = f.readlines()

    data = {}
    for line in lines:
        s = line.strip('\n').split(' ')
        print s
        if s[0] == '1':
            if 1 not in data.keys():
                data[1] = []
            data[1].append(float(s[1]))
        else:
            if 0 not in data.keys():
                data[0] = []
            
            data[0].append(float(s[1]))

    output_f = open('centerloss_cnn_roc.txt', 'w')

    data[0] = np.array(data[0])
    data[1] = np.array(data[1])
    print data[0]
    print data[1]

    for thresh in range(0, 500):
    
        cur_thresh = thresh * 0.01
        fpr = len(np.where(data[0] > cur_thresh)[0]) /  float(data[0].shape[0])
        tpr = len(np.where(data[1] > cur_thresh)[0]) /  float(data[1].shape[0])
        output_f.write(str(tpr) + ' ' + str(fpr) +  ' ' +  str(cur_thresh) + '\n') 

