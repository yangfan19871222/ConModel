#!/usr/bin/python

import os,sys
from sklearn.decomposition import PCA
import numpy as np

if __name__ == '__main__':
    f = open('feature1.txt')
    lines = f.readlines()
    data = []
    for line in lines:
        s = line.strip(' \n').split(' ')
        if len(s) < 10:
            continue
        print len(s)
        x = []
        for i in range(len(s)):
            x.append(float(s[i]))
#        x /= np.sqrt(np.sum(np.square(x)))
        data.append(x)

    pca = PCA(copy = False, n_components= 128)
    data = np.array(data)
    pca.fit(data)
    #pca_data =  pca.transform(data[0])

#    print pca.get_params()

    f.close()
    f = open('feature1.txt')
    output_f = open('feature_after_pca.txt', 'w')

    lines = f.readlines()
    for line in lines:
        s = line.strip(' \n').split(' ')
        if len(s) < 10:
            output_f.write(line)
            continue
        x = []
        for i in range(len(s)):
            x.append(float(s[i]))

      #  x /= np.sqrt(np.sum(np.square(x)))
        x0 = pca.transform(x)
        #print type(x0)
        #print x0.shape
       
        for i in range(len(x0[0])):
            output_f.write(str(x0[0][i]) + ' ')

        output_f.write('\n')

