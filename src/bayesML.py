# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:47:53 2021

@author: Markus
"""


import numpy as np
# Just import random, avoid name collision

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal as mlnorm
from scipy.stats import norm as norm
from sklearn import metrics


class Bayes(object):


    def __init__(self):
        pass

    # Reduce picture size
    def cifar10_X_times_X_color(self,X,imageSize):
        size = len(X) * 3 * imageSize **2 if X.ndim > 1 else 3 * imageSize **2
        X_reduced = X.reshape(size, -1)
        X_reduced = np.average(X_reduced, axis = 1)
        X_reduced = X_reduced.reshape(-1,3)
    
        return X_reduced
    
    def fit(self, X, Y,imageSize, method):
        
        self.method = method
        self.imageSize = imageSize

        
        y_list = Y.tolist()
        datadict = {}
    
        # first need to ### data
        X_reduced = self.cifar10_X_times_X_color(X,imageSize)

        
        # Need to sort orginal data related to classes
        X_list = X_reduced.reshape(len(y_list),-1).tolist()
        X_reduced_sorted = [x for _, x in sorted(zip(y_list,X_list))]
        X_reduced_sorted = np.array(X_reduced_sorted).reshape(10,5000,-1)

        
        # Calculate mean values for every class picture
        means = np.average(X_reduced_sorted, axis = 1)
        # Calculate prior values for every class picture
        priorvalues = np.array([y_list.count(a) / len(y_list) for a in range(10)])
        
        if(method == "naive"):
            # Calculate covariances for every class picture
            covariances = np.std(X_reduced_sorted, axis = 1,dtype = np.float64)
        elif (method == "multivariate"):
            # Calculate covariances for every class picture
            covariances = np.array([np.cov(e,rowvar=0) for e in X_reduced_sorted])
        else:
            print("Error")

        # Initialize dictionary 
        datadict['mu'] = means
        datadict['sigma'] = covariances
        datadict['p'] = priorvalues
        
        self.datadict = datadict
        
        
    def predict(self, X,Y, sampleSize):
        self.Y_true = Y
        B_pred = np.array([])
        if(self.method == "naive"):
            for ind in tqdm(range(sampleSize)):
                densities = np.prod(norm.pdf(self.cifar10_X_times_X_color(X[ind],self.imageSize).flatten(), self.datadict['mu'], self.datadict['sigma']), axis = 1) 
                class_pred = np.flatnonzero(densities == max(densities))[0]
                B_pred = np.append(B_pred, class_pred)
            self.predicted_values = B_pred
        elif(self.method == "multivariate"):
            for ind in tqdm(range(sampleSize)):
                densities = np.array([mlnorm.pdf(self.cifar10_X_times_X_color(X[ind],self.imageSize).flatten(), mean = self.datadict['mu'][a].flatten(), cov = self.datadict['sigma'][a]) for a in range(10)])
                class_pred = np.flatnonzero(densities == max(densities))[0]
                B_pred = np.append(B_pred, class_pred)
            self.predicted_values = B_pred
        else:
            pass
        self.accuracy = self.class_acc(Y[0:sampleSize])
        print(f'\nPredicting accuracy is {self.accuracy}.')

            
            
    def class_acc(self,gt):
        if len(self.predicted_values) == len(gt) and len(self.predicted_values) > 0:
            is_correct = np.equal(self.predicted_values,gt)
            return sum(is_correct) / len(is_correct)
        else:
            print("Length of predicted labels differs from the length of ground truth!")
        return 0
            

    def roc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.Y_true,self.predicted_values, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        



        