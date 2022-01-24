import os
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict




def getCifarData(path):
    datadict_1 = unpickle(path + str('\\Cifar10Data\\data_batch_1')) 
    X_train = datadict_1["data"]
    Y_train = datadict_1["labels"]


    for a in (range(2,6)):
        datadict = unpickle(path + str('\\Cifar10Data\\data_batch_') + str(a))
        X_train_part = datadict["data"]
        Y_train_part = datadict["labels"]
        X_train = np.concatenate((X_train, X_train_part), axis = 0)
        Y_train = np.concatenate((Y_train, Y_train_part), axis = 0)
    
    X_train = X_train.astype('float64')
    X_train = X_train /  255
    
    # Test data

    datadict = unpickle(path + str('\\Cifar10Data\\test_batch')) 
    X_test = datadict["data"]
    Y_test = datadict["labels"]
    X_test = X_test.astype('float64')
    X_test = X_test / 255
    return X_train,Y_train, X_test, Y_test

