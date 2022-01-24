# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 21:59:59 2021

@author: Markus
"""

import os
from bayesML import Bayes
from cifarData import getCifarData





path = os.path.dirname(os.path.abspath('')) 

# Get the data
X_train,Y_train, X_test, Y_test = getCifarData(path)

# Create model
bay = Bayes()
bay.fit(X_train, Y_train,2,'multivariate')
bay.predict(X_test, Y_test, len(X_test))

bay.roc()





    
    