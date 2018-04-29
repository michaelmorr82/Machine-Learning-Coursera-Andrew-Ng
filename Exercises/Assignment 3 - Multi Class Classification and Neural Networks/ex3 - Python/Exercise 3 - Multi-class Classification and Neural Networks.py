
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# load MATLAB files
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression


#%% 

'''---------------------------------------------
            LOAD MATLAB DATAFILES
--------------------------------------------------'''
# Load data
data = loadmat('ex3data1.mat')
data.keys()

#Load th weights
weights = loadmat('ex3weights.mat')
weights.keys()

y = data['y']

# Add x0 
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]

m,n = X.shape
 
print('X: {} (with intercept)'.format(X.shape))
print('y: {}'.format(y.shape))

print('\nNumber of features (n): %.0f'%(n))
print('Number of training examples (nm): %.0f'%(m))

theta1, theta2 = weights['Theta1'], weights['Theta2']

print('\ntheta1: {}'.format(theta1.shape))
print('theta2: {}'.format(theta2.shape))

sample = np.random.choice(X.shape[0], 20)
plt.imshow(X[sample,1:].reshape(-1,20).T)
plt.axis('off');


#%% 
'''----------------------------------------------
            REGULARISAED COST FUNCTION
--------------------------------------------------'''

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
    
#logitic regression cost fucntion (Regularised)
def lrcostFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    #cost fuinction
    J = -1 * (1/m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (reg/(2 * m))*np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])    

# logistic regreession gradient of cost function
def lrgradientReg(theta, reg, X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
     
    #Gradient
    grad = (1/m)*X.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())

#%% 
'''------------------------------------------------
            ONE - Vs - ALL CLASSIFICATION
----------------------------------------------------'''

# logistic regression - one - Vrs - all
def oneVsAll(features, classes, n_labels, reg):
    initial_theta = np.zeros((X.shape[1],1))  # 401x1
    all_theta = np.zeros((n_labels, X.shape[1])) #10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args = (reg, features, (classes == c) * 1), 
                       method = None,jac = lrgradientReg, options = {'maxiter':50})
        all_theta[c-1] = res.x
    return(all_theta)

theta = oneVsAll(X, y, 10, 0.1)
print(theta)

#%% 
'''--------------------------------------------------
            ONE - Vs- ALL PREDICTION
------------------------------------------------------'''

def predictOneVsAll(all_theta, features):
    probs = sigmoid(X.dot(all_theta.T))
        
    # Adding one because Python uses zero based indexing for the 10 columns (0-9),
    # while the 10 classes are numbered from 1 to 10.
    return(np.argmax(probs, axis = 1) + 1)
 
pred = predictOneVsAll(theta, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))

#%%  

'''--------------------------------------------------------------
            MULTICLASS LOGFISTIC REGRESSION WITH SCKIT-LERN
---------------------------------------------------------------------'''

clf = LogisticRegression(C = 10, penalty='l2', solver='liblinear')
# Scikit-learn fits intercept automatically, so we exclude first column with 'ones' from X when fitting.
clf.fit(X[:,1:],y.ravel())

pred2 = clf.predict(X[:,1:])
print('Training set accuracy: {} %'.format(np.mean(pred2 == y.ravel())*100))



#%% 
'''------------------------------------------------
                  NEURAL NETWORKS
----------------------------------------------------'''


def predict(theta_1, theta_2, features):
    z2 = theta_1.dot(features.T)
    a2 = np.c_[np.ones((data['X'].shape[0],1)), sigmoid(z2).T]
    
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)
        
    return(np.argmax(a3, axis=1)+1) 

pred3 = predict(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred3 == y.ravel())*100))


print('---------------------------------------')
print('\tSummary')
print('----------------------------------------')
print('Training set accuracy:')
print('\tMulti-class Logistical regression using 1-Vrs_all {} %'.format(np.mean(pred == y.ravel())*100))
print('\tMulti-class Logistical regression using sklearn {} %'.format(np.mean(pred2 == y.ravel())*100))
print('\tMulti-class Logistical regression using Neural network {} %'.format(np.mean(pred3 == y.ravel())*100))



