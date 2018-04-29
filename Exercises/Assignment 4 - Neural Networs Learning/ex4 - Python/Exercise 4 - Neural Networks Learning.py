# %load ../../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# load MATLAB files
from scipy.io import loadmat

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns

#%%
'''---------------------------------------------
            LOAD MATLAB DATAFILES
--------------------------------------------------'''

data = loadmat('ex4data1.mat')
data.keys()

y = data['y']

# Add intercept
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]

print('X:',X.shape, '(with intercept)')
print('y:',y.shape)

m,n = X.shape
print('\nNumber of features (n): %.0f'%(n))
print('Number of training examples (nm): %.0f'%(m))

weights = loadmat('ex4weights.mat')
weights.keys()

theta1, theta2 = weights['Theta1'], weights['Theta2']
print('\ntheta1 :', theta1.shape)
print('theta2 :', theta2.shape)
params = np.r_[theta1.ravel(), theta2.ravel()]
print('params :', params.shape)


#%%
'''---------------------------------------------
            NEURAL NETWORKS
--------------------------------------------------'''
L = 400 # Input layer size = 400 (20x20 pixels)
# Hidden layer size = 25
# Number of labels = 10
# Neural Networks - Feed Forward and Cost Function

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def sigmoidGradient(z):
    return(sigmoid(z) * (1 - sigmoid(z)))

#Neural Network Cost Function
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, features, classes, reg):
    
    theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size,(input_layer_size + 1))
    theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels,(hidden_layer_size + 1))

    m = features.shape[0]
    y_matrix = pd.get_dummies(classes.ravel()).as_matrix() 
    
    # Cost
    a1 = features # 5000x401
        
    z2 = theta1.dot(a1.T) # 25x401 * 401x5000 = 25x5000 
    a2 = np.c_[np.ones((features.shape[0],1)),sigmoid(z2.T)] # 5000x26 
    
    z3 = theta2.dot(a2.T) # 10x26 * 26x5000 = 10x5000 
    a3 = sigmoid(z3) # 10x5000
    
    J = -1 * (1/m) * np.sum((np.log(a3.T) * (y_matrix) + np.log(1 - a3).T * (1 - y_matrix))) + \
        (reg/(2 * m)) * (np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))

    # Gradients
    d3 = a3.T - y_matrix # 5000x10
    d2 = theta2[:,1:].T.dot(d3.T) * sigmoidGradient(z2) # 25x10 *10x5000 * 25x5000 = 25x5000
    
    delta1 = d2.dot(a1) # 25x5000 * 5000x401 = 25x401
    delta2 = d3.T.dot(a2) # 10x5000 *5000x26 = 10x26
    
    theta1_ = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]
    theta2_ = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]
    
    theta1_grad = delta1/m + (theta1_ * reg)/m
    theta2_grad = delta2/m + (theta2_ * reg)/m
    
    return(J, theta1_grad, theta2_grad)

#%%

# Regularization parameter = 0
Lambda = 0
nnCostFunction(params, 400, 25, 10, X, y, 0)[0]
#%%
#print('For Lambda = %.0f, \n\tCost  = %.2f \n\t Theta1_grad = %.2f \n\t Theta2_grad = %.2f' %(Lambda, J, theta1_grad, theta2_grad ) )


# Regularization parameter = 1]
Lambda = 1
nnCostFunction(params, 400, 25, 10, X, y, Lambda)[0]


[sigmoidGradient(z) for z in [-1, -0.5, 0, 0.5, 1]]

