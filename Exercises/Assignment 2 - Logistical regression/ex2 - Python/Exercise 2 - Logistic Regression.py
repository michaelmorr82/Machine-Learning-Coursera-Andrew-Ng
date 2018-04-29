# %load ../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

 
#%config InlineBackend.figure_formats = {'pdf',}
#%matplotlib inline

import seaborn as sns

#%% 
# =============================================================================
#                   LOAD THE RAW DATA AND PLOT
# =============================================================================

file_name = 'ex2data1.txt'
data = np.loadtxt(file_name, delimiter =',')
print('Dimensions: ',data.shape)

X = np.c_[np.ones((len(data),1)), data[:,0:2]] # create X array colum of 1's and 2 colums of data
y = np.c_[data[:,2]]# create y array colum of output data data

m = X.shape[0] # number of training examples
n = X.shape[1]# number of features... includeing the addition of the 1's

print('file: %s'%(file_name))
print('number of features (n): %.0f'%(n))
print('number of examples (m):%.0f\n'%(m))


def plotData(data, label_x, label_y, label_pos, label_neg, axes = None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0 # ind where all the negative values are
    pos = data[:,2] == 1 # ind where all the postive values are
    
    # If no specific axes object has been passed, get the current axes.
    plt.figure()
    if axes == None:
        axes = plt.gca()
    #plot
    axes.scatter(data[pos][:,0], data[pos][:,1], marker = '+', c = 'k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c = 'y', s = 60, label = label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon = True, fancybox = True);
    plt.grid(True)
    
    print(data[pos][:,0])

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')


#%% 
    
# =============================================================================
#                  LOGISTIC REGRESSION
# =============================================================================

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def costFunction(theta, X, y):
    h = sigmoid(X.dot(theta))
    
    J = -1 * (1/m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
               
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    dJ =(1/m) * X.T.dot(h - y)

    return(dJ.flatten())

initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
dJ = gradient(initial_theta, X, y)

print('-----------------------------------------------------------------')
print('values at initial thetas: %.2f, %.2f, %.2f' %(initial_theta[0], initial_theta[1], initial_theta[2] ))
print('Cost of initial thetas: %.3f'%(cost))
print('Grad Cost: %.3f,  %.3f,  %.3f'%(dJ[0], dJ[1], dJ[2]))
print('-----------------------------------------------------------------')


#%%
# =============================================================================
#                       OPTIMISATION COST FUNCTION
# =============================================================================

res_opt = minimize(costFunction, initial_theta, args = (X,y), method = None, jac = gradient, options={'maxiter':400})
theta = res_opt.x # set the optimised theta values
#%%

# =============================================================================
#                     PREDICT
# =============================================================================
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

#predicts the pass or fail of students from exam1 & 2 (provided in X)
def predict(theta, X, threshold = 0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold # rue if sigmoid of exams results using optimised theta > 0.5
    return(p.astype('int'))# 1 if true (i.e. pass) 0 = false (i.e. fail)

# Student with Exam 1 score 45 and Exam 2 score 85
# Predict using the optimized Theta values from above
Student_A = np.array([1,45, 85]) 
Student_A_prob = sigmoid(Student_A.dot(theta.T))
grade_Stu_A = predict(theta, Student_A)


Class_Admissions = predict(theta, X)
correct_pred = sum(Class_Admissions == y.ravel())
predict_rate = correct_pred/Class_Admissions.size

print('-----------------------------------------------------------')
print('RESULTS USING THE MINIMISE FUNCTION\n')
print('Optimisation fucntion:minimise()')
print('Values of thetea: %.2f, %.2f %.2f\n'%(theta[0], theta[1], theta[2]))
print('Single Student  results:')
print('student Grades: %.2f,%.2f'%(Student_A[1],Student_A[2]))
print('Predicted Grade: %s'%('PASS' if grade_Stu_A == 1 else 'FAIL')) #prints pass if pass is predited else prints fail
print('Probability of Admission: %.2f%%'%(Student_A_prob * 100))
print('\n')

print('Full Class  results:')
print('number of correction prediction: %.2f'%(correct_pred))
print('Train accuracy %.2f%%'%(100 * predict_rate))
print('-----------------------------------------------------------')


#%% 
# =============================================================================
#                        BOUNDARY DECISION
# =============================================================================

#find the max and min of exam results to create limits for trend line
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max)) 

#create a of X_data in one line... rshape it later.. this is faster
x0_line = np.ones((xx1.ravel().shape[0],1)) 
x1_line = xx1.ravel()
x2_line = xx2.ravel()

h_prep = np.c_[x0_line, x1_line, x2_line].dot(theta)
h = sigmoid(h_prep) # pass it thought eh sigmoid fucntion
h = h.reshape(xx1.shape) # rehape it to the size of teh grid

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b'); 

#%% 
# =============================================================================
#                 REGULARISATION LOGISTIC REGRESSION
# =============================================================================

data2 = np.loadtxt('ex2data2.txt', delimiter =',')

y = np.c_[data2[:,2]]
X = data2[:,0:2]

plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')

# =============================================================================
#                         POLYNOMIALS
# =============================================================================

# Note that this function inserts a column with 'ones' in the design matrix for the intercept.
poly = PolynomialFeatures(6)
XX = poly.fit_transform(data2[:,0:2])
XX.shape


# Determine Cost Function
def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    
    J = -1*(1/m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg/(2 * m)) * np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

# Gradient of Regression
def gradientReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
      
    grad = (1/m) * XX.T.dot(h - y) + (reg / m) * np.r_[[[0]],theta[1:].reshape(-1, 1)]
        
    return(grad.flatten())

initial_theta = np.zeros(XX.shape[1])
costFunctionReg(initial_theta, 1, XX, y)

fig, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))

# Decision boundaries
# Lambda = 0 : No regularization --> too flexible, overfitting the training data
# Lambda = 1 : Looks about right
# Lambda = 100 : Too much regularization --> high bias

for i, C in enumerate([0, 1, 100]):
    # Optimize costFunctionReg

    res_opt2 = minimize(costFunctionReg, initial_theta, args = (C, XX, y), method = None, jac = gradientReg, options = {'maxiter':3000})
    
    # Accuracy
    accuracy = 100 * sum(predict(res_opt2.x, XX) == y.ravel())/y.size    

    # Scatter plot of X,y
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    
    # Plot decisionboundary
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res_opt2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))

