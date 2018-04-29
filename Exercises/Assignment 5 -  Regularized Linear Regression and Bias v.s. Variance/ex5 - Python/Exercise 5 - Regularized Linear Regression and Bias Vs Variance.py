
# %load ../../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

#%%

'''---------------------------------------------
            LOAD DATAFILES
--------------------------------------------------'''
data = loadmat('ex5data1.mat')

y_train = data['y']
X_train = np.c_[np.ones_like(data['X']), data['X']]

y_val = data['yval']
X_val = np.c_[np.ones_like(data['Xval']), data['Xval']]
print('----------------------------------------------------')
print('DATA')
print('Training Data X:', X_train.shape)
print('Traingin Data y:', y_train.shape)
print('Cross validation Data X:', X_val.shape)
print('Cross validation Data y:', y_val.shape)

#%%
'''---------------------------------------------
            REGULARISED LINEAR REGRESSION
--------------------------------------------------'''
plt.figure()
plt.scatter(X_train[:,1], y_train, s = 50, c = 'r', marker = 'x', linewidths = 1)
plt.scatter(X_val[:,1], y_val, s = 50, c = 'b', marker = 'o', linewidths = 1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.legend(('Trainging Data','Validation Data'))
#plt.grid(True)

#%%            REGULARISED COST FUNCTION


def linearRegCostFunction(theta, X, y, Lambda):
    m = y.size
    
    h = X.dot(theta)
    #cost funtion
    J = (1/(2 * m)) * np.sum(np.square(h - y)) + (Lambda/(2 * m)) * np.sum(np.square(theta[1:]))
   
    return(J)


def lrgradientReg(theta, X, y, Lambda):
    m = y.size
    
    h = X.dot(theta.reshape(-1,1))
        
    grad = (1/m) * (X.T.dot(h - y)) + (Lambda / m) * np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())

initial_theta = np.ones((X_train.shape[1],1))
cost = linearRegCostFunction(initial_theta, X_train, y_train, 0)
gradient = lrgradientReg(initial_theta, X_train, y_train, 0)

print('results')
print('cost at theta = [1,1]: %.2f'%(cost))
print('cost at theta = [1,1]: %.2f , %.2f'%(gradient[0],gradient[1]))

def trainLinearReg(X, y, Lambda):
    #initial_theta = np.zeros((X.shape[1],1))
    initial_theta = np.array([[1],[1]])
    # For some reason the minimize() function does not converge when using
    # zeros as initial theta.
        
    res = minimize(linearRegCostFunction, initial_theta, args = (X_train,y_train,Lambda), method = None, jac = lrgradientReg,
                   options = {'maxiter':5000})
    
    return(res)

#fit the linear regression using minimize 
fit = trainLinearReg(X_train, y_train, 0)
print('\n\nfitting parameters of linear regression with regaularisation')
print('Lambda  = 0')
print('theta0 = %.2f'%(fit.x[0]))
print('theta1 = %.2f'%(fit.x[1]))
print('other parameters:')
print(fit)

#%%
'''---------------------------------------------
        Comparison: coefficients and cost obtained with LinearRegression in Scikit-learn
--------------------------------------------------'''
# fit using linear regression from sklearn
regr = LinearRegression(fit_intercept = False)
regr.fit(X_train,y_train.flatten())
print(regr.coef_)
print('theta0 = %.2f'%(regr.coef_[0]))
print('theta1 = %.2f'%(regr.coef_[1]))
print(linearRegCostFunction(regr.coef_, X_train, y_train, 0))

#compere 2 methods of fitting
plt.figure()
plt.plot(np.linspace(-50,40), (fit.x[0]+ (fit.x[1]*np.linspace(-50,40))), label = 'Scipy optimize')
plt.plot(np.linspace(-50,40), (regr.coef_[0]+ (regr.coef_[1]*np.linspace(-50,40))), label='Scikit-learn')
plt.scatter(X_train[:,1], y_train, s = 50, c = 'r', marker ='x', linewidths = 1,label = 'Training data')
plt.scatter(X_val[:,1], y_val, s = 50, c = 'b', marker = 'o', linewidths = 1,label = 'Validation data')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.legend(loc = 4);



#%%
def learningCurve(X, y, X_val, y_val, reg):
    m = y.size
    
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    
    for i in np.arange(m):
        res = trainLinearReg(X_train[:i+1], y_train[:i+1], reg)
        error_train[i] = linearRegCostFunction(res.x, X_train[:i + 1], y[:i + 1], reg)
        error_val[i] = linearRegCostFunction(res.x, X_val, y_val, reg)
    
    return(error_train, error_val)

Error_train, Error_val = learningCurve(X_train, y_train, X_val, y_val, 0)

plt.plot(np.arange(1,13), Error_train, label='Training error')
plt.plot(np.arange(1,13), Error_val, label='Validation error')
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend();
#
#def learningCurve(X_train, y_train, X_val, y_val, reg):
#    m = y_train.size
#    
#    error_train = np.zeros((m, 1))
#    error_val = np.zeros((m, 1))
#    theta_fit = np.zeros((m,2))
#    
#    for i in np.arange(m): #for loop incremenst over m
#        res = trainLinearReg(X_train[:i + 1], y_train[:i + 1], reg)
#        theta_fit[i,:] = res.x 
#        error_train[i] = linearRegCostFunction(theta_fit[i,:], X_train[:i + 1], y_train[:i + 1], reg)
#        error_val[i] = linearRegCostFunction(theta_fit[i,:], X_val, y_val, reg)
#    
#    return(error_train, error_val,theta_fit)
#
#Error_train, Error_val, theta_fit = learningCurve(X_train, y_train, X_val, y_val, 0)
#
#plt.plot(np.arange(1,13), Error_train, label='Training error')
#plt.plot(np.arange(1,13), Error_val, label='Validation error')
#plt.title('Learning curve for linear regression')
#plt.xlabel('Number of training examples')
#plt.ylabel('Error')
#plt.legend();
#%%
'''----------------------------------------------------------------
                     POLYNOMIAL REGRESSION (SCIKIT-LEARN)
------------------------------------------------------------------------'''

poly = PolynomialFeatures(degree=8)
X_train_poly = poly.fit_transform(X_train[:,1].reshape(-1,1))

regr2 = LinearRegression()
regr2.fit(X_train_poly, y_train)

regr3 = Ridge(alpha = 20) #Linear least squares with l2 regularization.
regr3.fit(X_train_poly, y_train)

# plot range for x
plot_x = np.linspace(-60,45)
# using coefficients to calculate y
plot_y = regr2.intercept_+ np.sum(regr2.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)
plot_y2 = regr3.intercept_ + np.sum(regr3.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)

plt.plot(plot_x, plot_y, label='Scikit-learn LinearRegression')
plt.plot(plot_x, plot_y2, label='Scikit-learn Ridge (alpha={})'.format(regr3.alpha))
plt.scatter(X_train[:,1], y_train, s=50, c='r', marker='x', linewidths=1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial regression degree 8')
plt.legend(loc=4);

