
# %load ../../standard_import.txt
import pandas as pd # for data structuring and data analysis tools
import numpy as np # arrays, algebra
import matplotlib.pyplot as plt #2-d plotting
import sympy

from mpl_toolkits.mplot3d import axes3d # 3 d plots

# setting pandas configuration
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns # statistical data visualisation

sns.set_context('notebook')
sns.set_style('white')

from sympy.abc import theta

plt.close("all") # Clsoe all plots


#%%
# =============================================================================
#         LOAD DATA
# =============================================================================

data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size

# Print out some data points
print('First 10 examples from the dataset:')
print(np.column_stack( (X[:10], y[:10]) ))
#%%
# =============================================================================
#                     FEATURE NORMALISATION
# =============================================================================

# Function to normalise features
def featureNormalize(X):
    
    mu = np.mean(X[:, :1])
    sigma = np.std(X[:, :1])

    mu1 = np.mean(X[:, 1:])
    sigma1 = np.std(X[:, 1:])

    x_ = (X[:, :1] - mu) / sigma
    x1_ = (X[:, 1:] - mu1) / sigma1

    X_norm = np.append(x_, x1_, axis = 1)
    return X_norm, mu, sigma
    
X_norm, mu, sigma = featureNormalize(X)

print('Normalisation Parameters:')
print('\tmu: %.2f'%(mu))
print('\tsigma: %.2f'%(sigma))

# Add intercept term to X
X_norm = np.concatenate((np.ones((m, 1)), X), axis = 1)

n = len(X_norm[0])
#%%
# =============================================================================
#                    GRADIENT DESCENT
# =============================================================================

def computeCostMulti(X, y, theta_gd):
    m = y.size
    J = 0

    h = np.dot(X, theta_gd)
    J = 1 / (2 * m) * np.sum(np.square(h - y))
    
    return J

def gradientDescentMulti(X, y, theta_gd, alpha, num_iters):
    theta_gd_history = np.zeros((num_iters,3)) 
    
    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):

        h = np.dot(X_norm, theta_gd) # calculate hypothesis
        theta_gd = theta_gd - ((alpha / m) * (np.dot(X.T, (h - y))))
        
        theta_gd_history[i][0] = theta_gd[0] 
        theta_gd_history[i][1] = theta_gd[1] 
        theta_gd_history[i][2] = theta_gd[2] 
        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta_gd))
        

    return theta_gd, theta_gd_history, J_history

# Gradient descent parameters
alpha = 0.000000001
num_iters = 500

# Init Theta and Run Gradient Descent 
theta_gd = np.zeros(n)
theta_gd, theta_gd_history, J_history = gradientDescentMulti(X_norm, y, theta_gd, alpha, num_iters)

#detertine the y values according to the theta values from gradient descent
y_gd = X_norm.dot(theta_gd) 
#determine the r^2 value
from sklearn.metrics import r2_score 
r2_gd = r2_score(y, y_gd) 

# Plot the convergence graph
plt.figure()
plt.plot(theta_gd_history[:,0], '-r')
plt.plot(theta_gd_history[:,2], '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Value of Parameter')
plt.grid(True)
plt.legend([r'$\theta_0$', r'$\theta_2$'])

# Plot the convergence graph
plt.figure()
plt.plot(theta_gd_history[:,1], '-g')
plt.xlabel('Number of iterations')
plt.ylabel('value of Parameter')
plt.legend([r'$\theta_1$'])
plt.grid(True)

plt.figure()
plt.plot(J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.grid(True)

print('-----------------------------------')
print('\t\tFITING RESULTS GRADIENT DESCENT')
print('-----------------------------------')
print('\nFitting: Gradient Descent')
print('Learning rate (alpha): %.2f'%(alpha))
print('Number of iterations: %.0f'%(num_iters))
print('Fitting parameters:')
print('\t%s_0: %.3f'%(sympy.pretty(theta), theta_gd[0]))
print('\t%s_1: %.3f'%(sympy.pretty(theta), theta_gd[1]))
print('\t%s_2: %.3f'%(sympy.pretty(theta), theta_gd[2]))
print('Goodness of fit r^2: %.3f'%(r2_gd))





