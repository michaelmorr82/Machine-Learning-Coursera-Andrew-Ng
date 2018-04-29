
# %load ../../standard_import.txt
import pandas as pd # for data structuring and data analysis tools
import numpy as np # arrays, algebra
import matplotlib.pyplot as plt #2-d plotting
import sympy

from mpl_toolkits.mplot3d import axes3d # 3 d plots

import seaborn as sns # statistical data visualisation

sns.set_context('notebook')
sns.set_style('white')

from sympy.abc import theta

plt.close("all") # Clsoe all plots


#%%

# =============================================================================
#                       WARM UP EXERCISE
# =============================================================================

I_num = 5
def warmUpExercise(I_num):
    Ident = np.identity(I_num)
    return(Ident)

Ident = warmUpExercise(I_num)
# print('Identity Matrix %d x %d'%(I_num, I_num))
print('Identy Maxtric of %d x %d:'%(I_num, I_num))

print(Ident)

#%% 
# =============================================================================
#                     LOAD THE RAW DATA AND PLOT
# =============================================================================

print(' =============================================================================')
print(' \t\tLOAD RAW DATA & PLOT')
print(' =============================================================================')
data = np.loadtxt('ex1data1.txt', delimiter = ',') #imports 96 x 2 array

X = np.c_[np.ones(len(data)),data[:,0]] #concatinates column of 1's with first imported colums (x-data)

y = np.c_[data[:,1]] # set y data to the second imported column

plt.figure()
plt.scatter(X[:,1], y, s = 30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
plt.title('Raw Data Imported')
plt.grid(True)


#%%
# =============================================================================
#             GRADIENT DESCENT
# =============================================================================


print(' =============================================================================')
print(' \t\t\tGRADIENT DESCENT')
print(' =============================================================================')
m = y.size # number of training examples
theta_gd=[[0],[0]]  # create first theta values as a list.. theta0 and theta1

def computeCost(X, y, theta_gd):

    J = 0
    
    hypoth = X.dot(theta_gd) # determin the hypothesis

    J = 1 / (2 * m) * np.sum(np.square(hypoth - y)) # determine the cost function
    
    return(J) # return the cost function

J = computeCost(X, y, theta_gd) # run the cos function
print('The initial Parameters \n\tTheta1 = 0 \n\tTheta1 = 0 \n\tCost: %.3f'%(J))


alpha = 0.01 #define teh learning rate 
num_iters = 2000 # define the number of iterations of gradient descent
x_scale = np.arange(num_iters)

def gradientDescent(X, y, theta_gd, alpha, num_iters):
   
    J_history = np.zeros(num_iters) # create array of zeros for the cost function values
    theta_gd_history = np.zeros((num_iters,2)) # create array for the theta history values
    
    for iter in np.arange(num_iters): #gradient descent iteration loop
        
        hypoth = X.dot(theta_gd) # determine the hypothesis for the current value of thetas
        dJ_theta_gd  = (1/m) * (X.T.dot(hypoth - y)) # Determine derivative of cost function
        theta_gd = theta_gd - alpha * dJ_theta_gd #calculate new values of theta
        
        # store calculated values of theta  
        theta_gd_history[iter][0] = theta_gd[0] 
        theta_gd_history[iter][1] = theta_gd[1]
        
        J_history[iter] = computeCost(X, y, theta_gd) #calclate the cost fucntion
        
    return(theta_gd, J_history, theta_gd_history) # return final theta and the list of cost function results

# run gradient descent to return final theta for minimes cost funtiona as well as all the cost function values calculated (for each iteration)
theta_gd, J_history, theta_gd_history = gradientDescent(X, y, theta_gd, alpha, num_iters) 
print('\nThe values of minimised cost function:')
print('\tTheta1: %f' %(theta_gd[0]))  
print('\tTheta2: %f' %(theta_gd[1]))      

#detertine the y values according to the theta values from gradient descent
y_gd = X.dot(theta_gd) 
#determine the r^2 value
from sklearn.metrics import r2_score 
r2_gd = r2_score(y, y_gd) 

#%%
# plot theta values as a function of iterations of gradient descent
plt.figure()
plt.plot(x_scale, theta_gd_history)
plt.ylabel('Theta Values')
plt.xlabel('Iterations');
plt.title('Theta Values over iterations')
plt.grid(True)
plt.legend([r'$\theta_0$', r'$\theta_1$'])

# plot cost function histtory as a function of iterations of gradient descent
plt.figure()
plt.plot(x_scale, J_history)
plt.ylabel('Cost J')
plt.xlabel('Iterations');
plt.title('Cost Function over iterations')
plt.grid(True)

# determine the fit using linear regression classifier from sklearn
from sklearn.linear_model import LinearRegression # data classification, regeression, clustering etc
lr_regr = LinearRegression()
lr_regr.fit(X[:,1].reshape(-1,1), y.ravel())
theta_lr = np.zeros(2)
theta_lr[0] = lr_regr.intercept_
theta_lr[1] = lr_regr.coef_
y_lr = X.dot(theta_lr)
r2_lr = r2_score(y, y_lr) 

print('-----------------------------------------------------------------')
print('\t\t\tFITING RESULTS GRADIENT DESCENT')
print('-----------------------------------------------------------------')
print('\nFitting: Gradient Descent')
print('Learning rate (alpha): %.2f'%(alpha))
print('Number of iterations: %.0f'%(num_iters))
print('Fitting parameters:')
print('\tIntercept %s_0: %.3f'%(sympy.pretty(theta), theta_gd[0]))
print('\tSlope %s_1: %.3f'%(sympy.pretty(theta), theta_gd[1]))
print('Goodness of fit r^2: %.3f'%(r2_gd))

print('-----------------------------------------------------------------')
print('\t\t\tFITING RESULTS SKLEARN')
print('-----------------------------------------------------------------')
print('\nFitting: Linear Regression sklearn library')
print('Fitting parameters:')
print('\tIntercept %s_0: %.3f'%(sympy.pretty(theta), theta_lr[0]))
print('\tSlope %s_1: %.3f'%(sympy.pretty(theta), theta_lr[1]))
print('Goodness of fit r^2: %.3f'%(r2_lr))

# Plot gradient descent & Scikit-learn Linear regression 
plt.figure()
plt.scatter(X[:,1], y, c = 'r', marker = 'x', linewidths = 2)
plt.plot(X[:,1], y_gd, label = 'Linear regression (Gradient descent)') 
plt.plot(X[:,1],y_lr , label = 'Linear regression (Scikit-learn)')
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend()

#%%
# =============================================================================
#                          PROFIT PREDICTION
# =============================================================================

print(' =============================================================================')
print('\t\t\tPROFIT PREDICTION')
print('=============================================================================')

# Predict profit for a city with population of 35000 and 70000
print('\nlcvbnm,./Predict profit for a city with population of:') 
print('Population of %.0f : %.0f' %(35000, theta_gd.T.dot([1, 3.5]) * 10000))
print('Population of %.0f : %.0f' %(70000,theta_gd.T.dot([1, 7]) * 10000))

#%%
# =============================================================================
#                        3-D COST FUNCTION
# =============================================================================

print('=============================================================================')
print('\t\t\t3-D COST FUNCTION')
print('=============================================================================')
# Create grid coordinates for plotting
theta0_axis = np.linspace(-10, 10, 50) # create axis for theta0 grid
theta1_axis = np.linspace(-1, 4, 50)# create axis for theta1 grid

theta0_grid, theta1_grid = np.meshgrid(theta0_axis, theta1_axis, indexing='xy') # create mesh grid of theta0 and theta1
Cost_3d = np.zeros((theta0_axis.size,theta1_axis.size)) # creats grid of zeros from cost values

# Calculate Cost-values based on grid of coefficients
for (i,j),v in np.ndenumerate(Cost_3d): #loop through the Cost grid ... double for loop
    theta_3d = [[theta0_grid[i,j]], [theta1_grid[i,j]]] #create current value of theta
    Cost_3d[i,j] = computeCost(X, y, theta_3d) #create cost for the give values of theta

fig = plt.figure(figsize = (15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(theta0_grid, theta1_grid, Cost_3d, np.logspace(-2, 3, 20), cmap = plt.cm.jet)
ax1.scatter(theta_gd[0],theta_gd[1], c = 'r') # plot the final theta values

# Right plot
ax2.plot_surface(theta0_grid, theta1_grid, Cost_3d, rstride = 1, cstride = 1, alpha = 0.6, cmap = plt.cm.jet)
ax2.set_zlabel('Cost - J')
ax2.set_zlim(Cost_3d.min(), Cost_3d.max())
ax2.view_init(elev = 15, azim = 230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)
