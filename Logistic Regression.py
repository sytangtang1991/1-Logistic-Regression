#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 07:41:36 2019

@author: yangsong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 07:54:04 2019

@author: yangsong
"""


from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


########################
# Load data
########################
iris = datasets.load_iris()
X=iris.data[0:99,:2]
y=iris.target[0:99]

# Plot the training points
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)




#Step 1: Initial Model Parameter
Learning_Rate=0.01
num_iterations=100000
N=len(X)

w=np.zeros((2,1))
b=0
costs=[]


for i in range(num_iterations):
    #Step 2: Apply sigmoid Function and get y prediction
    Z=np.dot(w.T,X.T)+b
    y_pred=1/(1+1/np.exp(Z))
    
    #Step 3: Calculate Loss Function
    cost=-(1/N)*np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
    
    #Step 4: Calculate Gradient
    dw=1/N*np.dot(X.T,(y_pred-y).T)
    db=1/N*np.sum(y_pred-y)
    
    #Step 5: Update w & b
    w = w - Learning_Rate*dw
    b = b - Learning_Rate*db
    
    #Records cost
    if i%100==0:
        costs.append(cost)
        print(cost)
        
        
# Plot cost function        
Epoch=pd.DataFrame(list(range(100,100001,100)))
Cost=pd.DataFrame(costs)
Cost_data=pd.concat([Epoch, Cost], axis=1)        
Cost_data.columns=['Epoch','Cost']
plt.scatter(Cost_data['Epoch'], Cost_data['Cost'])
plt.xlabel('Epoch')
plt.ylabel('Cost')


# Plot linear classification
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
line=mlines.Line2D([3.701,7],[2,4.1034],color='red')
ax.add_line(line)
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
plt.show()
