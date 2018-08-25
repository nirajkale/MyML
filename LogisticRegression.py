import numpy as np
import pandas as pd
import os
import DataConditioner as dc
import math
import matplotlib.pyplot as plt

np.random.seed(3)

sigmoid= lambda z: 1/(1+np.exp(-z))

def init_weights(n):
    return np.matrix(np.random.randn(1,n))/10000,0

def Hx(w,b,x):
    return sigmoid(np.dot(w,x)+b)

def Cost(w,b,x,y,lambda1=0.1):
    A = sigmoid(np.dot(w,x)+b)
    m = x.shape[1]
    return np.sum(np.dot(y,np.log(A).T)+np.dot(1-y,np.log(1-A).T))/(-m) + (np.square(w).sum()*lambda1)/(2*m)
    #sum is just for dimensionality reduction

def gradient_descent(w,b,x,y,alpha=0.8,iter_max=2000,lambda1=0.1):
    m = x.shape[1]
    costs=[]
    for iter in range(iter_max):
        A = sigmoid(np.dot(w,x)+b)
        if iter%2==0:
            costs.append(Cost(w,b,x,y,lambda1))
        dw = (np.dot(A-y,x.T) + np.multiply(lambda1,w))/m
        db = np.sum(A-y)/m
        w = w - (alpha*dw)
        b = b - (alpha*db)
    return costs,w,b

def cal_accuracy(w,b,x,y):
    A = Hx(w,b,x)
    A = np.select([A>=0.5,A<0.5],[1,0])
    return (np.select([A==y,A!=y],[1,0]).sum()*100)/A.shape[1]

#region prepare data, Output: x_train,x_test,y_train,y_test,w,b,n
csv=os.path.join(os.path.dirname(__file__),'Datasets','nba.csv')
df= pd.read_csv(csv)
del df['Name']
dc.col_standardize(df,exclude=['TARGET_5Yrs'])
if df.isnull().values.any():
    df=df.replace(to_replace=pd.np.nan,value=0)
#divide data b/w train-test
index_split=math.ceil(df.shape[0]*0.6)
x_train= df[df.index < index_split]
x_test= df[df.index > index_split]
y_train= np.matrix(x_train['TARGET_5Yrs'])
y_test= np.matrix(x_test['TARGET_5Yrs'])
del x_train['TARGET_5Yrs']
del x_test['TARGET_5Yrs']
x_train= np.matrix(x_train).T
x_test= np.matrix(x_test).T
# insert bias in x
x_train= np.insert(x_train,0,1,axis=0)
x_test= np.insert(x_test,0,1,axis=0)
# initialize params
n=x_train.shape[0]
w,b= init_weights(n)
#endregion

print('initial cost='+str(Cost(w,b,x_train,y_train)))
costs,w,b = gradient_descent(w,b,x_train,y_train)
print('final cost='+str(Cost(w,b,x_train,y_train)))
print('Training Accuracy='+str(cal_accuracy(w,b,x_train,y_train)))
print('Testing Accuracy='+str(cal_accuracy(w,b,x_test,y_test)))
plt.plot(costs)
plt.show()


