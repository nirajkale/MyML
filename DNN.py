import numpy as np
import pandas as pd
import os
import DataConditioner as dc
import math
import matplotlib.pyplot as plt

np.random.seed(3)

getshape = lambda arr: [item.shape for item in arr]
getparam = lambda dict,param: [dict[param] if param in dict.keys() else None][0] 
getparam_default = lambda dict,param,default: [dict[param] if param in dict.keys() else default][0] 
#region Activation functions
sigmoid = lambda z: 1/(1+np.exp(-z))
tanh = lambda z: (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
ReLU = lambda z: np.maximum(0,z)
LeakyReLU = lambda z: np.maximum(0.1*z,z)
#endregion

#region Gradients of Activation functions
def sigmoid_prime(z):
    a=sigmoid(z)
    return np.multiply(a,(1-a))

tanh_prime = lambda z: 1-np.square(tanh(z))
ReLU_prime = lambda z: np.where(z>0,1,0)
LeakyReLU_prime = lambda z: np.where(z>0,1,0.1) 
#endregion

def init_weights(layers=[],multiplier=0.01):
    w = []
    b = []
    for dims in zip(layers,layers[1:]):
        w.append(np.matrix(np.random.randn(dims[1],dims[0]))*multiplier)
        b.append(np.matrix(np.random.randn(dims[1],1))*multiplier)
    return w,b

cost = lambda a,y,**kwargs:_cost(a,y,kwargs)

def _cost(a,y,kwargs):
    m = a.shape[1]
    cost,cost_reg=np.sum(np.dot(y,np.log(a).T)+np.dot(1-y,np.log(1-a).T))/(-m),0
    if getparam(kwargs,'regularization'):
        cost_reg=cost + getparam(kwargs,'lambd')* np.sum([np.sum(np.square(weights)) for weights in getparam(kwargs,'w')])
    return cost,cost_reg

def forward_prop(w,b,x,activations,cache={},kwargs={}):
    cache['A']=[x]
    cache['Z']=[]
    cache['dl']= []
    keep_prob = getparam_default(kwargs,'Dropout_keep_prob',0)
    for weights,bias,g in zip(w,b,activations):
        cache['Z'].append(np.dot(weights,cache['A'][-1])+bias)
        cache['A'].append(eval(g)(cache['Z'][-1]))
        if getparam(kwargs,'Dropout'):
            assert(keep_prob!=0) # to avoid divide by zero error
            #prepare 
            cache['dl'].append( np.matrix(np.random.randn(dims[1],dims[0]))<keep_prob )
            #shunt out neuron output for dropout regularization
            cache['A'][-1] = np.multiply(cache['dl'],cache['A'][-1])
            #bump the neuron output to neutralize loss in input to next layer
            cache['A'][-1] /= keep_prob 
    return cache

def backward_prop(w,b,x,y,activations,cache={},kwargs={}):
    m = x.shape[1]
    dAL = (-y/cache['A'][-1])+(1-y)/(1-cache['A'][-1])
    #init cache for gradients
    cache['dA'],cache['dZ'],cache['dW'],cache['db']=[dAL],[],[],[]
    keep_prob = getparam_default(kwargs,'Dropout_keep_prob',0)
    lambd = getparam_default(kwargs,'lambd',0)
    for layer in reversed(range(len(w))):
        #for each layer calculate gradients
        cache['dZ'].append( np.multiply(cache['dA'][-1], eval(activations[layer]+'_prime')(cache['Z'][layer]) ) )
        cache['dW'].append( np.dot(cache['dZ'][-1],cache['A'][layer].T)/m + np.multiply(lambd,w[layer])/m )
        cache['db'].append( np.sum(cache['dZ'][-1],axis=1)/m ) 
        if(layer!=0):
            cache['dA'].append( np.dot(w[layer].T,cache['dZ'][-1]) )
            if getparam(kwargs,'Dropout'):
                cache['dA'][-1] = np.multiply(cache['dA'],cache['dl'][layer])
                cache['dA'][-1] /= keep_prob
    #update all the parameters
    return cache

def gradient_descent(w,b,x,y,activations,alpha=2.4,iter_max=10000,**kwargs):
    assert(len(w)==len(activations) and len(w)==len(b))
    costs,costs_reg=[],[]
    kwargs['w']=w # to fwd to cost function
    print('GD with:'+str(alpha))
    cost_step = getparam_default(kwargs,'cost_step',1)
    per,per_temp=0,0
    for i in range(iter_max):
        cache={}
        cache= forward_prop(w,b,x_train,activations)
        if i==0:
            print('Initial Cost: '+str(_cost(cache['A'][-1],y,kwargs)))
        if i%cost_step==0:
            cost=_cost(cache['A'][-1],y,kwargs)
            costs.append(cost[0])
            costs_reg.append(cost[1])
        per_temp = i*100/iter_max
        if per_temp-per>1:
            per=per_temp
            print('Iterations:'+str(round(per,2))+'%')
        cache= backward_prop(w,b,x_train,y_train,activations,cache,kwargs)
        #update weights & bias
        L=len(w)
        for layer in range(L):
            w[layer]= w[layer]-(alpha*cache['dW'][L-layer-1])
            b[layer]= b[layer]-(alpha*cache['db'][L-layer-1])
    print('GD Completed 100%')
    if getparam(kwargs,'print_costs'):
        plt.subplot(2,1,1)
        plt.plot(costs)
        plt.subplot(2,1,2)
        plt.plot(costs_reg)
        plt.show()
    print('Final Cost: '+str(_cost(cache['A'][-1],y,kwargs)))
    return cache,w,b,costs

def calculate_accuracy(y,a,approx_threshold=0.5):
    a = np.select([a >= approx_threshold, a<approx_threshold],[1,0])
    return (np.select([a==y, a!=y],[1,0]).sum()*100)/a.shape[1]

def calculate_accuracy_prop(w,b,x,y,activations,approx_threshold=0.5):
    cache={}
    cache= forward_prop(w,b,x,activations)
    return calculate_accuracy(y,cache['A'][-1],approx_threshold)

layers=[20,5,1]
#region prepare data, Output: x_train,x_test,y_train,y_test,layers,w,b,n
csv=os.path.join(os.path.dirname(__file__),'Datasets','nba.csv')
df= pd.read_csv(csv)
del df['Name']
dc.col_standardize(df,exclude=['TARGET_5Yrs'])
if df.isnull().values.any():
    df=df.replace(to_replace=pd.np.nan,value=0)
#divide data b/w train-test
index_split=math.ceil(df.shape[0]*0.8)
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
w,b= init_weights(layers)
#endregion

activations=['sigmoid','sigmoid']
cache,w,b,costs = gradient_descent(w,b,x_train,y_train,activations,alpha=2.4,iter_max=15000
                                   ,regularization=True,lambd=1.5,
                                   print_costs=True)
print('Training Accuracy: '+str(calculate_accuracy(y_train,cache['A'][-1])))
print('Testing Accuracy: '+str(calculate_accuracy_prop(w,b,x_test,y_test,activations)))
print('done')