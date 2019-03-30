# -*- coding: utf-8 -*-

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('train35.digits')
labels = np.loadtxt('train35.labels')
def dist(a, b, ax=0):
    return np.linalg.norm(b - a, axis=ax)
v0=np.zeros(784)
#normalize data
for i in range(2000):
  data[i]=np.divide(data[i],dist(v0,data[i]))
store=deepcopy(data)
#split data for traning and validation
data_list=np.split(data,[1700,2000])
data_train=data_list[0]
data_vali=data_list[1]
labels_list=np.split(labels,[1700,2000])
labels_train=labels_list[0]
labels_vali=labels_list[1]
w=np.zeros(784)
#use cross validation to set M
vali_error=0
old=1
M=0
while old!=vali_error:  
  M=M+1
  if M==10:
    break
  for i in range(1700):
    dot=np.dot(data_train[i],w)
    if dot>=0:
      if labels_train[i]==-1:
        w=w-data_train[i]
    else:
      if labels_train[i]==1:
        w=w+data_train[i]
  old=vali_error
  vali_error=0
  for i in range(300):
    dot=np.dot(data_vali[i],w)
    if dot>=0:
      if labels_train[i]==-1:
        vali_error=vali_error+1
    else:
      if labels_train[i]==1:
        vali_error=vali_error+1

#have found M, begin to use all data to train the parameter
w=np.zeros(784)
error_sequence=np.zeros(M*2000)
error=0
for j in range(M):  
  index=np.arange(2000)
  np.random.shuffle(index)
  for i in range(2000):
    dot=np.dot(store[index[i]],w)
    if dot>=0:
      if labels[index[i]]==-1:
        error=error+1      
        w=w-store[index[i]]
    else:
      if labels[index[i]]==1:
        error=error+1
        w=w+store[index[i]]
    error_sequence[j*2000+i]=error
#plot the figure with x-axis number of examples and y-axis number of mistakes
temp=range(2000*M)
plt.plot(temp, error_sequence)
plt.savefig('Mistakes versus examples')
plt.show()
      
test=np.loadtxt('test35.digits')
labels_test=np.zeros(200)
for i in range(200):
  dot=np.dot(test[i],w)
  if dot>=0:
    labels_test[i]=1
  else:
    labels_test[i]=-1
np.savetxt('test labels',labels_test)