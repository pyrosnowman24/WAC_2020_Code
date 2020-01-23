#!/usr/bin/python
import pandas as pd
import numpy as np
import xarray as xr
import math
import sys
import matplotlib.pyplot as plt

ds = xr.open_dataset('data2.nc')

schedules = ['exponential','linear']
epochs = [10]
trials = [5,25]
amount = 20
labels = ['points','error','time']
amountPointArray = [50,100,200,400]
points_array = np.array([], dtype=np.int64).reshape(0,amount)
errors_average_array = np.array([], dtype=np.int64).reshape(0,len(epochs)*len(trials))
times_average_array = np.array([], dtype=np.int64).reshape(0,len(epochs)*len(trials))
for amount_point in amountPointArray :
    for i,schedule in enumerate(schedules):
      errors_average_temp = np.zeros((0,len(epochs)*len(trials)))
      times_average_temp = np.zeros((0,len(epochs)*len(trials)))
      for j,epoch in enumerate(epochs):
        for k,trial in enumerate(trials):
          x = ds["data"].loc[dict(points = amount_point,schedule = schedule,epoch = epoch,trial = trial)].data
          # print("points",amount_point,"schedule:",schedule,"epoch:",epoch,"trial:",trial,'\n',x,'\n')
          error_average = sum(x[:,1])/20
          time_average = sum(x[:,2])/20
          errors_average_temp = np.append(errors_average_temp,error_average)
          times_average_temp = np.append(times_average_temp,time_average)
          points_array = np.vstack((points_array,x[:,0]))
      errors_average_array = np.vstack((errors_average_array,errors_average_temp))
      times_average_array = np.vstack((times_average_array,times_average_temp))
fig,ax = plt.subplots(2,1)
medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
labels = ['50,10,5','50,10,25','100,10,5','100,10,25','200,10,5','200,10,25','400,10,5','400,10,25']
ax[0].boxplot(points_array[:8,:].T,medianprops=medianprops,labels = labels)
ax[0].set_title('Exponential')
ax[0].grid(True)
ax[1].boxplot(points_array[8:16,:].T,medianprops=medianprops,labels = labels)
ax[1].set_title('Linear')
ax[1].grid(True)
plt.show()
fig,ax = plt.subplots(1,1)
width = .10
x = np.arange(len(labels))  # the label locations
rects1 = ax.bar(x - 3*width/2, errors_average_array.T[0,:], width, label='Exponential')
rects2 = ax.bar(x - width/2, errors_average_array.T[1,:], width, label='Linear')
ax.set_ylabel('Errors')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()
print(times_average_array)
