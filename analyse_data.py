#!/usr/bin/python
import pandas as pd
import numpy as np
import xarray as xr
import math
import sys
import matplotlib.pyplot as plt

ds = xr.open_dataset('data.nc')

schedules = ['exponential','linear','log','quadratic']
epochs = [5,10,25]
trials = [5,10,25]
amount = 20
labels = ['points','error','time']
points_array = np.array([], dtype=np.int64).reshape(0,20)
errors_average_array = np.array([], dtype=np.int64).reshape(0,9)
times_average_array = np.array([], dtype=np.int64).reshape(0,9)
for i,schedule in enumerate(schedules):
  errors_average_temp = np.zeros((0,len(epochs)*len(trials)))
  times_average_temp = np.zeros((0,len(epochs)*len(trials)))
  for j,epoch in enumerate(epochs):
    for k,trial in enumerate(trials):
      x = ds["data"].loc[dict(schedule = schedule,epoch = epoch,trial = trial)].data
      # print("schedule:",schedule,"epoch:",epoch,"trial:",trial,'\n',x[:,1],'\n')
      error_average = sum(x[:,1])/20
      time_average = sum(x[:,2])/20
      errors_average_temp = np.append(errors_average_temp,error_average)
      times_average_temp = np.append(times_average_temp,time_average)
      points_array = np.vstack((points_array,x[:,0]))
  errors_average_array = np.vstack((errors_average_array,errors_average_temp))
  times_average_array = np.vstack((times_average_array,times_average_temp))
# print(times_average_array)
avg_time = [0,0]
avg_start_time = [0,0,0]
for row in times_average_array:
    avg_start_time = np.add([row[0],row[3],row[6]],avg_start_time)
    avg_time = np.add([row[1]/row[0],row[2]/row[0]],avg_time)
    avg_time = np.add([row[4]/row[3],row[5]/row[3]],avg_time)
    avg_time = np.add([row[7]/row[6],row[8]/row[6]],avg_time)
print(np.divide(avg_start_time,4))
print(np.divide(avg_time,12))
fig,ax = plt.subplots(2,2)
medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
labels = ['5,5','5,10','5,25','10,5','10,10','10,25','25,5','25,10','25,25',]
ax[0,0].boxplot(points_array[:9,:].T,medianprops=medianprops,labels = labels)
ax[0,0].set_title('Exponential')
ax[0,0].grid(True)
ax[0,1].boxplot(points_array[9:18,:].T,medianprops=medianprops,labels = labels)
ax[0,1].set_title('Linear')
ax[0,1].grid(True)
ax[1,0].boxplot(points_array[18:27,:].T,medianprops=medianprops,labels = labels)
ax[1,0].set_title('Logarithmic')
ax[1,0].grid(True)
ax[1,1].boxplot(points_array[27:,:].T,medianprops=medianprops,labels = labels)
ax[1,1].set_title('Quadratic')
ax[1,1].grid(True)
plt.show()
fig,ax = plt.subplots(1,1)
width = .10
x = np.arange(len(labels))  # the label locations
rects1 = ax.bar(x - 3*width/2, errors_average_array[0,:], width, label='Exponential')
rects2 = ax.bar(x - width/2, errors_average_array[1,:], width, label='Linear')
rects3 = ax.bar(x + width/2, errors_average_array[2,:], width, label='Logarithmic')
rects4 = ax.bar(x + 3*width/2, errors_average_array[3,:], width, label='Quadratic')
ax.set_ylabel('Errors')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()
