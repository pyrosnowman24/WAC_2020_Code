#!/usr/bin/python
import numpy as np
import time
import timeit
import multiprocessing
import random
import sys
from scipy import optimize
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import xarray as xr

"""# Sensor"""

# https://www.sciencedirect.com/science/article/pii/S037704270100485X
class Sensor:
  def sensor(self,pos):
#     return pos[0]**5+pos[1]**5
#     return .88 - (.8*(pos[0]-.75)**2+.8*(pos[1]-.25)**2)
#     return np.exp(pos[0]*pos[1])-1
#     return multivariate_normal.pdf([pos[0],pos[1]],mean=[.1,.8],cov=[[.18,.01],[.01,.18]])
# Franke's function
#     term1 = 0.75 * np.exp(-np.power(9*pos[0]-2,2)/4 - np.power(9*pos[1]-2,2)/4);
#     term2 = 0.75 * np.exp(-np.power(9*pos[0]+1,2)/49 - (9*pos[1]+1)/10);
#     term3 = 0.5 * np.exp(-np.power(9*pos[0]-7,2)/4 - np.power(9*pos[1]-3,2)/4);
#     term4 = -0.2 * np.exp(-np.power(9*pos[0]-4,2) - np.power(9*pos[1]-7,2));
#     return term1+term2+term3+term4
# http://delivery.acm.org/10.1145/310000/305745/p78-renka.pdf?ip=129.115.236.164&id=305745&acc=ACTIVE%20SERVICE&key=F82E6B88364EF649%2E01FBE2B8DA4426C6%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1565281100_395709d94b8ccff767cd9a0b75e4f645
# F8
#     term1 = np.exp(-np.power(5-10*pos[0],2)/2)
#     term2 = .75*np.exp(-np.power(5-10*pos[1],2)/2)
#     term3 = .75*np.exp(-np.power(5-10*pos[0],2)/2)*np.exp(-np.power(5-10*pos[1],2)/2)
#     return term1+term2+term3
# F10
#     term1 = np.exp(-.04*np.sqrt(np.power(80*pos[0]-40,2)+np.power(90*pos[1]-45,2)))
#     term2 = np.cos(.15*np.sqrt(np.power(80*pos[0]-40,2)+np.power(90*pos[1]-45,2)))
#     return term1*term2
# F7
#     term1 = 2*np.cos(10*pos[0])*np.sin(10*pos[1])
#     term2 = np.sin(10*pos[0]*pos[1])
#     return term1+term2
# # F3
#     term1 = 1.25+np.cos(5.4*pos[1])/1
#     term2 = 6+6*np.power(3*pos[0]-1,2)/1
#     return (term1/term2)

# Franke's function
    scalar = .1
    x_trans = 0
    y_trans = 0
    term1 = 0.75 * np.exp(-np.power(9*scalar*pos[0]-2+x_trans,2)/4 - np.power(9*scalar*pos[1]-2+y_trans,2)/4);
    term2 = 0.75 * np.exp(-np.power(9*scalar*pos[0]+1+x_trans,2)/49 - (9*scalar*pos[1]+1+y_trans)/10);
    term3 = 0.5 * np.exp(-np.power(9*scalar*pos[0]-7+x_trans,2)/4 - np.power(9*scalar*pos[1]-3+y_trans,2)/4);
    term4 = -0.2 * np.exp(-np.power(9*scalar*pos[0]-4+x_trans,2) - np.power(9*scalar*pos[1]-7+y_trans,2));
    return term1+term2+term3+term4

"""# Simulated Annealing"""

class Sim_Annealing():
  def __init__(self,iterations = 10,trials = 25,**kwargs):
    self.n = iterations
    self.trials = trials
    self.a = kwargs.pop('a', .5)
    self.sigma = kwargs.pop('sigma', 1)
    self.temp = self.form_Temp(iterations,kwargs.pop('schedule', 'exponential'))
    self.v = kwargs.pop('v', 0)

  def __call__(self,data):
    self.start_time = timeit.default_timer()
    self.data = data
    self.full_model = interp.Rbf(data[:,0],data[:,1],data[:,2], function='cubic', smooth=0)
    current_solution = np.ones((1,data.shape[0]))
    current_cost = 100000000
    cycle = 0
    manager = multiprocessing.Manager()
    results = manager.list(range(self.trials))

    # Begin going through the simulated annealing
    for temp in self.temp: # Runs each set of trials for the current cycle
      if self.v >= 2:print("cycle :",cycle)
      processes = []
      new_w_matrix = np.empty((self.trials,data.shape[0]))
      new_w_matrix[0,:] = self.neighbor(current_solution)
      for i in range(1,self.trials): # creates the weights that will be used for the trials
        new_w_matrix[i,:] = self.neighbor(np.reshape(new_w_matrix[i-1,:],(1,current_solution.shape[1])))

      if self.v >= 3:print("Weights formed",timeit.default_timer()-self.start_time)
      for i in range(self.trials): # calculates the cost of each generated weights
        # results[i] = self.get_cost(new_w_matrix[i,:],current_solution,i,results)
        p = multiprocessing.Process(target = self.get_cost, args=(new_w_matrix[i,:],current_solution,i,results))
        p.start()
        processes.append(p)
      for process in processes:
        process.join()
      if self.v >= 3: print("Costs found",timeit.default_timer()-self.start_time)
      # print(results)

      for i,new_cost in enumerate(results): # works through the costs to decide what to move to
        if new_cost < current_cost:
        # If the cost of the new solution is better, then make it the new solution
          current_solution = np.reshape(new_w_matrix[i,:],(1,current_solution.shape[1]))
          current_cost = new_cost
        else:
          dCost = current_cost - new_cost
          p = np.exp(dCost/temp) # Probability of accepting worse solution
          if random.random()<p:
            # Accept the worse solution
            current_solution = np.reshape(new_w_matrix[i,:],(1,current_solution.shape[1]))
            current_cost = new_cost
      cycle +=1
      if self.v >= 3:print("Cycle complete",timeit.default_timer()-self.start_time)
    if self.v >= 1:print("Final cost:",current_cost)
    self.total_time = timeit.default_timer()-self.start_time
    if self.v >=1:print("Time ellapsed:",timeit.default_timer()-self.start_time)
    return current_solution,current_cost,self.total_time

  def get_cost(self,temp,current_solution,index,results):
      new_solution = self.neighbor(current_solution)
      current_cost = self.cost(current_solution)
      new_cost = self.cost(new_solution)
      results[index] = new_cost
      return new_cost

  def form_Temp(self,step,schedule):
    if schedule == 'exponential':
      exp_func = lambda t: np.power(self.a,t)
      T = exp_func(np.linspace(0,step,step))
    elif schedule == 'linear':
      T = self.sigma*np.linspace(1,0,step)
    elif schedule == 'log':
      log_func = lambda t: 1/(1+self.sigma*np.log(1+t))
      T = log_func(np.linspace(0,step,step))
    elif schedule == 'quadratic':
      quad_func = lambda t: 1/(1+self.sigma*t**2)
      T = quad_func(np.linspace(0,step,step))
    else:
      T = self.form_Temp(step,'exponential')
    T[-1] += .000001
    return T

  def neighbor(self,solution):
    # randomly select value, if 0 switch to 1 and vice versa, only switches 1
    index = random.randint(0,solution.shape[1]-1)
    if solution[:,index] == 0:
      solution[:,index] = 1
    else:
      solution[:,index] = 0
    return solution

  def cost(self,solution):
    # Calculates objective function
    temp_time = timeit.default_timer()
    cost = 0
    sparse_data = np.dot(np.diagflat(solution),self.data)
    sparse_data = sparse_data[sparse_data[:,0]!=0,:]
    sparse_model = interp.Rbf(sparse_data[:,0],sparse_data[:,1],sparse_data[:,2], function='cubic', smooth=0)
    # print("Time to find model",timeit.default_timer()-temp_time)
    cost = np.power(np.linalg.norm(self.gen_values(self.full_model) - self.gen_values(sparse_model),2),2)
    # print("Time to find l2",timeit.default_timer()-temp_time)
    cost += self.sigma * np.linalg.norm(solution,1)
    # print("Time to find cost",timeit.default_timer()-temp_time)
    return cost

  def gen_values(self,model):
    temp_time = timeit.default_timer()
    # creates matrix from model for error to be calculated from
    min_x = 0
    max_x = 15
    min_y = 0
    max_y = 15
    grid_x,grid_y = np.mgrid[min_x:max_x:50j,min_y:max_y:50j]
    # print("Time to find values",timeit.default_timer()-temp_time)
    return np.reshape(model(grid_x, grid_y),(-1,1))

"""# Optimization Test"""

max = 10
min = 0
size = 200
simulated_data = np.zeros((size,3))
bigS = Sensor()
for i in range(0,simulated_data.shape[0]):
  x = random.uniform(min,max)
  y = random.uniform(min,max)
  simulated_data[i] = [x,y,bigS.sensor((x,y))]
simulated_model = interp.Rbf(simulated_data[:,0],simulated_data[:,1],simulated_data[:,2], function='cubic', smooth=0)
################################################
def new_data(amount = 200):
  max = 10
  min = 0
  size = amount
  simulated_data = np.zeros((size,3))
  bigS = Sensor()
  for i in range(0,simulated_data.shape[0]):
    x = random.uniform(min,max)
    y = random.uniform(min,max)
    simulated_data[i] = [x,y,bigS.sensor((x,y))]
  return simulated_data

amountPointArray = [50,100,200,400]
# schedules = ['exponential','linear','log','quadratic']
# epochs = [5,10,25]
# trials = [5,10,25]
# amount = 20
# labels = ['points','error','time']
schedules = ['exponential','linear']
epochs = [10]
trials = [5,25]
amount = 20
labels = ['points','error','time']


x = np.zeros((len(amountPointArray),len(schedules),len(epochs),len(trials),amount,len(labels)))
db = xr.Dataset({"data":(("points","schedule","epoch","trial","iteration","labels"),x)},{"points":amountPointArray,"schedule":schedules,"epoch":epochs,"trial":trials,"iteration":np.linspace(0,amount-1,amount),"labels":labels})
for amount_point in amountPointArray :
    simulated_data = new_data(amount_point)
    for schedule in schedules:
      for epoch in epochs:
        for trial in trials:
          print "Data Size:",amount_point,"schedule:",schedule,"epochs:",epoch,"trials:",trial
          points = np.zeros(amount)
          errors = np.zeros(amount)
          times = np.zeros(amount)
          for i in range(amount):
            hi = Sim_Annealing(iterations = epoch,trials = trial, schedule = schedule, sigma = 1, v = 0)
            simulated_weights,simulated_cost,simulated_time = hi(simulated_data)
            simulated_weighted_data = np.dot(np.diagflat(simulated_weights),simulated_data)
            simulated_new_data = simulated_weighted_data[simulated_weighted_data[:,0]>.01,:]
            times[i] = float(simulated_time)
            points[i] = simulated_data.shape[0]-simulated_new_data.shape[0]
            errors[i] = simulated_cost
          db["data"].loc[dict(points=amount_point,schedule = schedule,epoch = epoch,trial = trial,labels = "points")] = points
          db["data"].loc[dict(points=amount_point,schedule = schedule,epoch = epoch,trial = trial,labels = "error")] = errors
          db["data"].loc[dict(points=amount_point,schedule = schedule,epoch = epoch,trial = trial,labels = "time")] = times

db.to_netcdf('results.nc')
