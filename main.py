# Yifei Yin 20054101

import csv
import random
import math
try:
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np
except: pass

epsilon = -1/5
theta = 1
network_config = [3,2]
threshold = 0.1

def main():
  data = readFile("dataset_noclass.csv")
  ann = makeANN(network_config)

  history = []
  condition = True
  prev_total = 0
  iteration = 0

  while condition:
    total = 0
    iteration += 1

    for row in data:
      feedData(row,ann)
      winner = ann[kohonen(ann[0]['distance'],ann[1]['distance'])]
      total += math.sqrt(ann[0]['distance']) + math.sqrt(ann[1]['distance'])
      updateWeights(row,winner)

    # Termination
    if total == prev_total:
      condition = False

    prev_total = total
    history.append(ann[0]['weights'][0])

  print("Network converged after",iteration,"iteration.")

  try: showData(data,ann)
  except: pass
## END MAIN


## Feed data to the network, output layer stores distance(d_i,w_i)
def feedData(row, ann):
  for node in ann:
    node['distance'] = calculateDistance(node['weights'],row)


## use kohonen inhibition to find the winner node
def kohonen(a,b):
  if a == b: # when a and b are equal, default a to be the winner
    return 0
  while (a != 0 and b != 0):
    old_a = a
    a = max(0, a - 0.5 * b)
    b = max(0, b - 0.5 * old_a)
  if a == 0:
    return 0
  else:
    return 1


## Update weights using dynamic learning rate
def updateWeights(data, node):
  rate = eta(node['distance'])
  for i in range(len(node['weights'])):
    node['weights'][i] += rate * (data[i] - node['weights'][i])
  return sum(node['weights'])


## update the learning rate
def eta(x):
  return min(1,x/10)


## Calculates the Euclidean distance of two vectors
def calculateDistance(a,b):
  square_sum = 0
  for i in range(len(a)):
    square_sum += (a[i] - b[i]) ** 2
  return square_sum


## Build a 2 layer network with bias
def makeANN(config):
  return [{'weights':[random.random()] * config[0],'distance':0},{'weights':[random.random()] * config[0],'distance':0}]


## plot data in a 3D space
def showData(data, ann): # points in classA, points in classB, A center, B center
  plot_data = [[], []]
  for row in data:
    feedData(row, ann)
    plot_data[kohonen(ann[0]['distance'], ann[1]['distance'])].append(row)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for xs, ys, zs in plot_data[0]:
    ax.scatter(xs, ys, zs, c='r', marker='o', s=0.5)
  for xs, ys, zs in plot_data[1]:
    ax.scatter(xs, ys, zs, c='b', marker='o', s=0.5)

  center0 = ann[0]['weights']
  ax.scatter(center0[0], center0[1], center0[2], c='r', marker='*', s=100)
  center1 = ann[1]['weights']
  ax.scatter(center1[0], center1[1], center1[2], c='b', marker='*', s=100)
  plt.show()


## read in file
def readFile(filename):
  file_content = []
  with open(filename) as file:
    next(file)
    for row in file:
      file_content.append([float(i) for i in row[:-1].split(',')])
  return file_content[1:]


main()