# Yifei Yin 20054101

import random


try:
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np
except: pass


epsilon = -1/5
theta = 1
network_config = [3,2]
threshold = 0.1
output_filename = "output.txt"


def main():

  header,data = readFile("dataset_noclass.csv")
  header.append("kohonen")
  header.append("K-Means")

  kohonenANN_out = kohonenANN(data)

  kMeans_out = kMeans(data)

  writeOutput([header] + [data[i] + [kohonenANN_out[i],kMeans_out[i]] for i in range(len(data))])
## END MAIN


## Use K-Means to identify two clustering
def kMeans(data):

  # choose two points from the dataset to be the initial center points
  w1,w2 = data[0],data[len(data)//2]

  err = 0
  prev_err = err + 1
  iteration = 0

  while err != prev_err:
    prev_err = err

    err = 0
    iteration += 1

    c1 = []
    c2 = []
    c1_sum = [0] * 3
    c2_sum = [0] * 3

    for row in data:
      d1 = calculateDistance(w1,row)
      d2 = calculateDistance(w2,row)

      if d1 < d2:
        c1.append(row)
        c1_sum = [c1_sum[i] + row[i] for i in range(len(row))]
      else:
        c2.append(row)
        c2_sum = [c2_sum[i] + row[i] for i in range(len(row))]


    w1 = [i/len(c1) for i in c1_sum]
    w2 = [i/len(c1) for i in c2_sum]


    err += sum([d ** 2 for d in [p[j] - w1[j] for j in range(len(w1)) for p in c1]])
    err += sum([d ** 2 for d in [p[j] - w1[j] for j in range(len(w2)) for p in c2]])

  print("K-Means stabilized after", iteration, "iteration.\n")

  output = []
  for row in data:
    output.append("A") if calculateDistance(w1, row) < calculateDistance(w2, row) else output.append("B")
  return output


## Use kohonen ANN to identify two clustering
def kohonenANN(data):
  ann = makeANN(network_config)

  err = 0
  prev_err = err + 1
  iteration = 0

  while err != prev_err:
    err = 0
    iteration += 1

    for row in data:
      feedData(row, ann)
      winner = ann[kohonen(ann[0]['distance'], ann[1]['distance'])]
      err += ann[0]['distance'] + ann[1]['distance']
      updateWeights(row, winner)

    prev_err = err

  print("Network converged after", iteration, "iteration.\n")

  output = []
  for row in data:
    feedData(row, ann)
    output.append('A' if kohonen(ann[0]['distance'], ann[1]['distance']) == 0 else 'B')
  #showData(data, ann)
  return output



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


## Calculates the square of Euclidean distance of two vectors
def calculateDistance(a,b):
  square_sum = 0
  for i in range(len(a)):
    square_sum += (a[i] - b[i]) ** 2
  return square_sum


## Build a 2 layer network with bias
def makeANN(config):
  return [{'weights':[random.random()] * config[0],'distance':0},{'weights':[random.random()] * config[0],'distance':0}]


## plot data in a 3D space
## This funuction is not called in my submission
def showData(data, ann):
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
  plt.show(block = False)
  plt.show()


## write output file
def writeOutput(content,filename = output_filename):
  file = open(filename, "w")
  for row in content:
    line = ""
    for item in row:
      line += str(item) + '\t'
    file.write(line + '\n')
  print("output has been written to " + "\"" + output_filename + "\"." )

## read in file
def readFile(filename):
  file_content = []
  with open(filename) as file:
    header = next(file)[:-1].split(',')
    for row in file:
      file_content.append([float(i) for i in row[:-1].split(',')])

  return [h + '\t' for h in header],file_content[1:]


main()