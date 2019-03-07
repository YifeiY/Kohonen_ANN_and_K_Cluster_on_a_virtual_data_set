# Yifei Yin 20054101

import csv
import random

network_config = [3,2]
def main():
  data = readFile("dataset_noclass.csv")
  network = make_ANN(network_config)


  condition = True
  while condition:
    for row in data:






def feed_data(data, ann):
  for neuron in ann:
    neuron['distance'] = 0
    for i in range(len(neuron['weights'])):
      neuron['distance'] += neuron['weights'][i] * data[i]



## Build a 2 layer network with bias
def make_ANN(config):
  return [{'weights':[random.random()] * config[0],'distance':0}] * config[1]


def readFile(filename):
  file_content = []
  with open(filename) as file:
    for row in file:
      file_content.append(row[:-1].split(','))
  return file_content[1:]


main()