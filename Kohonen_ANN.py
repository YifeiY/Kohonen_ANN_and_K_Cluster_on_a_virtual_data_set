# Yifei Yin 20054101

import csv

def main():
  data = readFile("dataset_noclass.csv")



def readFile(filename):
  file_content = []
  with open(filename) as file:
    for row in file:
      file_content.append(row[:-1].split(','))
  return file_content[1:]


main()