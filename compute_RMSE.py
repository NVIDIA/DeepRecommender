# Copyright (c) 2017 NVIDIA Corporation
import argparse
from math import sqrt

parser = argparse.ArgumentParser(description='RMSE_calculator')

parser.add_argument('--path_to_predictions', type=str, default="", metavar='N',
                    help='Path file with actual ratings and predictions')
parser.add_argument('--round', action='store_true',
                    help='round predictions to nearest')

args = parser.parse_args()
print(args)

def main():
  with open(args.path_to_predictions, 'r') as inpt:
    lines = inpt.readlines()
    n = 0
    denom = 0.0
    for line in lines:
      parts = line.split('\t')
      prediction = float(parts[2]) if not args.round else round(float(parts[2]))
      rating = float(parts[3])
      denom += (prediction - rating)*(prediction - rating)
      n += 1
  print("####################")
  print("RMSE: {}".format(sqrt(denom/n)))
  print("####################")

if __name__ == '__main__':
  main()