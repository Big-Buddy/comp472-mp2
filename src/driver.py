import sys
from sklearn import tree

ds_option = sys.argv[1]
fp        = "./data/ds" + str(ds_option) + "/ds" + str(ds_option) + "Train.csv"

with open(fp, 'r') as file:
	data = [line.split(',') for line in file.read().split('\n')][1:]

data = [[int(element) for element in row] for row in data]
features = [d[:-1] for d in data]
labels = [d[-1] for d in data
