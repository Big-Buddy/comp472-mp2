from sklearn import naive_bayes

with open(, 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][1:]
data = [[int(element) for element in row] for row in data]
features = [d[:-1] for d in data]
labels = [d[-1] for d in data]