from discretization import *
from unic_clustering import *
import csv


x = []
previous = 0
for i in range(101):
    x.append(previous + random.randint(0, 10))
    previous = previous + random.randint(0, 10)

# x = [1, 3, 6, 7, 8, 9.5, 10, 11]
x = []
with open('target.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        x.append(float(row[0]))

discretization(x)

"""
int1 = equal_width_intervals(x, 3)

int2 = equal_frequency_intervals(x, 3)

int3 = kmeans_clustering(np.reshape(x, (-1, 1)), 3)

cl1 = unic_algorithm(x)

print(cl1)
"""


