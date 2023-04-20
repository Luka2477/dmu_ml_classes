import matplotlib.pyplot as plt
import pandas as pd
import random
import math


def create_random_route():
    tour = [[i] for i in range(1000)]
    random.shuffle(tour)
    return tour


# plot the tour - Adjust range 0..len, if you want to plot only a part of the tour.
def plot_city_route(route):
    for i in range(0, len(route)):
        plt.plot(x[i:i + 2], y[i:i + 2], 'ro-')
    plt.show()


# calculate distance between cities
def distance_between_cities(city1x, city1y, city2x, city2y):
    xDistance = abs(city1x - city2x)
    yDistance = abs(city1y - city2y)
    distance = math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
    return distance


data = pd.read_csv('data/TSPcities1000.txt', sep='\s+', header=None)
data = pd.DataFrame(data)

x = data[1]
y = data[2]
plt.plot(x, y, 'r.')
plt.show()


# Alternativ kode:
#  for i in range(0, len(route)-1:
#     plt.plot([x[route[i]],x[route[i+1]]], [y[route[i]],y[route[i+1]]], 'ro-')

tour = create_random_route()
print(tour)
plot_city_route(tour)

# distance between city number 100 and city number 105
dist = distance_between_cities(x[100], y[100], x[105], y[105])
print('Distance, % target: ', dist)

best_score_progress = []  # Tracks progress

# replace with your own calculations
fitness_gen0 = 1000  # replace with your value
print('Starting best score, % target: ', fitness_gen0)

best_score = fitness_gen0
# Add starting best score to progress tracker
best_score_progress.append(best_score)

# Here comes your GA program...
best_score = 980
best_score_progress.append(best_score)
best_score = 960
best_score_progress.append(best_score)

# GA has completed required generation
print('End best score, % target: ', best_score)

plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best Fitness - route length - in Generation')
plt.show()
