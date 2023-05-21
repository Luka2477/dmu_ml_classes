import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import sys

data = pd.read_csv('../data/TSPcities1000.txt', sep='\s+', header=None)
data = pd.DataFrame(data)

x = data[1].to_numpy()
y = data[2].to_numpy()


def createRandomRoute():
    tour = [[i] for i in range(ROUTE_STOPS)]
    random.shuffle(tour)
    return tour


# plot the tour - Adjust range 0..len, if you want to plot only a part of the tour.
def plotCityRoute(route):
    for i in range(len(route) - 1):
        plt.plot([x[route[i][0]], x[route[i + 1][0]]], [y[route[i][0]], y[route[i + 1][0]]], 'ro-')
    # Add the line connecting the last city to the first city to complete the tour
    plt.plot([x[route[-1][0]], x[route[0][0]]], [y[route[-1][0]], y[route[0][0]]], 'ro-')
    plt.show()


def generate_tours(size):
    tours = []
    for _ in range(size):
        individual = createRandomRoute()
        tours.append(individual)
    return tours


def distance_of_tour(tour):
    distance = 0
    for i in range(len(tour) - 1):
        distance += distancebetweenCities(tour[i], tour[i + 1])
    # Add distance from last city back to the first city
    distance += distancebetweenCities(tour[-1], tour[0])
    return distance


def distancebetweenCities(city1, city2):
    xDistance = abs(x[city1] - (x[city2]))
    yDistance = abs(y[city1] - (y[city2]))
    distance = math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
    return distance


def select_parents(tours):
    parent1 = random.choice(tours)
    parent2 = random.choice(tours)
    if distance_of_tour(parent1) <= distance_of_tour(parent2):
        return parent1
    else:
        return parent2


def crossover(parent1, parent2):
    return ordered_crossover(parent1, parent2)


def ordered_crossover(parent1, parent2):
    size = len(parent1)

    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end] = parent1[start:end]

    pointer = end
    for i in range(size):
        parent2_pointer = (i + end) % size
        if parent2[parent2_pointer] not in child:
            child[pointer] = parent2[parent2_pointer]
            pointer = (pointer + 1) % size
    return child


def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)

        if i > j:
            i, j = j, i

        individual[i:j] = individual[i:j][::-1]

    return individual


def genetic_algorithm(population_size, generations):
    tours = generate_tours(population_size)
    best_individual = None
    best_fitness = sys.maxsize
    for generation in range(generations):
        new_population = []
        for i in range(population_size):
            parent1 = select_parents(tours)
            parent2 = select_parents(tours)
            child = crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)

        tours = new_population

        curr_best = min(tours, key=distance_of_tour)
        curr_fitness = distance_of_tour(curr_best)
        if curr_fitness < best_fitness:
            best_individual = curr_best
            best_fitness = curr_fitness

        if generation % 2 == 0:
            print(f"Generation {generation}:"
                  #   f" Best solution: {best_individual},"
                  f" Fitness: {best_fitness}")

    return best_individual, best_fitness


# fake_tour = [[1], [2], [6], [50]]
# print(distance_of_tour(fake_tour))
# plotCityRoute(fake_tour)

# Here comes your GA program...
POP_SIZE = 500
MUTATION_RATE = 0.01
ROUTE_STOPS = 30

best_solution, best_fitness = genetic_algorithm(population_size=POP_SIZE, generations=200)

print(f"\nBest Solution: {best_solution}\nBest Fitness: {best_fitness}")
plotCityRoute(best_solution)
