import random as module_random
import string as module_string
import time as module_time

# Constants
FINAL_STRING = "To be or not to be that is the question"
FINAL_CHARS = {'T': 1, 'o': 5, ' ': 9, 'b': 2, 'e': 4, 'r': 1, 'n': 2, 't': 6, 'h': 2, 'a': 1, 'i': 2, 's': 2, 'q': 1,
               'u': 1}
LETTERS = module_string.ascii_letters + " "

# Types
FitnessType = tuple[str, list[int], int]
GenesType = list[str]


class Chromosome:
    genes: GenesType
    fitness: int

    def set_genes(self: "Chromosome", genes: GenesType) -> "Chromosome":
        self.genes = genes
        return self

    def generate_gene(self: "Chromosome", index: int) -> None:
        self.genes[index] = module_random.choice(LETTERS)

    def generate_genes(self: "Chromosome") -> "Chromosome":
        self.genes = [module_random.choice(LETTERS) for _ in range(len(FINAL_STRING))]
        return self

    def calc_fitness(self: "Chromosome") -> "Chromosome":
        self.fitness = 0
        chars = FINAL_CHARS.copy()

        for i in range(len(FINAL_STRING)):
            cs = self.genes[i]
            ds = FINAL_STRING[i]

            if cs == ds:
                self.fitness += 5
                chars[cs] = max(0, chars[cs] - 1)
            elif chars.get(cs) not in [None, 0]:
                self.fitness += 1
                chars[cs] -= 1

        return self

    def get_string(self: "Chromosome") -> str:
        return "".join(self.genes)


class Population:
    size: int
    chromosomes: list[Chromosome]

    def __init__(self: "Population", size: int):
        self.size = size

    def init_population(self: "Population") -> None:
        self.chromosomes = [Chromosome().generate_genes() for _ in range(self.size)]

    def calc_fitness(self: "Population") -> None:
        for chromosome in self.chromosomes:
            chromosome.calc_fitness()
        self.chromosomes.sort(key=lambda c: c.fitness, reverse=True)


class GeneticAlgo:
    population_size: int
    population: Population
    selection_percentage: float
    selection_size: int
    chromosome_mutation_rate: float
    gene_mutation_rate: float
    selection: list[Chromosome]
    crossover: list[Chromosome]
    generation: int

    def __init__(self: "GeneticAlgo", population_size: int = 100, selection_percentage: float = 0.2,
                 chromosome_mutation_rate: float = 0.05, gene_mutation_rate: float = 0.1) -> None:
        self.population_size = population_size
        self.population = Population(population_size)
        self.selection_percentage = selection_percentage
        self.selection_size = int(
            self.population_size * self.selection_percentage - self.population_size * self.selection_percentage % 2)
        self.chromosome_mutation_rate = chromosome_mutation_rate
        self.gene_mutation_rate = gene_mutation_rate
        self.generation = 1

    def calc_selection(self: "GeneticAlgo") -> None:
        self.selection = [Chromosome().set_genes(chromosome.genes) for chromosome in
                          self.population.chromosomes[:self.selection_size]]

    def calc_crossover(self: "GeneticAlgo") -> None:
        self.crossover = list()
        for i in range(0, self.selection_size, 2):
            temp = Chromosome().set_genes(self.selection[i].genes)
            chromosome1 = self.selection[i]
            chromosome2 = self.selection[i + 1]
            crossover_point = module_random.randint(0, len(chromosome1.genes))
            chromosome1.genes = chromosome2.genes[:crossover_point] + chromosome1.genes[
                                                                      min(crossover_point, len(chromosome1.genes)):]
            chromosome2.genes = temp.genes[:crossover_point] + chromosome2.genes[
                                                               min(crossover_point, len(chromosome2.genes)):]
            self.crossover.append(chromosome1)
            self.crossover.append(chromosome2)

    def calc_mutation(self: "GeneticAlgo") -> None:
        for chromosome in self.crossover:
            chromosome_should_mutate = module_random.random() <= self.chromosome_mutation_rate
            if chromosome_should_mutate:
                for i in range(len(chromosome.genes)):
                    gene_should_mutate = module_random.random() <= self.gene_mutation_rate
                    if gene_should_mutate:
                        chromosome.generate_gene(i)

    def new_generation(self: "GeneticAlgo") -> None:
        self.population.chromosomes = self.population.chromosomes[
                                      :self.population_size - self.selection_size] + self.crossover
        self.generation += 1

    def get_best(self: "GeneticAlgo") -> Chromosome:
        return self.population.chromosomes[0]


if __name__ == '__main__':
    st = module_time.time()

    genetic_algo = GeneticAlgo(population_size=1000, chromosome_mutation_rate=0.5, gene_mutation_rate=0.5,
                               selection_percentage=0.5)
    genetic_algo.population.init_population()
    genetic_algo.population.calc_fitness()

    print(f"First generation | Best is '{genetic_algo.get_best().get_string()}' with a fitness of "
          f"{genetic_algo.get_best().fitness}")

    while genetic_algo.get_best().get_string() != FINAL_STRING:
        genetic_algo.calc_selection()
        genetic_algo.calc_crossover()
        genetic_algo.calc_mutation()
        genetic_algo.new_generation()
        genetic_algo.population.calc_fitness()

        print(f"--- New generation | Best is '{genetic_algo.get_best().get_string()}' with a fitness of "
              f"{genetic_algo.get_best().fitness}")

    et = module_time.time()
    print(f"Solution found | Best is '{genetic_algo.get_best().get_string()}' with a fitness of "
          f"{genetic_algo.get_best().fitness}")
    print(f"Number of generations: {genetic_algo.generation}")
    print(f"Execution time: {et - st} seconds")
