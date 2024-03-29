import random


# Might be useful in cross over operation
def fill_with_parent1_genes(child, parent, genes_n):
    start_at = random.randint(0, len(parent.genes) - genes_n - 1)
    finish_at = start_at + genes_n

    for i in range(start_at, finish_at):
        child.genes[i] = parent.genes[i]


def fill_with_parent2_genes(child, parent):
    j = 0

    for i in range(0, len(parent.genes)):
        if child.genes[i] is None:
            while parent.genes[j] in child.genes:
                j += 1

            child.genes[i] = parent.genes[j]
            j += 1


# Might be useful in mutation operation
def mutate(route_to_mut, k_mut_prob=0.2):
    """
    Route() --> Route()
    Swaps two random indexes in route_to_mut.route. Runs
    k_mut_prob*100 % of the time
    """
    # k_mut_prob %
    if random.random() < k_mut_prob:
        # two random indices:
        mut_pos1 = random.randint(0, len(route_to_mut.route) - 1)
        mut_pos2 = random.randint(0, len(route_to_mut.route) - 1)

        # if they're the same, skip to the chase
        if mut_pos1 == mut_pos2:
            return route_to_mut

        # Otherwise swap them:
        city1 = route_to_mut.route[mut_pos1]
        city2 = route_to_mut.route[mut_pos2]

        route_to_mut.route[mut_pos2] = city1
        route_to_mut.route[mut_pos1] = city2
