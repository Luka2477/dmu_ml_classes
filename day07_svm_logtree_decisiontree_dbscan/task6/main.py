import random as module_random
import string as module_string
import time as module_time

# Constants
FINAL_STRING = "To be or not to be that is the question"

# Types
FitnessType = tuple[str, list[int], int]
CharsType = dict[str, int]


def get_chars(string: str) -> CharsType:
    chars = CharsType()

    for char in string:
        chars[char] = (chars.get(char) or 0) + 1

    return chars


def random_string(string: str = None) -> str:
    if not string:
        string = "".join("#" for _ in range(len(FINAL_STRING)))

    letters = module_string.ascii_letters + " "
    for i in range(len(string)):
        if string[i] == "#":
            string = string[:i] + module_random.choice(letters) + string[i + 1:]

    return string


def reset_string(fit: FitnessType) -> str:
    string = ""

    for i in range(len(fit[0])):
        string += (fit[1][i] == 2 and fit[0][i]) or "#"

    return string


def fitness(string: str, chars: CharsType) -> FitnessType:
    f = list[int]()

    for i in range(len(string)):
        cs = string[i]
        ds = FINAL_STRING[i]
        lf = (cs == ds and 2) or 0

        if chars.get(cs) not in [None, 0]:
            lf = (lf == 2 and 2) or 1
            chars[cs] -= 1

        f.append(lf)

    return string, f, sum(f)


def best_fitness(fitnesses: list[FitnessType]) -> FitnessType:
    maxf = ("", list[int](), 0)

    for f in fitnesses:
        if f[2] > maxf[2]:
            maxf = f

    return maxf


if __name__ == '__main__':
    # Task b - Random
    iterations = 10_000
    print(f"Random - Iterations: {iterations}")
    st = module_time.time()

    curr_fitnesses = [fitness(random_string(), get_chars(FINAL_STRING)) for _ in range(iterations)]
    best = best_fitness(curr_fitnesses)

    et = module_time.time()
    print(f"Best was: {best}")
    print(f"Execution time: {et - st} seconds")
    print()
    print()

    # Task c - Hill Climb
    neighbor_count = 75
    verbose = True
    retries = 5
    curr_retries = retries
    count = 1
    curr_fitness = fitness(random_string(), get_chars(FINAL_STRING))
    print(f"Hill Climb - Neighbors per iteration: {neighbor_count} | Verbose: {verbose} | Retries: {retries}")
    print()
    st = module_time.time()

    while curr_fitness[0] != FINAL_STRING:
        neighbors = [fitness(random_string(reset_string(curr_fitness)), get_chars(FINAL_STRING)) for _ in
                     range(neighbor_count)]
        best = best_fitness(neighbors)
        count += 1

        if best[2] > curr_fitness[2]:
            curr_fitness = best
            curr_retries = retries

            if verbose:
                print(f"New best: {best}")
        elif curr_retries != 0:
            curr_retries -= 1

            if verbose:
                print(f"Retry number {retries - curr_retries}: {curr_fitness}")
        else:
            print(f"No new state in {retries} retries - {curr_fitness}")
            break

    et = module_time.time()
    print()
    print(f"Count: {count} - {curr_fitness}")
    print(f"Execution time: {et - st} seconds")
    print()
    print()
