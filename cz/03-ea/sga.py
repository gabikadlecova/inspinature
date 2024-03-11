import random

POP_SIZE = 50
IND_LEN = 100
CX_PROB = 0.8
MUT_PROB = 0.2
MUT_PPB = 1/IND_LEN

def random_individual():
    return [random.randint(0, 1) for _ in range(IND_LEN)]

def random_initial_population():
    return [random_individual() for _ in range(POP_SIZE)]

def fitness(ind):
    return sum(ind)

def select(pop, fit):
    return random.choices(pop, fit, k=POP_SIZE)

def cross(p1, p2):
    point = random.randrange(0, len(p1))
    if random.random() < CX_PROB:
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1[:], p2[:]

def crossover(pop):
    o = []
    for p1, p2 in zip(pop[::2], pop[1::2]):
        o1, o2 = cross(p1, p2)
        o.append(o1)
        o.append(o2)
    return o

def mutate(p):
    if random.random() < MUT_PROB:
        return [1 - v if random.random() < MUT_PPB else v for v in p]
    return p[:]

def mutation(pop):
    return [mutate(p) for p in pop]

def evolution(pop):
    log = []
    for G in range(1000):
        fit = [fitness(ind)[0] for ind in pop]
        obj = [fitness(ind)[1] for ind in pop]
        log.append(min(obj))
        m = select(pop, fit)
        o = crossover(m)
        o = mutation(o)
        pop = o[:]

    return pop, log

input_set = [random.randrange(0,50) + 100 for _ in range(100)] # 100 random numbers between 100 and 150
number = 3*sum(input_set)//5

def fitness(ind):
    subset_sum = sum([ind_i*is_i for ind_i, is_i in zip(ind, input_set)])
    return 1/(1 + abs(subset_sum - number)), abs(subset_sum - number)

import pprint
# print(random_individual())
pop = random_initial_population()
# pprint.pprint([(fitness(ind), ind) for ind in pop])
pop, log = evolution(pop)
# pprint.pprint([(fitness(ind), ind) for ind in pop])
# print(number)
# print(input_set)
#pprint.pprint(log)
import matplotlib.pyplot as plt
plt.plot(log)
plt.show()