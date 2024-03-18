import click
import random

from deap import base
from deap import creator
from deap import tools

import math
import os
import json


from deap import algorithms
import numpy as np


IND_DIM = 20
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12


def setup_algo():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     # fitness for minimization - notice the negative weight
    creator.create("Individual", list, fitness=creator.FitnessMin)  # create class for individuals - derived from list and uses the fitness defined above

    toolbox = base.Toolbox()
    toolbox.register("attr_val", random.uniform, LOWER_BOUND, UPPER_BOUND)             # generates a single random number between 0 and 1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_val, IND_DIM)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=2)


    toolbox.register("evaluate", rastrigin)
    
    return toolbox


def rastrigin(ind):
    return 10*len(ind) + sum([(x**2 - 10 * math.cos(2 * math.pi * x)) for x in ind]),


def setup_stats():
    s = tools.Statistics(lambda x: x.fitness.values[0])
    s.register("mean", np.mean)
    s.register("max", max)
    s.register("min", min)

    hof = tools.HallOfFame(1)  # best individual
    
    return s, hof


@click.command()
@click.argument("pop_size", default=100)
@click.option("--cxpb", defaul=0.8)
@click.option("--verbose/--no_verbose", default=False)
@click.option("--out_path", default=None)
def main(pop_size, cxpb, verbose, out_path)
    settings = {'pop_size': pop_size, 'cxpb': cxpb}

    toolbox = setup_algo()
    s, hof = setup_stats()
    
    pop = toolbox.population(pop_size)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=100, cxpb=cxpb, mutpb=0.2, ngen=500, stats=s, halloffame=hof, 
                                         verbose=verbose)

    print(hof, hof[0].fitness.values[0])
    
    if out_path is not None:
        # You can check this, sometimes you want to warn instead
        # You can also ask via input() whether you want to overwrite it (not good if running batch jobs)
        assert not os.path.exists(out_path), "Output path exists!"
        with open(out_path, 'w') as f:
            json.dump({**settings, 'result': hof[0].fitness.values[0]}, f)

    
if __name__ == "__main__":
    main()
