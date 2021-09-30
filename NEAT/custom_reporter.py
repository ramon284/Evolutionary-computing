"""
Gathers (via the reporting interface) and provides (to callers and/or a file)
the most-fit genomes and information on genome/species fitness and species sizes.
"""
import copy
import csv
import os

from neat.math_util import mean, stdev
from neat.reporting import BaseReporter
from neat.six_util import itervalues


# TODO: Make a version of this reporter that doesn't continually increase memory usage.
# (Maybe periodically write blocks of history to disk, or log stats in a database?)

class EvomanReporter(BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genome/species fitness and species sizes.
    """
    def __init__(self, enemy, run):
        BaseReporter.__init__(self)
        self.gen = 0
        self.filename = "enemy_" + str(enemy) + "_run_" + str(run) + ".txt"
        self.results_dir = os.path.join(os.getcwd(), "NEAT_results")

        #create results directory if it does not exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        #create full file name
        self.filename = os.path.join(self.results_dir, self.filename)

        # write header
        with open(self.filename, 'w') as f:
            w = csv.writer(f, delimiter=' ')
            w.writerow(["gen", "best", "mean", "std"])

    def start_generation(self, generation):
        self.gen = generation

    def post_evaluate(self, config, population, species, best_genome):
            gen = self.gen
            fitnesses = [c.fitness for c in itervalues(population)]
            fit_mean = mean(fitnesses)
            fit_std = stdev(fitnesses)
            fit_best = best_genome.fitness
            self.save_stats(gen, fit_best, fit_mean, fit_std)


    def save_stats(self, gen, best, mean, std):
        with open(self.filename, 'a') as f:
            w = csv.writer(f, delimiter=' ')
            w.writerow([gen, best, mean, std])
