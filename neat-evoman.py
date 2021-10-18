"""
evoman algorithm using neat. Based on example given here:
https://neat-python.readthedocs.io/en/latest/xor_example.html
"""

# import neat

from __future__ import print_function
import os
import neat
import NEAT.visualize as visualize
import pygame


# imports framework
import sys
sys.path.insert(0, 'evoman')

from NEAT.player_controller import PlayerController
from NEAT.custom_reporter import EvomanReporter

from environment import Environment

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob

import argparse

parser = argparse.ArgumentParser(description="set exp_name and enemy_type")
parser.add_argument("--exp_name", dest="exp_name", default="neat")
parser.add_argument("--enemy_type", dest="enemy_type",default="2")
parser.add_argument("--run_no", dest="run_no", default="1")
args = parser.parse_args()


# parameters:
n_generations = 20
headless = True
should_visualize = False

# choose this for not using visuals and thus making experiments faster
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = args.exp_name+'_'+args.enemy_type+'_'+args.run_no
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.


env = Environment(experiment_name=experiment_name,
                  enemies=[int(args.enemy_type[0]),int(args.enemy_type[1]),int(args.enemy_type[2])],
                  randomini = "yes",
                  playermode="ai",
                  player_controller=PlayerController(),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  multiplemode="yes")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

def simulation(pcont):
    f, p, e, t = env.play(pcont=pcont)
    return p-e

def get_mean_individual_gain(winner):
    igs = []
    env2 = Environment(experiment_name=experiment_name,
                  enemies=[1,2,3,4,5,6,7,8],#,int(args.enemy_type[2])],
                  randomini = "yes",
                  playermode="ai",
                  player_controller=PlayerController(),
                  enemymode="static",
                  level=2,
                  speed="normal",
                  multiplemode="yes")
    for _ in range(5):
        f,p,e,t = env2.play(pcont=winner)
        igs.append(p - e)
    return np.mean(igs)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        pcont = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = simulation(pcont)

def write_genome_weights(genome):
    folder_name = os.path.join(os.getcwd(), "NEAT_results")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #create full file name
    filename = os.path.join(folder_name, str(args.enemy_type+'_'+args.run_no+"_best.txt"))
    nodes = list(genome.nodes.values())
    nodes.sort()
    connections = list(genome.connections.values())
    connections.sort()
    # write header
    with open(filename, 'w') as f:
        #first write the biases for the hidden neurons
        for node in nodes[5:]:
            f.write(str(getattr(node, "bias")) + "\n")
        #write the connections between the input layer and the hidden layer
        for connection in connections[:200]:
            f.write(str(getattr(connection, "weight")) + "\n")
        #write the biases for the output neurons
        for node in nodes[:5]:
            f.write(str(getattr(node, "bias")) + "\n")
        #write the connections between the hidden layer and the output layer
        for connection in connections[200:]:
            f.write(str(getattr(connection, "weight")) + "\n")
        
def run(config_file, n_run= 0):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(EvomanReporter(env.enemies, n_run))
    # p.add_reporter(neat.Checkpointer(5))

    # Determine number of generations
    winner = p.run(eval_genomes, n_generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'} #example of node names from xor problem

    print("Took", time.time() - ini, "seconds in total.")

    if should_visualize:
        # try implementing the visualisation of the winning network
        visualize.draw_net(config, winner, True, node_names=None)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
    ## runs can be restored from checkpoints as follows:
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 1)
    write_genome_weights(winner)
    wcont = neat.nn.FeedForwardNetwork.create(winner, config)
    return get_mean_individual_gain(wcont)
#
#
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    print("local dir:", local_dir)
    config_path = os.path.join(local_dir, 'NEAT/config_neat')
    migs = []
    #for i in range(10):
    #    print("-"*60)
    #    print("run", i)
    migs.append(run(config_path, 0))
    print(migs)
    folder_name = os.path.join(os.getcwd(), "NEAT_results")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #create full file name
    filename = os.path.join(folder_name, str(args.enemy_type+'_'+args.run_no+"_mean.txt"))
    with open(filename, 'w') as f:
        f.write(str(migs))
