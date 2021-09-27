# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import argparse

parser = argparse.ArgumentParser(description="set run_no, exp_name and enemy_type")
parser.add_argument("--run_no",dest="run_no",default="1")
parser.add_argument("--exp_name", dest="exp_name", default="elitism_demo_2")
parser.add_argument("--enemy_type", dest="enemy_type",default="2")
args = parser.parse_args()

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = args.exp_name+'_'+args.run_no
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[int(args.enemy_type)],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")


env.state_to_log() # checks environment state
ini = time.time()  # sets time marker


run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
nweights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


limits = [-1,1]
npop = 100
gens = 15
mutationChance = [0.2, 0.1] ## chance of mutation per child, and per genome 
mutation = 0.45 ## dictates how much a genome can be mutated in percentage
mutationT = -0.02 ## decrease/increase mutation over time
last_best = 0
elitism_size = 0.4 ## percentage of surviving "best parents"
elitism_sizeT = -0.01 ## decrease/increase elitism size over time


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x): 
    return np.array(list(map(lambda y: simulation(env,y), x)))

def checkLimits(x):
    if x>limits[1]:
        return limits[1]
    elif x<limits[0]:
        return limits[0]
    else:
        return x

# select parents
def parentSelect(pop):
    fitness = evaluate(pop) ## we need to order population from best to worst fitness
    fitness_sorted = np.argsort(-fitness) ## use negation to flip the ordering from ascending to descending
    fitness = fitness[fitness_sorted]
    pop = pop[fitness_sorted]
    parents = pop[:int(npop*elitism_size)] ## select certain number of best parents
    fitness = fitness[:int(npop*elitism_size)] 
    return parents, fitness ## return the parents and their fitness scores seperately


## use best performing parents for crossover
def crossover(parents):
    nchildren = int(npop - (npop*elitism_size)) ## how many children to make
    ngenomes = len(parents[0]) ## number of genomes
    offspring = np.empty([nchildren, ngenomes]) ## empty offspring matrix
    for i in range (nchildren):
        first, second = np.random.choice(len(parents), 2, replace=False)
        for j in range(ngenomes):
            if(np.random.uniform(0, 1) < 0.5): 
                offspring[i][j] = parents[first][j]  
            else: ## take 1 genome from either parent 1 or 2 at random, for all genomes
                offspring[i][j] = parents[second][j]
    mutate(offspring)
    return offspring

## mutate the offspring
def mutate(offspring):
    for child in offspring:
        if(np.random.uniform(0, 1) < mutationChance[0]):
            for i in range(len(child)): ## iterate through all the genomes
                if(np.random.uniform(0, 1) < mutationChance[1]):
                    child[i] *= (1 + np.random.uniform(-mutation, mutation)) ## apply mutation
                    child[i] = checkLimits(child[i])
    return offspring


def testBest(): ## tests the best solution
    bestSolution = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bestSolution])

    sys.exit(0)

def firstGeneration(): ## creates the first generation
    print( '\nNEW EVOLUTION\n')
    pop = np.random.uniform(limits[0], limits[1], (npop, nweights))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()
    return pop


# evolution
if run_mode =='test':
    testBest() ## will sys.exit after testing

pop = firstGeneration()
for i in range(gens): ## evolutional loop
    print('we are in loop number ', i, ' now baby!')
    parents, fit_parents = parentSelect(pop) ## select the best parents
    offspring = crossover(parents)           ## makes offspring, also mutates them.
    fit_offspring = evaluate(offspring)
    pop = np.concatenate((parents, offspring)) ## create the new generation by adding survivors + offspring
    fit_pop = np.concatenate((fit_parents, fit_offspring)) 
    best = np.argmax(fit_pop) 
    best_sol = fit_pop[best]
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues

    mutation += mutationT ## decrease mutation severity over time
    elitism_size += elitism_sizeT 
    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' , best:'+str(round(fit_pop[best],6))+' , mean:'+str(round(mean,6))+' m, std:'+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()


fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state
