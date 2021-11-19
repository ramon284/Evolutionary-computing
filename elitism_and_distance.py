# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
from math import ceil, floor
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import argparse

parser = argparse.ArgumentParser(description="set run_no, exp_name and enemy_type")
parser.add_argument("--run_no",dest="run_no",default="1")
parser.add_argument("--exp_name", dest="exp_name", default="elitism")
parser.add_argument("--enemy_type", dest="enemy_type",default="2")
parser.add_argument("--run_mode", dest="run_mode",default="train")
args = parser.parse_args()

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = args.exp_name+'_'+args.enemy_type+'_'+args.run_no
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
if(args.run_mode == "test"):
    env = Environment(experiment_name=experiment_name,
                  enemies=[1,2,3,4,5,6,7,8],#,int(args.enemy_type[2])],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini="yes",
                  multiplemode="yes")

else:
# initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2,5,6],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    randomini="yes",
                    multiplemode="yes")


env.state_to_log() # checks environment state
ini = time.time()  # sets time marker

run_mode = str(args.run_mode) # train or test

# number of weights for multilayer with 10 hidden neurons
nweights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

limits = [-1,1]
npop = 100
gens = 10

mutationChance = [0.2, 0.1] ## chance of mutation per child, and per genome 
mutation = 0.45 ## dictates how much a genome can be mutated in percentage
mutationT = -0.02 ## decrease/increase mutation over time

elitism_size = 0.20 ## percentage of surviving "best parents"
elitism_sizeT = 0.02 ## decrease/increase elitism size over time
distance_size = 2 ## how many of the most genetically distant parents we choose.

distanceMethod = "genotype" ## "genotype" to select for distant genomes, "phenotype" to select for distant (worst) fitness. leave empty to ignore.

crossoverMethod = "average" ## "even" takes 50% of genomes of both parents, and "uniform" flips a coin for every genome


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

def checkGeneticDistance(parent, pop): ## check who is the most genetically distant
    distanceList = []
    for individual in pop: ## compare parent to each individual here
        distance = 0
        for x, y in zip(parent, individual):
            distance += abs(x-y) ## calculate absolute distance between all genomes combined
        distanceList.append(distance)
    print('smallest: ', min(distanceList), 'biggest: ',max(distanceList))
    return distanceList.index(max(distanceList)) ## I.E. [1,1,1,-1,1] genetically most distant -> [-1,-1,-1,1,-1]

# select parents
def parentSelect(pop):
    fitness = evaluate(pop) ## we need to order population from best to worst fitness
    fitness_sorted = np.argsort(-fitness) ## use negation to flip the ordering from ascending to descending
    fitness = fitness[fitness_sorted]
    pop = pop[fitness_sorted]
    parents = pop[:int(npop*elitism_size)] ## select certain number of best parents
    fitnessParents = fitness[:int(npop*elitism_size)] 
    if(distanceMethod == "genotype"):
        nonparents = pop[int(npop*elitism_size):] ## for this to work we need the population without the selected parents
        nonfitness = fitness[int(npop*elitism_size):] ## fitness of non-selected parents
        for i in range(distance_size):
            x = checkGeneticDistance(parents[i], nonparents)
            parents = np.append(parents, np.array([nonparents[x]]), 0)
            nonparents = np.delete(nonparents, x, 0)
            fitness = np.append(fitness, np.array([nonfitness[x]]), 0)
            nonfitness = np.delete(nonfitness, x , 0)
    elif(distanceMethod == "phenotype"): ## add X number of parents with the worst fitness for diversity
        worstParents = pop[int(npop-distance_size):]
        worstFitness = fitness[int(npop-distance_size):]
        parents = np.append(parents, worstParents, 0)
        fitnessParents = np.append(fitnessParents, worstFitness, 0)
    return parents, fitnessParents ## return the parents and their fitness scores seperately


## use best performing parents for crossover
def crossover(parents):
    nchildren = int(npop - (npop*elitism_size)-distance_size) ## how many children to make
    print('hoeveelheid parents:', parents.shape, 'hoeveelheid children:', nchildren)
    ngenomes = len(parents[0]) ## number of genomes
    offspring = np.empty([nchildren, ngenomes]) ## empty offspring matrix
    if (crossoverMethod  == "uniform"):
        for i in range (nchildren):
            first, second = np.random.choice(len(parents), 2, replace=False)
            for j in range(ngenomes):
                if(np.random.uniform(0, 1) < 0.5): 
                    offspring[i][j] = parents[first][j]  
                else: ## take 1 genome from either parent 1 or 2 at random, for all genomes
                    offspring[i][j] = parents[second][j]
    if (crossoverMethod == "even"):
        for i in range(nchildren):
            first, second = np.random.choice(len(parents), 2, replace=False)
            cutoff = (int(floor(nweights/2)))
            offspring[i][:cutoff] = parents[first][:cutoff]
            offspring[i][cutoff:] = parents[second][cutoff:]
    if (crossoverMethod == "average"): ## you can try to add a different crossover method here
        for i in range (nchildren):
            first, second = np.random.choice(len(parents), 2, replace=False)
            for j in range(ngenomes):
                offspring[i][i] = (parents[first][j] + parents[second][j]) / 2
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
for i in range(1,gens): ## evolutional loop
    print('we are in loop number ', i, ' now baby!')
    parents, fit_parents = parentSelect(pop) ## select the best parents
    offspring = crossover(parents)           ## makes offspring, also mutates them.
    fit_offspring = evaluate(offspring)
    pop = np.concatenate((parents, offspring)) ## create the new generation by adding survivors + offspring
    fit_pop = np.concatenate((fit_parents, fit_offspring)) 
    best = np.argmax(fit_pop) 
    best_sol = fit_pop[best]
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues

    mutation += mutationT ## decrease/increase mutation severity over time
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
