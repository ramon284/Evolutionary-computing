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


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'elitism_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutationChance = [0.2, 0.1] ## chance of mutation per child, and per genome 
mutation = 0.1 ## dictates how much a genome can be mutated in percentage
last_best = 0
elitism_size = 0.4 ## percentage of surviving "best parents"


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x): 
    return np.array(list(map(lambda y: simulation(env,y), x)))

# select parents
def parentSelect(pop):
    fitness = evaluate(pop) ## we need to order population from best to worst fitness
    #pop = [x for _, x in sorted(zip(fitness, pop), reverse=True)] ## <-- couldn't get this to work
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
                    #child[i] = child[i]+np.random.normal(0,1)
                    child[i] *= (1 + np.random.uniform(-mutation, mutation)) ## apply mutation
    return offspring


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop,fit_pop):
    pass  ## might implement later, this messes with population size I think.


# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens): ## evolutional loop
    print('we are in loop number ', i, ' now baby!')
    parents, fit_parents = parentSelect(pop) ## select the best parents
    offspring = crossover(parents)           ## makes offspring, also mutates them.
    fit_offspring = evaluate(offspring)
    pop = np.concatenate((parents, offspring)) ## create the new generation by adding survivors + offspring
    fit_pop = np.concatenate((fit_parents, fit_offspring)) 
    best = np.argmax(fit_pop) 
    best_sol = fit_pop[best]
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues

    # searching new areas

    # if best_sol <= last_sol:
    #     notimproved += 1
    # else:
    #     last_sol = best_sol
    #     notimproved = 0

    # if notimproved >= 15:
    #     file_aux  = open(experiment_name+'/results.txt','a')
    #     file_aux.write('\ndoomsday')
    #     file_aux.close()

    #     pop, fit_pop = doomsday(pop,fit_pop)
    #     notimproved = 0

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
