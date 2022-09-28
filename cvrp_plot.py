from copy import deepcopy
import math
import random
from struct import pack
from this import d
import numpy as np
import matplotlib.pyplot as plt
import time

MAP_SIZE = 1000
N_CUSTOMER = 50
N_GENERATION = 100000
N_POPULATION = 30
CAPACITY = 1.0
LOAD_LIMIT = 3
MUTATE_RATE = 0.6
PATIENCE = 10000

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"

def initialize_population():
    population = []

    while len(population) < N_POPULATION:
        chromosome = np.random.permutation(np.arange(N_CUSTOMER)+1).astype(int)
        if not any((chromosome == c).all() for c in population):
            population.append(chromosome)

    return population

def get_fitness_min_distance(chromosome, distance_matrix, package_weights):
    current_load = 0
    n_vehicle = 1
    route = [0]
    n_load = 0
    for customer_id in chromosome:
        current_load += package_weights[customer_id-1]        
        n_load += 1
        if current_load > CAPACITY or n_load >= LOAD_LIMIT:
            route.append(0)
            n_vehicle += 1            
            n_load = 0
            current_load = package_weights[customer_id-1]
            route.append(customer_id)
        else:        
            route.append(customer_id)            
    route.append(0)
    
    total_distance = 0
    for i in range(len(route)-1):
        total_distance += distance_matrix[route[i]][route[i+1]]

    # subroutes = get_subroutes(chromosome, package_weights)
    # distances = []
    # for route in subroutes:
    #     distance = 0
    #     for i in range(len(route)-1):
    #         distance += distance_matrix[route[i]][route[i+1]]
    #     distances.append(distance)

    # fitness = 1 / (np.max(distances) + total_distance  + n_vehicle)
    fitness = 1 / (total_distance + n_vehicle)
    
    return fitness, total_distance

def get_fitness_min_energy(chromosome, distance_matrix, package_weights): # https://www.sciencedirect.com/science/article/pii/S2772390922000142#f0005
    total_energy = 0
    total_distance = 0
    subroutes = get_subroutes(chromosome, package_weights)
    n_vehicle = len(subroutes)

    for route in subroutes:
        customers = np.array(route)
        customers = customers[customers > 0]
        current_load = np.sum(np.array(package_weights)[customers-1])
        for i in range(len(route)-1):         
            total_energy += distance_matrix[route[i]][route[i+1]] * (1 + current_load)
            total_distance += distance_matrix[route[i]][route[i+1]]
            current_load -= package_weights[route[i+1]-1]        
   
    fitness = 1 / (total_energy)
    
    return fitness, total_energy

def get_subroutes(chromosome, package_weights):
    current_load = 0
    n_vehicle = 1
    route = [0]        
    subroutes = []
    n_load = 0
    for customer_id in chromosome:
        current_load += package_weights[customer_id-1]
        n_load += 1
        if current_load > CAPACITY or n_load >= LOAD_LIMIT:
            route.append(0)
            n_vehicle += 1
            n_load = 0
            subroutes.append(route[:])
            route = [0]
            current_load = package_weights[customer_id-1]
            route.append(customer_id)
        else:        
            route.append(customer_id)
    route.append(0)
    subroutes.append(route[:])

    return subroutes

def tournament_selection(population, fitnesses, k=3, p=1):
    ix = np.arange(len(population))
    np.random.shuffle(ix)    
    selected_indices = ix[:k]    
    chromosomes = np.array(population)[selected_indices]
    selecetd_fitnesses = np.array(fitnesses)[selected_indices]
    arr1inds = selecetd_fitnesses.argsort()
    chromosomes = chromosomes[arr1inds[::-1]]
    
    selection_probs = []
    selection_probs.append(p)
    for i in range(1, k):
        selection_probs.append(p*math.pow(1-p,i))

    return chromosomes[np.random.choice(len(chromosomes), p=selection_probs)]

def crossover(parent1, parent2):
    start_index = np.random.randint(0, N_CUSTOMER)
    end_index =  np.random.randint(0, N_CUSTOMER)
    if start_index > end_index:
        start_index, end_index = end_index, start_index
    
    offspring1 = np.zeros_like(parent1)
    offspring1[start_index:end_index+1] = 1
    offspring1 = offspring1 * parent1
    parent2_remaining = deepcopy(parent2).tolist()
    for a in offspring1[start_index:end_index+1]:
        for b in parent2_remaining:
            if a == b:
                parent2_remaining.remove(b)    
    for i in range(0, N_CUSTOMER):
        if i >= start_index and i <= end_index:
            continue
        offspring1[i] = parent2_remaining.pop(0)

    offspring2 = np.zeros_like(parent2)
    offspring2[start_index:end_index+1] = 1
    offspring2 = offspring2 * parent2
    parent1_remaining = deepcopy(parent1).tolist()
    for a in offspring2[start_index:end_index+1]:
        for b in parent1_remaining:
            if a == b:
                parent1_remaining.remove(b)    
    for i in range(0, N_CUSTOMER):
        if i >= start_index and i <= end_index:
            continue
        offspring2[i] = parent1_remaining.pop(0)

    return offspring1, offspring2

def mutate1(chromosome, p=MUTATE_RATE):
    prob = np.random.random()
    if prob >= p:
        return chromosome

    index1 = np.random.randint(0, N_CUSTOMER)
    index2 = np.random.randint(0, N_CUSTOMER)
    while index1 == index2:
        index2 = np.random.randint(0, N_CUSTOMER)
    
    chromosome[[index1,index2]] = chromosome[[index2,index1]]  

    return chromosome

def mutate2(chromosome, p=MUTATE_RATE):
    if np.random.random() < p:
        index1, index2 = np.random.randint(0, N_CUSTOMER, 2)
        chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
        index1, index2 = sorted(random.sample(range(len(chromosome)), 2))
        mutated = np.concatenate((chromosome[:index1], np.flip(chromosome[index1:index2+1])))        
        if index2 < len(chromosome) - 1:
            mutated = np.concatenate((mutated, chromosome[index2+1:]))            
        return mutated
    return chromosome



def main():
    # np.random.seed(1107)
    customers = list(np.random.randint(MAP_SIZE, size=(N_CUSTOMER + 1, 2)) - MAP_SIZE/2) # Customer 0 as depot
    customers[0] = np.zeros(2)
    # package_weights = list(np.random.randint(1,CAPACITY*100,size=N_CUSTOMER)/100)
    candidates = np.random.normal(size=N_CUSTOMER*100) + 0.5     
    candidates -= np.min(candidates)
    candidates /= np.max(candidates)        
    np.delete(candidates, np.argmax(candidates))
    np.delete(candidates, np.argmin(candidates))
    candidates *= 0.6  
    candidates = (candidates * 100).astype(int) / 100    
    np.random.shuffle(candidates)
    package_weights = list(candidates[:N_CUSTOMER])
    # package_weights = np.random.normal(0.5,0.05,size=N_CUSTOMER)
    package_weights = np.random.random(size=N_CUSTOMER)*0.4 + 0.2
    package_weights = (package_weights * 100).astype(int) / 100   
        
    distance_matrix = np.zeros((N_CUSTOMER+1, N_CUSTOMER+1))
    for i in range(N_CUSTOMER+1):
        for j in range(N_CUSTOMER+1):
            if i == j:
                continue
            distance_matrix[i][j] = np.linalg.norm(customers[i] - customers[j])        

    population = [10, 30, 50]
    datas = []
    for i in range(3):

        t1 = time.time()
        np.random.seed()
        plt.ion()
        patience = 0
        min_distances = []
        max_fitnesses = []
        population = initialize_population() 

        # get_fitness = get_fitness_min_distance
        get_fitness = get_fitness_min_energy
        fitnesses, distances = zip(*[get_fitness(chromosome, distance_matrix, package_weights) for chromosome in population]) 
        elite_chromosome, elite_fitness, elite_distance = population[np.argmax(fitnesses)], np.max(fitnesses), np.min(distances)        
        for g in range(N_GENERATION):                
            plt.clf()    
            selected = [tournament_selection(population, fitnesses) for _ in range(len(population))]  
            
            children = []
            for i in range(0, len(population)-1, 2):
                parent1, parent2 = selected[i], selected[i+1]
                for offspring in crossover(parent1, parent2):
                    children.append(mutate2(offspring))
            population = children
            
            fitnesses, distances = zip(*[get_fitness(chromosome, distance_matrix, package_weights) for chromosome in population])        
            best_chromosome, best_fitness, best_distance = population[np.argmax(fitnesses)], np.max(fitnesses), np.min(distances)        
            patience += 1

            if best_fitness <= elite_fitness:
                del population[np.argmin(fitnesses)]
                population.append(elite_chromosome)
            else:
                elite_chromosome = best_chromosome
                elite_fitness = best_fitness
                elite_distance = best_distance
                patience = 0
                print("asdd")
            
            print('Gen %d: max_fit = %.5f, min_distance = %.2f' % (g, elite_fitness, elite_distance))        
            min_distances.append(elite_distance)
            max_fitnesses.append(elite_fitness)

            if patience > PATIENCE:
                break
                                                    
        # print(time.time()-t1)
        # plt.ioff()    
        # sub_routes = get_subroutes(elite_chromosome, package_weights)
        # print(len(sub_routes))    
        # for route in sub_routes:         
        #     points = [customers[index] for index in route]
        #     p = plt.plot(np.array(points)[:,0], np.array(points)[:,1], '-')
        #     color = p[0].get_color()

        #     xs, ys = np.array(points)[:,0],np.array(points)[:,1]
        #     for i in range(len(xs)-1):
        #         d = np.linalg.norm(np.array([xs[i+1],ys[i+1]])-np.array([xs[i],ys[i]]))
        #         plt.arrow((xs[i+1]+xs[i])/2, (ys[i+1]+ys[i])/2, (xs[i+1]-xs[i])/d, (ys[i+1]-ys[i])/d, shape='full', lw=0, length_includes_head=True, head_width=MAP_SIZE/50, color=color)            
        # plt.plot(np.array(customers)[1:,0], np.array(customers)[1:,1], 'o', color='blue')
        # plt.plot(np.array(customers)[0,0], np.array(customers)[0,1], 'ro')
        
        # plt.title('Routing Result')
        # plt.xlabel(r'$\mathit{x}\textrm{-coordinate}\ (\mathrm{m})$', fontsize=14)
        # plt.ylabel(r'$\mathit{y}\textrm{-coordinate}\ (\mathrm{m})$', fontsize=14)
        # plt.xlim([-MAP_SIZE/2, MAP_SIZE/2])
        # plt.ylim([-MAP_SIZE/2, MAP_SIZE/2])

        # for i in range(1, len(customers)):
        #     x, y = customers[i][0], customers[i][1]
        #     plt.text(x, y, str(package_weights[i-1]), color='black', fontsize=10)
        # plt.figure()      

        datas.append(min_distances)
    
    # plt.plot(min_distances)
    plt.ioff()    
    plt.figure()      
    plt.plot(datas[0],label='popultation = 10')
    plt.plot(datas[1],label='popultation = 30')
    plt.plot(datas[2],label='popultation = 50')
    plt.title('Minimum energy cost evolution')
    plt.xlabel('Generation number', fontsize=14)
    plt.ylabel('Minimum energy cost', fontsize=14)

    plt.legend(fontsize=12)
    plt.show()

if __name__ == '__main__':
    main()