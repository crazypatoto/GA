import time
import math
import colorsys
import numpy as np
import matplotlib.pyplot as plt

MAP_SIZE = 100
N_AGENTS = 3
N_CITIES = 10
DEPOT = np.zeros(2)

POPULATION = 10
# CROSS_RATE = 0.1
MUTATE_RATE = 0.5
N_GENERATIONS = 200

def populate(n_population):    
    population = []
    # population = np.array([]).reshape(0,N_CITIES+N_AGENTS)
    for _ in range(n_population):
        chromosome_1 = np.random.permutation(np.arange(N_CITIES))
        chromosome_2 = np.zeros(N_AGENTS)
        for _ in range(N_CITIES):
            agent = np.random.randint(low=0, high=N_AGENTS, size=1)
            chromosome_2[agent] += 1    

        chromosome = np.concatenate((chromosome_1, chromosome_2))        
        population.append(chromosome.astype(int))
        # population = np.vstack([population,chromosome])
    
    return population

def get_fitnesses(cities, chromosomes):    
    total_distances = np.zeros(POPULATION).tolist()
    fitnesses = np.zeros(POPULATION).tolist()
    for i in range(POPULATION):        
        offset = 0
        for j in range(N_CITIES, N_CITIES+N_AGENTS):
            city_count = chromosomes[i][j]
            if city_count == 0: 
                continue            
            total_distances[i] += np.linalg.norm(cities[chromosomes[i][offset]])
            for k in range(offset, offset + city_count - 1):
                city1_index = chromosomes[i][k]
                city2_index = chromosomes[i][k+1]
                total_distances[i] += np.linalg.norm(cities[city2_index] - cities[city1_index])                
            total_distances[i] += np.linalg.norm(cities[chromosomes[i][offset+city_count-1]])             
            offset += city_count
            fitnesses[i] = 1/(total_distances[i])
    return fitnesses, total_distances 

def roulette_wheel_selection(population, fitnesses):
    total = np.sum(fitnesses)
    selection_probs = fitnesses / total   
    return population[np.random.choice(len(population), p=selection_probs)]    

def tournament_selection(population, fitnesses, k=3, p=1):
    ix = np.arange(POPULATION)
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

# Part 1: Order Crossover (OX), Part 2: Asexual Crossover
def crossover(parent1, parent2):
    start_index = np.random.randint(0, N_CITIES)
    end_index =  np.random.randint(0, N_CITIES)
    while start_index == end_index:
        end_index = np.random.randint(0, N_CITIES)
    if start_index > end_index:
        tmp = start_index
        start_index = end_index
        end_index = tmp        
    
    offstring1 = np.zeros_like(parent1)    
    offstring1[start_index:end_index+1] = 1
    offstring1 = offstring1 * parent1
    parent2_remaining = np.copy(parent2[:N_CITIES]).tolist()
    for a in offstring1[start_index:end_index+1]:
        for b in parent2_remaining:
            if a == b:
                parent2_remaining.remove(b)    
    for i in range(0, N_CITIES):
        if i >= start_index and i <= end_index:
            continue
        offstring1[i] = parent2_remaining.pop(0)    


    offstring2 = np.zeros_like(parent2)    
    offstring2[start_index:end_index+1] = 1
    offstring2 = offstring2 * parent2
    parent1_remaining = np.copy(parent1[:N_CITIES]).tolist()
    for a in offstring2[start_index:end_index+1]:
        for b in parent1_remaining:
            if a == b:
                parent1_remaining.remove(b)    
    for i in range(0, N_CITIES):
        if i >= start_index and i <= end_index:
            continue
        offstring2[i] = parent1_remaining.pop(0) 

    asexual_index = np.random.randint(1, N_AGENTS)    
    
    temp1 = parent1[-N_AGENTS:] 
    temp1 = np.concatenate((temp1[asexual_index:], temp1[:asexual_index]))    
    offstring1[-N_AGENTS:] = temp1
    temp2 = parent2[-N_AGENTS:] 
    temp2 = np.concatenate((temp2[asexual_index:], temp2[:asexual_index]))    
    offstring2[-N_AGENTS:] = temp2
    
    return [offstring1, offstring2]

def mutation(chromosome):
    prob = np.random.random()
    if prob >= MUTATE_RATE:
        return chromosome

    index1 = np.random.randint(0, N_CITIES)
    index2 = np.random.randint(0, N_CITIES)
    while index1 == index2:
        index2 = np.random.randint(0, N_CITIES)
    
    chromosome[[index1,index2]] = chromosome[[index2,index1]]
    
    index1 = np.random.randint(-N_AGENTS, 0)
    index2 = np.random.randint(-N_AGENTS, 0)
    while index1 == index2:
        index2 = np.random.randint(-N_AGENTS, 0)

    chromosome[[index1,index2]] = chromosome[[index2,index1]]

    return chromosome

np.random.seed(0)

cities = list(np.random.randint(MAP_SIZE, size=(N_CITIES, 2)) - MAP_SIZE/2)
print(cities)
np.random.seed()
# plt.figure()
# plt.title('City Distribution')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.plot(cities[:,0],cities[:,1],'x') # Plot cities
# plt.plot(0, 0, 'rx') # Plot depot
# # plt.plot(cities[:,0],cities[:,1],'-')
# plt.xlim([-MAP_SIZE/2, MAP_SIZE/2])
# plt.ylim([-MAP_SIZE/2, MAP_SIZE/2])
# plt.show()


population = populate(POPULATION)

# fitnesses, distances = get_fitnesses(cities, population)
# print(population)
# print(fitnesses)
# print(tournament_selection(population,fitnesses))
colors = []
for i in range(N_AGENTS):
    hue = i / N_AGENTS
    (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    colors.append((r, g, b))

min_distances = []
plt.ion()
for g in range(N_GENERATIONS):    
    fitnesses, distances = get_fitnesses(cities, population) 
    print('Gen %d: max_fit = %.5f, min_distance = %.2f' % (g, np.max(fitnesses), np.min(distances)), end='\r')
    min_distances.append(np.min(distances))
    
    best_chrosome = population[np.argmax(fitnesses)]
    offset = 0
    for j in range(N_CITIES, N_CITIES+N_AGENTS):
        city_count = best_chrosome[j]        
        if city_count == 0: 
            continue                 
        route = np.zeros(2)   
        for k in range(offset, offset + city_count):
            city_index = best_chrosome[k]
            route = np.vstack((route, cities[city_index]))      
        route = np.vstack((route, np.zeros(2)))
        offset += city_count
        plt.plot(np.array(route)[:,0], np.array(route)[:,1], '-', color=colors[j-N_CITIES])
                
    # selected = [roulette_wheel_selection(population, fitnesses) for _ in range(POPULATION)] 
    selected = [tournament_selection(population, fitnesses) for _ in range(POPULATION)] 
    children = []
    for i in range(0, POPULATION-1, 2):
        # Get parent pair
        parent1, parent2 = selected[i], selected[i+1]
        # Crossover and mutation
        for c in crossover(parent1, parent2):   
            children.append(mutation(c))        
        population = children            
    del population[np.argmin(fitnesses)]
    population.append(best_chrosome)
   
    plt.plot(np.array(cities)[:,0], np.array(cities)[:,1], 'rx')
    plt.plot(0,0,'rx')        
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

plt.ioff()
offset = 0
for j in range(N_CITIES, N_CITIES+N_AGENTS):
    city_count = best_chrosome[j]        
    if city_count == 0: 
        continue                 
    route = np.zeros(2)   
    for k in range(offset, offset + city_count):
        city_index = best_chrosome[k]
        route = np.vstack((route, cities[city_index]))      
    route = np.vstack((route, np.zeros(2)))
    offset += city_count
    plt.plot(np.array(route)[:,0], np.array(route)[:,1], '-', color=colors[j-N_CITIES])

plt.plot(np.array(cities)[:,0], np.array(cities)[:,1], 'rx')
plt.plot(0,0,'rx')

plt.figure()
plt.plot(min_distances)

plt.show()