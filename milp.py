from copy import deepcopy
from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit
import matplotlib.pyplot as plt
import numpy as np

MAP_SIZE = 100
N_CUSTOMER = 8
CAPACITY = 1.0
LOAD_LIMIT = 2
RANDOM_SEED = 1107

np.random.seed(RANDOM_SEED)
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



# Create the mip solver with the SCIP backend.
solver = pywraplp.Solver.CreateSolver('SCIP')
if not solver:
    print("failed")
    quit()

infinity = solver.infinity()
# x and y are integer non-negative variables.
x_matrix = []
for i in range(N_CUSTOMER+1):
    x_matrix_j = []
    for j in range(N_CUSTOMER+1):
        x_matrix_j.append(solver.IntVar(0,1,'x(%d,%d)' % (i,j)))
    x_matrix.append(x_matrix_j)

f_matrix = []
for i in range(N_CUSTOMER+1):
    f_matrix_j = []
    for j in range(N_CUSTOMER+1):
        f_matrix_j.append(solver.Var(0.0,CAPACITY,False,'f(%d,%d)' % (i,j)))
    f_matrix.append(f_matrix_j)

f_matrix_prime = []
for i in range(N_CUSTOMER+1):
    f_matrix_prime_j = []
    for j in range(N_CUSTOMER+1):
        f_matrix_prime_j.append(solver.IntVar(0.0,LOAD_LIMIT,'f_prime(%d,%d)' % (i,j)))
    f_matrix_prime.append(f_matrix_prime_j)

const1 = []
for i in range(1,N_CUSTOMER+1):    
    const1.append(0)
    for j in range(N_CUSTOMER+1):
        const1[i-1] += x_matrix[i][j]        
    solver.Add(const1[i-1] == 1)
    

const2=[]
for i in range(1, N_CUSTOMER+1):    
    const2.append(0)
    for j in range(N_CUSTOMER+1):
        const2[i-1] += x_matrix[j][i]
    solver.Add(const2[i-1] == 1)

fconst=[]
for i in range(1, N_CUSTOMER+1):    
    fconst.append(0)
    for j in range(N_CUSTOMER+1):
        fconst[i-1] += f_matrix[j][i] - f_matrix[i][j]
    solver.Add(fconst[i-1] == package_weights[i-1])

    
for i in range(N_CUSTOMER+1):    
    for j in range(N_CUSTOMER+1):
         solver.Add(0 <= f_matrix[i][j] <= CAPACITY * x_matrix[i][j])

fconst_prime=[]
for i in range(1, N_CUSTOMER+1):    
    fconst_prime.append(0)
    for j in range(N_CUSTOMER+1):
        fconst_prime[i-1] += f_matrix_prime[j][i] - f_matrix_prime[i][j]
    solver.Add(fconst_prime[i-1] == 1)

    
for i in range(N_CUSTOMER+1):    
    for j in range(N_CUSTOMER+1):
         solver.Add(0 <= f_matrix_prime[i][j] <= LOAD_LIMIT *  x_matrix[i][j])

for i in range(N_CUSTOMER+1):  
    solver.Add(x_matrix[i][i]==0)


print('Number of variables =', solver.NumVariables())
print('Number of constraints =', solver.NumConstraints())

objective = 0
for i in range(N_CUSTOMER+1):    
    for j in range(N_CUSTOMER+1):
        objective += distance_matrix[i][j] * x_matrix[i][j] # * (1+f_matrix[i][j])        
       


solver.Minimize(objective)
print(solver.Objective())
# solver_parameters = pywraplp.MPSolverParameters()
# solver_parameters.SetDoubleParam(pywraplp.MPSolverParameters.PRIMAL_TOLERANCE, 0.001)
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    # print('i =', x_matrix.solution_value())
    # print('y =', j.solution_value())
else:
    print('The problem does not have an optimal solution.')
    print('Solution:')
    print('Objective value =', solver.Objective().Value())

print('\nAdvanced usage:')
print('Problem solved in %f milliseconds' % solver.wall_time())
print('Problem solved in %d iterations' % solver.iterations())
print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


# for i in range(N_CUSTOMER+1):    
#     for j in range(N_CUSTOMER+1):
#         print(x_matrix[i][j].solution_value(), end="\t")
#     print()
# print()
# for i in range(N_CUSTOMER+1):    
#     for j in range(N_CUSTOMER+1):
#         print("%.2f" % f_matrix[i][j].solution_value(), end="\t")
#     print()
   