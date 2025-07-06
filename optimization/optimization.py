# optimization.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import itertools
from itertools import combinations

import pacmanPlot
import graphicsUtils
import util
from util import PriorityQueue

# You may add any helper functions you would like here:
# def somethingUseful():
#     return True



def findIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b)
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.
    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).
        If none of the constraint boundaries intersect with each other, return [].

    An intersection point is an N-dimensional point that satisfies the
    strict equality of N of the input constraints.
    This method must return the intersection points for all possible
    combinations of N constraints.

    """
    "*** YOUR CODE HERE ***"
    if not constraints:
        return []
    totalConstraints= len(constraints)
    variablesCount = len(constraints[0][0])

    # Preparing to identify all possible intersection points
    potentialIntersections = []

    # Constructing a list of binary strings to represent combinations
    binaryCombos = []
    for num in range(2**variablesCount - 1, 2**totalConstraints):
        binaryRep = format(num, f"0{totalConstraints}b")
        if binaryRep.count('1') == variablesCount:
            binaryCombos.append(binaryRep)

    for combo in binaryCombos:
        systemMatrix = []
        constantVector = []
        # Building the system of equations based on the combination
        for idx, bit in enumerate(combo):
            if bit == '1':
                equation, constant = constraints[idx]
                systemMatrix.append(equation)
                constantVector.append(constant)

        # Solving the system if it is valid (square and full rank)
        if np.linalg.matrix_rank(systemMatrix) == variablesCount:
            try:
                solution = np.linalg.solve(systemMatrix, constantVector)
                potentialIntersections.append(tuple(solution))
            except np.linalg.LinAlgError:
                continue

    # Eliminating duplicates from the list of intersections
    uniqueIntersections = set(potentialIntersections)
    return sorted(list(uniqueIntersections))


def findFeasibleIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    feasible intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).

        If none of the lines intersect with each other, return [].
        If none of the intersections are feasible, return [].

    You will want to take advantage of your findIntersections function.

    """
    "*** YOUR CODE HERE ***"
    #https://www.stata.com/manuals13/m-1tolerance.pdf
    
    intersectionPoints = findIntersections(constraints)
    feasibleIntersections = []

    tolerance = 1e-12

    for point in intersectionPoints:
        isFeasible = True
        for constraint in constraints:
            coefficients, constant = constraint
            result = np.dot(coefficients, point)
            if result - constant > tolerance:
                isFeasible = False
                break  # No need to check further constraints if one is violated
        
        if isFeasible:
            feasibleIntersections.append(point)

    return feasibleIntersections

def solveLP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    find a feasible point that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your findFeasibleIntersections function.

    """
    "*** YOUR CODE HERE ***"

    feasiblePoints = findFeasibleIntersections(constraints)
    
    if not feasiblePoints:
        return None  

    # Initialize variables to track the minimum objective value and the corresponding point
    minObjectiveValue = float('inf')
    optimalPoint = None

    # find the one that minimizes the objective function
    for point in feasiblePoints:
        objectiveValue = np.dot(cost, point)  # Calculate the objective function value
        
        # Update the minimum objective value and the corresponding point if necessary
        if objectiveValue < minObjectiveValue:
            minObjectiveValue = objectiveValue
            optimalPoint = point

    return (optimalPoint, minObjectiveValue)

def wordProblemLP():
    """
    Formulate the word problem from the write-up as a linear program.
    Use your implementation of solveLP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
            ((sunscreen_amount, tantrum_amount), maximal_utility)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"
    # Constraints defined as ((a1, a2), b)
    constraints = [
        ((2.5, 2.5), 100),  # Space constraint
        ((0.5, 0.25), 50),  # Weight constraint
        ((-1, 0), -20),  # Minimum sunscreen requirement (- for 'at least')
        ((0, -1), -15.5)  # Minimum Tantrum requirement (- for 'at least')
    ]
    
    # Cost vector for the objective function (negative for maximization)
    cost = (-7, -4)
    
    optimalPoint, negativeMaxUtility = solveLP(constraints, cost)
    
    # negative costs for maximization, the optimal utility is negative of the result
    maximalUtility = -negativeMaxUtility
    
    return (optimalPoint, maximalUtility)
    
def isInt(x):
    return np.abs(x - int(round(x))) <= 1e-12

def solveIP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    use the branch and bound algorithm to find a feasible point with
    interger values that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your solveLP function.

    """
    "*** YOUR CODE HERE ***"
    def is_integer(solution):
        # Check if the solution is close enough to integer values
        return all(np.abs(x - round(x)) < 1e-12 for x in solution)

    def convert_constraints(constraints):
        return [(tuple(a), b) for a, b in constraints]

    best_solution = None
    best_objective = float('inf')

    nodes_to_explore = PriorityQueue()
    initial_node = (convert_constraints(constraints), 0) 
    nodes_to_explore.push(initial_node, 0)

    while not nodes_to_explore.isEmpty():
        current_node = nodes_to_explore.pop()
        current_constraints, _ = current_node

        lp_result = solveLP(current_constraints, cost)
        if lp_result is None:  
            continue

        lp_solution, lp_objective = lp_result
        if lp_objective < best_objective:
            if is_integer(lp_solution):
                best_solution = lp_solution
                best_objective = lp_objective
            else:
                for i, val in enumerate(lp_solution):
                    if not is_integer([val]): 
                        floor = np.floor(val)
                        ceil = np.ceil(val)

                        floor_constraints = list(current_constraints)
                        new_constraint_floor = list(np.zeros(len(cost)))
                        new_constraint_floor[i] = 1
                        floor_constraints.append((tuple(new_constraint_floor), floor))
                        nodes_to_explore.push((convert_constraints(floor_constraints), None), lp_objective)

                        ceil_constraints = list(current_constraints)
                        new_constraint_ceil = list(np.zeros(len(cost)))
                        new_constraint_ceil[i] = -1
                        ceil_constraints.append((tuple(new_constraint_ceil), -ceil))
                        nodes_to_explore.push((convert_constraints(ceil_constraints), None), lp_objective)
                        break  

    if best_solution is not None:
        return tuple(best_solution), best_objective
    else:
        return None
    
    
def wordProblemIP():
    """
    Formulate the word problem in the write-up as a linear program.
    Use your implementation of solveIP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
        ((f_DtoG, f_DtoS, f_EtoG, f_EtoS, f_UtoG, f_UtoS), minimal_cost)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"
    #Minimum food requirements for the communities
    # For Gates: Dunkin + Eatunique + Underground >= 15
    demand_gates = ((-1, 0, -1, 0, -1, 0), -15)
    # For Sorrells: Dunkin + Eatunique + Underground >= 30
    demand_sorrells = ((0, -1, 0, -1, 0, -1), -30)

    # Capacity constraints for each provider to both communities
    cap_dunkin_gates = ((1.2, 0, 0, 0, 0, 0), 30) # Maximum weight Dunkin can send to Gates
    cap_dunkin_sorrells = ((0, 1.2, 0, 0, 0, 0), 30) # Maximum weight Dunkin can send to Sorrells
    cap_eatunique_gates = ((0, 0, 1.3, 0, 0, 0), 30) # Maximum weight Eatunique can send to Gates
    cap_eatunique_sorrells = ((0, 0, 0, 1.3, 0, 0), 30) # Maximum weight Eatunique can send to Sorrells
    cap_underground_gates = ((0, 0, 0, 0, 1.1, 0), 30) # Maximum weight Underground can send to Gates
    cap_underground_sorrells = ((0, 0, 0, 0, 0, 1.1), 30) # Maximum weight Underground can send to Sorrells

    # Non-negativity constraints for all food units
    pos_dunkin_gates = ((-1, 0, 0, 0, 0, 0), 0)
    pos_dunkin_sorrells = ((0, -1, 0, 0, 0, 0), 0)
    pos_eatunique_gates = ((0, 0, -1, 0, 0, 0), 0)
    pos_eatunique_sorrells = ((0, 0, 0, -1, 0, 0), 0)
    pos_underground_gates = ((0, 0, 0, 0, -1, 0), 0)
    pos_underground_sorrells = ((0, 0, 0, 0, 0, -1), 0)

    # Combine all constraints into a single list
    all_constraints = [
        demand_gates, 
        demand_sorrells,
        cap_dunkin_gates, 
        cap_dunkin_sorrells,
        cap_eatunique_gates, 
        cap_eatunique_sorrells,
        cap_underground_gates, 
        cap_underground_sorrells,
        pos_dunkin_gates, 
        pos_dunkin_sorrells,
        pos_eatunique_gates, 
        pos_eatunique_sorrells,
        pos_underground_gates, 
        pos_underground_sorrells
    ]

    transportation_cost = (12, 20, 4, 5, 2, 1)  # Costs associated with each delivery route

    optimal_distribution = solveIP(all_constraints, transportation_cost)

    # Check if a solution exists
    if optimal_distribution:
        # Return the optimal distribution and cost
        distribution, cost_value = optimal_distribution
        return (distribution, cost_value)
    else:
        # Return None if no solution exists
        return None

def foodDistribution(truck_limit, W, C, T):
    """
    Given M food providers and N communities, return the integer
    number of units that each provider should send to each community
    to satisfy the constraints and minimize transportation cost.

    Input:
        truck_limit: Scalar value representing the weight limit for each truck
        W: A tuple of M values representing the weight of food per unit for each 
            provider, (w1, w2, ..., wM)
        C: A tuple of N values representing the minimal amount of food units each
            community needs, (c1, c2, ..., cN)
        T: A list of M tuples, where each tuple has N values, representing the 
            transportation cost to move each unit of food from provider m to
            community n:
            [ (t1,1, t1,2, ..., t1,n, ..., t1N),
              (t2,1, t2,2, ..., t2,n, ..., t2N),
              ...
              (tm,1, tm,2, ..., tm,n, ..., tmN),
              ...
              (tM,1, tM,2, ..., tM,n, ..., tMN) ]

    Output: A length-2 tuple of the optimal food amounts and the corresponding objective
            value at that point: (optimial_food, minimal_cost)
            The optimal food amounts should be a single (M*N)-dimensional tuple
            ordered as follows:
            (f1,1, f1,2, ..., f1,n, ..., f1N,
             f2,1, f2,2, ..., f2,n, ..., f2N,
             ...
             fm,1, fm,2, ..., fm,n, ..., fmN,
             ...
             fM,1, fM,2, ..., fM,n, ..., fMN)

            Return None if there is no feasible solution.
            You may assume that if a solution exists, it will be bounded,
            i.e. not infinity.

    You can take advantage of your solveIP function.

    """
    M = len(W)
    N = len(C)

    "*** YOUR CODE HERE ***"
    # Initializing the constraints list for optimization
    opt_constraints = []

    # Constructing constraints for each community's requirement and the truck's weight limit
    for com_idx in range(N):
        community_constraint = [0 for _ in range(M * N)]
        for prov_idx in range(M):
            # Adjusting indices to flatten the 2D structure into 1D
            food_idx = prov_idx * N + com_idx
            community_constraint[food_idx] = -1

            # Creating constraints for truck weight limit and non-negativity of food units
            weight_constraint = [0 if i != food_idx else W[prov_idx] for i in range(M * N)]
            non_negativity_constraint = [0 if i != food_idx else -1 for i in range(M * N)]

            # Adding constraints to the list
            opt_constraints.extend([
                (tuple(weight_constraint), truck_limit),
                (tuple(non_negativity_constraint), 0)
            ])
        # Ensuring each community's needs are met
        opt_constraints.append((tuple(community_constraint), -C[com_idx]))

    # Defining the cost function based on transportation costs
    cost_function = []
    for prov_costs in T:
        cost_function.extend(prov_costs)

    optimal_solution = solveIP(opt_constraints, cost_function)

    if optimal_solution:
        optimal_food_distribution, minimal_cost = optimal_solution
        return (optimal_food_distribution, minimal_cost)
    else:
        return None

    
if __name__ == "__main__":
    constraints = [((3, 2), 10),((1, -9), 8),((-3, 2), 40),((-3, -1), 20)]
    inter = findIntersections(constraints)
    print(inter)
    print()
    valid = findFeasibleIntersections(constraints)
    print(valid)
    print()
    print(solveLP(constraints, (3,5)))
    print()
    print(solveIP(constraints, (3,5)))
    print()
    print(wordProblemIP())
