# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Return the answer to the questions given in the writeup in their corresponding function below.

def question1():
    '''
    At what level of the planning graph was the solution found? If no solution is found, return "NO SOLUTION".
    '''

def question2():
    '''
    How many proposition nodes and action nodes do we have at level 1 of the planning graph? 
    Give your answer as a tuple (# proposition nodes, # action nodes)
    '''

def question3():
    '''
    If we change the starting state to start at a fuel level of 1 instead of 2, 
    why do we not find a solution? 

    A) Mutually exclusive goals 
    B) goals missing 
    C) None of the above, a solution is found.

    Give your answer as "A", "B", or "C"
    '''


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  {}:\t{}'.format(q, str(response)))
