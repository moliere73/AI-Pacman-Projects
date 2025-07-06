# quick_q2_demo.py
# ----------------
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



import logic
import logicPlan

if __name__ == '__main__' :
    
    action_strs = ['North','East','South','West']
    
    model = {
             logic.PropSymbolExpr('P[2,1,0]'): False,
             logic.PropSymbolExpr('East[1]'): False,
             logic.PropSymbolExpr('P[2,2,2]'): False,
             logic.PropSymbolExpr('West[0]'): False, 
             logic.PropSymbolExpr('P[2,1,2]'): False, 
             logic.PropSymbolExpr('P[1,2,1]'): False, 
             logic.PropSymbolExpr('South[1]'): False, 
             logic.PropSymbolExpr('South[0]'): True, 
             logic.PropSymbolExpr('North[1]'): False, 
             logic.PropSymbolExpr('North[0]'): False, 
             logic.PropSymbolExpr('East[0]'): False, 
             logic.PropSymbolExpr('West[1]'): True, 
             logic.PropSymbolExpr('P[1,1,2]'): True, 
             logic.PropSymbolExpr('P[1,1,0]'): False, 
             logic.PropSymbolExpr('P[2,2,1]'): False, 
             logic.PropSymbolExpr('P[1,1,1]'): False, 
             logic.PropSymbolExpr('P[1,2,0]'): False, 
             logic.PropSymbolExpr('P[2,1,1]'): True, 
             logic.PropSymbolExpr('P[1,2,2]'): False, 
             logic.PropSymbolExpr('P[2,2,0]'): True
             }

    plan = logicPlan.extractActionSequence(model, action_strs) 
    print(plan)
    
    
    
