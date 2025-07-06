# quick_q1_demo.py
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
    
    A = logic.PropSymbolExpr('A');
    B = logic.PropSymbolExpr('B');
    C = logic.PropSymbolExpr('C');
    D = logic.PropSymbolExpr('D');

    symbols = [A, B, C, D]
    
    atleast1 = logicPlan.atLeastOne(symbols)
    print(atleast1)
    
    atmost1 = logicPlan.atMostOne(symbols)
    print(atmost1)
    
    exactly1 = logicPlan.exactlyOne(symbols)
    print(exactly1)
    
    model1 = {A:False, B:False, C:False, D:False}
    print(logic.pl_true(atleast1,model1))
    
    model2 = {A:False, B:True, C:False, D:False}
    print(logic.pl_true(atleast1,model2))
    
    model3 = {A:True, B:True, C:False, D:False}
    print(logic.pl_true(atleast1,model2))
    
    
