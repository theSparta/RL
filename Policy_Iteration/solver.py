#! /usr/bin/python
from pulp import *
import sys
import numpy as np

class MDP:
    def __init__(self, S, A, R, T, gamma):
        self.S = S
        self.A = A
        self.R = R
        self.T = T
        self.gamma = gamma
        self.actions = range(0, A)
        self.states = range(0, S)
        self.__generateMatrices__()
    
    def __generateMatrices__(self):
        S, A, R, T = self.S, self.A, self.R, self.T
        self.b = np.sum(T*R, axis = -1)
        X = np.repeat(np.eye(S), A, axis = 0)
        self.X = X.reshape(S,A,S) 
        self.X -= self.gamma * T

    def getActionValue(self, V, s, a):
        return np.sum(self.T[s][a]*( self.R[s][a] + self.gamma * V))

    def getOptimalAction(self, V, s):
        return np.argmax([self.getActionValue(V, s, a) for a in self.actions])

    # Solve for the optimal values given the optimal actions
    def getOptimalValues(self, actions):
        X = self.X[range(self.S), actions, :]
        y = self.b[range(self.S), actions]
        return np.linalg.solve(X,y)

    def getOptimalActions(self, V, states = None):
        if states is None:
            states = self.states
        return [self.getOptimalAction(V, s) for s in states] 

    def printOptimalPolicy(self, V):
        actions = self.getOptimalActions(V)
        for v, a in zip(V, actions):
            print("{}\t{}".format(v,a))

class LP:
    def __init__(self, mdp):
        prob = LpProblem("Policy evaluation", LpMinimize)
        #A dictionary called 'V' is created to contain the referenced Variable 
        states = mdp.states
        actions = mdp.actions
        V = LpVariable.dicts("Value", states)
        T, R, gamma = mdp.T, mdp.R, mdp.gamma
        # The objective function is added to 'prob' first
        prob += lpSum(V)
        V_vals = np.array([V[i] for i in states])
        for s in states:
            for a in actions:
                prob += V[s] >= lpSum(T[s][a] *(R[s][a] + gamma * V_vals))
        self.mdp = mdp
        self.prob = prob
        self.V = V

    def solve(self, verbose = False):
        self.prob.solve()
        if verbose:
            print("Status: {}".format(LpStatus[self.prob.status]))
 
    def getOptimalValues(self):
        return np.array([self.V[i].varValue for i in self.mdp.states],
                dtype=np.float64)

class PI(object):
    
    def __init__(self, mdp, eps):
        self.mdp = mdp
        self.eps = eps
        self.num_iters = 0
        self.V = None
 
    def getOptimalValues(self):
        return self.V
    
    # Solves the mdp and returns the number of iterations taken
    def solve(self):
        prev_actions = [0] * self.mdp.S
        while True:
            self.num_iters += 1
            V = self.mdp.getOptimalValues(prev_actions)
            improved_actions = self.getImprovedActions(V, prev_actions)
            if improved_actions != prev_actions: #self.changed(V, improvedV):
                prev_actions = improved_actions
            else:
                break
        self.V = V
        return self.num_iters

    def getImprovedActions(self, V, prev_actions):
        raise NotImplementedError()

class howardPI(PI):

    def __init__(self, mdp, eps):
        super(self.__class__, self).__init__(mdp, eps)

    def getImprovedActions(self, V, prev_actions):
        return self.mdp.getOptimalActions(V) # All actions are updated

class randomPI(PI):
     def __init__(self, mdp, eps):
        super(self.__class__, self).__init__(mdp, eps)

     # Only a subset of improvable actions are updated
     def getImprovedActions(self, V, prev_actions):
        improved_actions = prev_actions[:]
        new_actions = self.mdp.getOptimalActions(V)
        actionsToSample = [i for i in self.mdp.states if new_actions[i] != prev_actions[i]]
        if not actionsToSample:
            return prev_actions

        # Sample a subset from actionsToSample with atleast one element
        while True:
            subsetIndices = np.random.randint(2, size= len(actionsToSample))
            if np.sum(subsetIndices) > 0:
                break

        for i, to_sample in zip(actionsToSample, subsetIndices):
            if to_sample:
                improved_actions[i] = new_actions[i]
        return improved_actions

class batchPI(PI):
   
    def __init__(self, mdp, eps, batch_size):
        super(self.__class__, self).__init__(mdp, eps)
        self.batch_size = batch_size
        self.num_batches = mdp.S // batch_size
        self.start_offset = mdp.S % batch_size

    #Only the rightmost improvable batch is updated
    def getImprovedActions(self, V, prev_actions):
        start_offset, batch_size = self.start_offset, self.batch_size
        improved_actions = prev_actions[:]
        for i in range(self.num_batches, -1, -1):
            if i != 0:
                start = start_offset + (i-1) * batch_size
                end = start + batch_size
            else:
                start, end = 0, start_offset
            new_actions = self.mdp.getOptimalActions(V, range(start, end))
            curr_actions = prev_actions[start : end]
            if curr_actions != new_actions:
                improved_actions[start : end] = new_actions
                break
        return improved_actions

def readMdpFile(filename): 
    with open(filename, 'r') as f:
        S = int((f.readline()).strip())
        A = int((f.readline()).strip())
        R = np.empty((S,A,S))
        T = np.empty_like(R)
        for s in range(0, S):
            for a in range(0, A):
                rewards = (f.readline().strip()).split("\t")
                R[s][a] = np.array(map(float, rewards))
        
        for s in range(0, S):
            for a in range(0, A):
                transitions = (f.readline().strip()).split("\t")
                T[s][a] = np.array(map(float, transitions))

        gamma = float((f.readline()).strip())
    return (S, A, R, T, gamma)

if __name__ == '__main__':

    filename = sys.argv[1]
    S, A, R, T, gamma = readMdpFile(filename)
    mdp = MDP(S, A, R, T, gamma)
    
    algo = sys.argv[2]
    eps = 1e-9
    if algo == 'lp':
        solver = LP(mdp)
    elif algo == 'hpi':
        solver = howardPI(mdp, eps)
    elif algo == 'rpi':
        randomseed = int(sys.argv[3])
        np.random.seed(randomseed)
        solver = randomPI(mdp, eps)
    elif algo == 'bspi':
        batchsize = int(sys.argv[3])
        solver = batchPI(mdp, eps, batchsize)
    
    solver.solve()
    V = solver.getOptimalValues()
    mdp.printOptimalPolicy(V)

