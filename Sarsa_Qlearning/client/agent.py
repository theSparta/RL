import numpy as np

ACTIONS = 'up down left right'.split()

class RandomAgent:
    def __init__(self):
        self.step = 0

    def getAction(self):
        '''samples actions in a round-robin manner'''
        self.step = (self.step + 1) % 4
        return 'up down left right'.split()[self.step]

    def observe(self, newState, reward, event):
        pass


class Agent:
    def __init__(self, numStates, state, gamma, lamb, algorithm, randomseed):
        '''
        numStates: Number of states in the MDP
        state: The current state
        gamma: Discount factor
        lamb: Lambda for SARSA agent
        '''
        np.random.seed(randomseed)
        self.lamb = lamb
        if algorithm == 'random':
            self.agent = RandomAgent()
        elif algorithm == 'qlearning':
            self.agent = QLearningAgent(numStates, state, gamma)
        elif algorithm == 'sarsa':
            self.agent = SarsaAgent(numStates, state, gamma, lamb)

    def getAction(self):
        '''returns the action to perform'''
        return self.agent.getAction()

    def observe(self, newState, reward, event):
        '''
        event:
            'continue'   -> The episode continues
            'terminated' -> The episode was terminated prematurely
            'goal'       -> The agent successfully reached the goal state
        '''
        self.agent.observe(newState, reward, event)



class QLearningAgent:

    def __init__(self, numStates, state, gamma):
        self.numStates = numStates
        self.gamma = gamma
        self.Q = np.zeros((numStates, 4))
        self.__initparams__()
        self.curr_state = state
        self.init_state = state
        self.epsilon = 0.2
        self.action = 0

    def getAction(self):
        prob = np.random.random()
        if prob < self.epsilon:
            self.action =  (self.step + 1) % 4
        else:
            self.action =  np.argmax(self.Q[self.curr_state])
        return ACTIONS[self.action]

    def observe(self, newState, reward, event):
        s, a = self.curr_state, self.action
        alpha, gamma = self.alpha, self.gamma
        self.Q[s][a] = self.Q[s][a] + alpha * (reward + gamma* np.max(self.Q[newState]) - self.Q[s][a])
        self.curr_state = newState
        if event != "continue":
            self.__initparams__()
            self.epsilon *= 0.99
        else:
            self.step += 1
            self.epsilon *= 0.9999

    def __initparams__(self):
        self.alpha = 0.2
        self.step = 0
        self.curr_state = self.init_state


class SarsaAgent:

    def __init__(self, numStates, state, gamma, lamb):
        self.numStates = numStates
        self.gamma = gamma
        self.Q = np.zeros((numStates, 4))
        self.lamb = lamb
        self.curr_state = state
        self.init_state = state
        self.epsilon = 0.5
        self.alpha = 0.15
        self.__initparams__()

    def __initparams__(self):
        self.step = 0
        self.e = np.zeros_like(self.Q)
        self.state = self.init_state
        self.curr_action = self.sampleAction(self.curr_state)

    def getAction(self):
        return ACTIONS[self.curr_action]

    def sampleAction(self, state):
        prob = np.random.random()
        if prob < self.epsilon:
            action =  (self.step + 1) % 4
        else:
            action = np.argmax(self.Q[state])
        return action

    def observe(self, newState, reward, event):
        s, a = self.curr_state, self.curr_action
        newAction = self.sampleAction(newState)
        alpha, lamb, gamma = self.alpha, self.lamb, self.gamma
        delta = reward + gamma * self.Q[newState][newAction] - self.Q[s][a]
        self.e[s][a] += 1
        self.Q += alpha * delta * self.e
        self.e *= gamma * lamb
        self.curr_state = newState
        self.curr_action = newAction

        if event != "continue":
            self.__initparams__()
            self.epsilon *= 0.95          
            self.alpha *= 0.999
        else:
            self.step += 1

