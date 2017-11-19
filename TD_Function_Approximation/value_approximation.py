from __future__ import print_function
import numpy as np
import sys

class LinearApproximator:

    def __init__(self, N, weights, lamb):
        self.w = weights
        self.N = N
        self.lamb = lamb
        self.num_states = 6
        self.gamma = 0.99
        self.alpha = 0.001
        self.state_features = [self.get_feature(state) for state in
                                range(1, self.num_states+1)]

    def get_next_state(self, state):
        next_state = 6
        if state == 6:
            prob = np.random.random()
            if (prob > 0.99):
                next_state = -1 # nextstate = -1 means the game has ended
        return next_state

    def get_feature(self, state):
        features = np.zeros(7)
        if 1 <= state and state <= 5:
            features[state - 1] = 2
            features[-1] = 1
        elif state == 6:
            features[state - 1] = 1
            features[-1] = 2
        return features

    def get_value(self, state):
        if(state == -1):
            return 0
        return np.dot(self.w, self.state_features[state-1])

    def print_values(self):
        values = [self.get_value(state) for state in range(1, self.num_states+1)]
        print(*values)

    def randomUpdates(self):
        N, alpha = self.N, self.alpha
        num_states, gamma = self.num_states, self.gamma
        for i in range(N):
            state = (i % num_states) + 1
            next_state = self.get_next_state(state)
            # TD zero update
            delta = gamma * self.get_value(next_state) - self.get_value(state)
            self.w += alpha * delta * self.state_features[state - 1]
            self.print_values()

    def TD_lambda(self):
        N,  gamma = self.N, self.gamma
        alpha, lambd = self.alpha, self.lamb
        num_updates = 0
        while num_updates < N:
            trace = np.zeros_like(self.w)
            s = np.random.randint(5) + 1
            # Continue until we reach the terminal state
            while(s != -1):
                next_s = self.get_next_state(s)
                delta = gamma * self.get_value(next_s) - self.get_value(s)
                # gradient of V(s) is  state_features(s)
                trace = gamma * lamb * trace + self.state_features[s - 1]
                self.w += alpha * delta * trace
                s = next_s
                num_updates += 1
                self.print_values()
                if (num_updates >= N):
                    break


if __name__ == '__main__':

    assert(len(sys.argv) == 11)
    experiment = int(sys.argv[1])
    assert(experiment in [1, 2])
    N = int(sys.argv[2])
    lamb = float(sys.argv[3])
    weights = np.array(map(float, sys.argv[4:11]))
    agent = LinearApproximator(N, weights, lamb)
    if experiment == 1:
        agent.randomUpdates()
    elif experiment == 2:
        agent.TD_lambda()
