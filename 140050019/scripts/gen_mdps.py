import os
import numpy as np

def writeMDP(filename):
    S = 50
    A = 2
    R = np.random.uniform(low = -1, high = 1, size = (S,A,S))
    T = np.random.rand(S,A,S)
    for s in range(S):
        for a in range(A):
            T[s][a] /= np.sum(T[s][a])
    gamma = np.random.uniform(low = 0.9, high = 0.999, size = 1)[0]
    with open(filename, 'w') as f:
        writeNum(f, S)
        writeNum(f, A)
        writeArray(f, R)
        writeArray(f, T)
        writeNum(f, gamma)
    print("{} generated!".format(filename))

def writeNum(f, num):
    f.write(str(num) + '\n')

def writeArray(f, arr):
    S, A, _ = arr.shape
    for s in range(S):
        for a in range(A):
            f.write( '\t'.join(map(str, arr[s][a])) + '\n')

if __name__ == '__main__':
    if not os.path.exists('mdp'): 
        os.makedirs('mdp')
    total = 100
    filenames = [os.path.join('mdp', str(i)) for i in range(total)]
    for filename in filenames:
        writeMDP(filename)
