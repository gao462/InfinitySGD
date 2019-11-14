import numpy as np

def pi2P(pi):
    P  = np.zeros((4, 4))
    for i in range(4 - 1):
        P[i, i + 1] = pi[i + 1]
        P[i + 1, i] = pi[i]
    for i in range(4):
        P[i, i] = 1 - P[i, :].sum()
    return P

def texmat(mx):
    print(r'\begin{bmatrix}')
    for i in range(len(mx)):
        for j in range(len(mx) - 1):
            print(r"{:.1f} & ".format(mx[i, j]), end='')
        print(r"{:.1f} \\".format(mx[i, len(mx) - 1]))
    print(r'\end{bmatrix}')

texmat(pi2P(np.array([0.4, 0.3, 0.2, 0.1])))
print()
texmat(pi2P(np.array([0.4, 0.3, 0.1, 0.2])))