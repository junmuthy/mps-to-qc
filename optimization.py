import numpy as np


class Optimizer:
    """Object to optimize a quantum circuit"""
    def __init__(self, circuit: list[np.ndarray], sweeps: int, fidelity: float, learning_rate: float):
        """class initialization"""
        self.sweeps = sweeps
        self.circuit = circuit
        self.fidelity = fidelity
        self.learning_rate = learning_rate




test_mpo_list = list()
layers = 4
L = 5
for i in range(layers):
    layer = list()
    for j in range(L-1):
        layer.append(np.random.random((2,2,2,2)))
    layer.insert(0, np.random.random((2,2,2,2)))
    test_mpo_list.append(layer)

# precompute layers here
precomputed_layers_forward = list()
precomputed_layers_backward = list()

sweeps = 2
learning_rate = 0.6
for n in range(sweeps):
    for i in range(layers):
        layer = test_mpo_list[i] # careful of looping through and copying
        left = precomputed_layers_forward[i:i+1]
        right = precomputed_layers_backward[i+1:i+2]
        for j in range(L):
            U = layer[j]        # carefule of looping and copying
            if j == 0:
                pass                # mpo product for zero case
            elif (0 < j < L-1):
                pass                # mpo product for bulk case
            else:
                pass            # mpo product for last one
            U, s, V = np.linalg.svd(F)
            Unew = U.dot(V)
            Utemp = U.conjugate().transpose().dot(Unew)
            evecs, evals = np.linalg.eig(Utemp)
            Utemp = evecs.dot(np.diag(evals**learning_rate)).dot(evecs.conjugate().transpose())
            Uprime = U.dot(Utemp)
            test_mpo_list[i][j] = Uprime # this is scary
            

                
# test = list(range(5))
# print(test[-1:0])
            
            
