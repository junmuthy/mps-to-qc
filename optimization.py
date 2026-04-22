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
for n in range(sweeps):
    for i in range(layers):
            left = precomputed_layers_forward[i]
            right = precomputed_layers_backward[i+2]
        for j in range(L):
            
