import numpy as np
import mps_2_qc_nq as mnq


class Optimizer:
    """Object to optimize a quantum circuit"""
    def __init__(self, circuit: list[np.ndarray], sweeps: int, fidelity: float, learning_rate: float):
        """class initialization"""
        self.sweeps = sweeps
        self.circuit = circuit
        self.fidelity = fidelity
        self.learning_rate = learning_rate


def update_mps_conjugate(mps: list[np.ndarray], unis: list[np.ndarray]) -> list[np.ndarray]:
    """
    Contructs a new set of MPS from unitary disentanglers and a
    list of MPS.

    Parameters
    ----------
    mps  : A list of MPS.
    unis : A list of unitaries that disentangle the given MPS.

    Returns
    -------
    new_mps : A list of new MPS built from the original MPS and the unitaries
    
    """
    shape = len(mps)
    assert shape == len(unis)
    new_mps = list()
    ms = mps[0].shape
    us1 = unis[1].shape
    cap = np.einsum('al, ijak', mps[0].conjugate(), unis[1].conjugate())
    us0 = unis[0].shape
    cap = np.einsum('ia, ajkl', unis[0].conjugate(), cap).reshape((us0[0], us1[1]*us1[2]*ms[1]))
    Al, cap, alpha = mnq.matrix_split(cap)
    new_mps.append(Al.reshape((us0[0], alpha)))
    cap = cap.reshape((alpha, us1[1], us1[2], ms[1]))
    for i in range(1, L-1):
        ms = mps[i].shape
        us = unis[i+1].shape
        cs = cap.shape
        coming = np.einsum('ial, jkam', mps[i], unis[i+1]).reshape((ms[0]*us[0], us[1]*ms[2]*us[3]))
        coming = np.einsum('ia, aj', cap.reshape((cs[0]*cs[1], cs[2]*cs[3])), coming)
        Al, cap, alpha = mnq.matrix_split(coming)
        Al = Al.reshape((cs[0], cs[1], alpha))
        new_mps.append(Al)
        cap = cap.reshape((alpha, us[1], ms[2], us[3]))
    ms = mps[-1].shape
    cs = cap.shape
    coming = np.einsum('ijb, b', mps[-1], cap.reshape(-1)).reshape((ms[0]*ms[1], 1))
    q, _ = np.linalg.qr(coming)
    new_mps.append(q.reshape((ms[0], ms[1])))
    return new_mps

        
if __name__ == '__main__':
    test_mpo_list = list()
    layers = 4
    L = 5
    for i in range(layers):
        layer = list()
        for j in range(L-1):
            layer.append(np.random.random((2,2,2,2)))
        layer.insert(0, np.random.random((2,2)))
        test_mpo_list.append(layer)

    dim = 2

    mps_left = [np.random.random((dim, 2, dim)) for _ in range(L-2)]
    mps_left.insert(0, np.random.random((2, dim)))
    mps_left.append(np.random.random((dim, 2)))

    mps_right = [np.random.random((dim, 2, dim)) for _ in range(L-2)]
    mps_right.insert(0, np.random.random((2, dim)))
    mps_right.append(np.random.random((dim, 2)))

    # This part needs to be redone
    precomputed_layers_forward = list()
    precomputed_layers_forward.append(mps_left)
    current = precomputed_layers_forward[0]
    for i in range(layers):
        current = update_mps_conjugate(current, test_mpo_list[i])
        precomputed_layers_forward.append(current)

    # This part is the normal update mps method I think
    precomputed_layers_backward = list()
    precomputed_layers_backward.append(mps_right)
    current = precomputed_layers_backward[0]
    for i in range(layers):
        current = mnq.update_mps(current, test_mpo_list[layers-i-1])
        precomputed_layers_forward.append(current)



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

            
