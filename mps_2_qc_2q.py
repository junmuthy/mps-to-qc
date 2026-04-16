import numpy as np
import scipy as sp


def left_orthgonalize(mps: list) -> np.ndarray:
    """
    Takes an MPS of bond dimension 2 and left orthogonalizes it.

    Parameters
    ----------
    mps : A list of matrix product states of bond dimension 2.

    Returns
    -------
    norm : The final normalization of the MPS stored in a
           Hermitian matrix.

    Notes
    -----
    Orthonalizes in place.
    
    """
    shape = len(mps)
    q, norm = np.linalg.qr(mps[0])
    mps[0] = q
    for i in range(1, shape-1):
        temp = np.einsum('ia, ajk', norm,  mps[i]).reshape((4,2))
        q, norm = np.linalg.qr(temp)
        mps[i] = q.reshape((2,2,2))
    temp = np.einsum('ia, aj', norm, mps[-1])
    q, norm = np.linalg.qr(temp)
    mps[-1] = q/np.linalg.norm(q)
    return norm


# def right_orthgonalize(mps: list) -> np.ndarray:
#     """
#     Takes an MPS of bond dimension 2 and right orthogonalizes it.

#     Parameters
#     ----------
#     mps : A list of matrix product states of bond dimension 2.

#     Returns
#     -------
#     norm : The final normalization of the MPS stored in a
#            Hermitian matrix.

#     Notes
#     -----
#     Orthonalizes in place.
    
#     """
#     shape = len(mps)
#     q, norm = np.linalg.qr(mps[0])
#     mps[0] = q
#     for i in range(1, shape-1):
#         temp = np.einsum('ia, ajk', norm,  mps[i]).reshape((4,2))
#         q, norm = np.linalg.qr(temp)
#         mps[i] = q.reshape((2,2,2))
#     temp = np.einsum('ia, aj', norm, mps[-1])
#     q, norm = np.linalg.qr(temp)
#     mps[-1] = q/np.linalg.norm(q)
#     return norm


def left_orthgonalize_general(mps: list) -> np.ndarray:
    """
    Takes an MPS and left orthogonalizes it.

    Parameters
    ----------
    mps : A list of matrix product states of bond dimension 2.

    Returns
    -------
    norm : The final normalization of the MPS stored in a
           Hermitian matrix.

    Notes
    -----
    Orthonalizes in place.
    
    """
    shape = len(mps)
    q, norm = np.linalg.qr(mps[0])
    mps[0] = q
    for i in range(1, shape-1):
        ms = mps[i].shape
        ns = norm.shape
        temp = np.einsum('ia, ajk', norm,  mps[i]).reshape((ns[0]*ms[1], ms[2]))
        q, norm = np.linalg.qr(temp)
        qs = q.shape
        mps[i] = q.reshape((ns[0],ms[1],qs[1]))
    temp = np.einsum('ia, aj', norm, mps[-1])
    q, norm = np.linalg.qr(temp)
    mps[-1] = q/np.linalg.norm(q)
    return norm

    
def mps_to_unitaries(mps:list) -> list:
    """
    Creates a list of unitaries which disentangle the input MPS.

    Parameters
    ----------
    mps : A list of matrix product states of bond dimension 2.

    Returns
    -------
    unitaries : A list of unitary matrices which disentangle the
                given MPS.

    Notes
    -----
    The unitaries are returned in the same order as the MPS they
    disentangle, i.e. unitaries[0] disentangles mps[0].
    
    """
    unitaries = list()
    shape = len(mps)
    unitaries.append(mps[0])
    for i in range(1, shape-1):
        A = mps[i].reshape((4,2))
        X = sp.linalg.null_space(A.conjugate().transpose())
        G = np.hstack((A, X)).reshape((2,2,2,2))
        unitaries.append(G)
    A = mps[-1].reshape((4,1))
    X = sp.linalg.null_space(A.conjugate().transpose())
    G = np.hstack((A, X)).reshape((2,2,2,2))
    unitaries.append(G)
    return unitaries


def build_wavefunction(mps:list) -> np.ndarray:
    """
    Constructs the exact state vector from an MPS.

    Parameters
    ----------
    mps : A list of MPS.

    Returns
    -------
    wave_function : The statevector for the MPS.
    
    """
    m0s = mps[0].shape
    m1s = mps[1].shape
    wave_function = np.einsum('ia, ajk', mps[0], mps[1])
    wave_function = wave_function.reshape((m0s[0]*m1s[1], m1s[2]))
    for i in range(2, len(mps)-1):
        m0s = wave_function.shape
        m1s = mps[i].shape
        wave_function = np.einsum('ia, ajk', wave_function, mps[i])
        wave_function = wave_function.reshape((m0s[0]*m1s[1], m1s[2]))
    m0s = wave_function.shape
    m1s = mps[-1].shape
    wave_function = np.einsum('ia, aj', wave_function, mps[-1])
    wave_function = wave_function.reshape((m0s[0]*m1s[1], 1))
    return wave_function


def build_circuit(unis:list) -> np.ndarray:
    """
    Builds the matrix representation of a quantum circuit that
    disentangles a MPS from the given unitaries.

    Parameters
    ----------
    unis : A list of unitaries which disentangle an MPS.

    Returns
    -------
    circuit : A unitary matrix which disentangles the state vector
              of an MPS.
    
    """
    shape = len(unis)
    u0s = unis[0].shape
    u1s = unis[1].shape
    circuit = np.einsum('ia, ajkl', unis[0], unis[1])
    circuit = circuit.reshape((u0s[0]*u1s[1], u1s[2], u1s[3]))
    for i in range(2, shape):
        cs = circuit.shape
        us = unis[i].shape
        circuit = np.einsum('ika, ajlm', circuit, unis[i])
        circuit = circuit.reshape((cs[0]*us[1], cs[1]*us[2], us[3]))
    cs = circuit.shape
    circuit = circuit.reshape((cs[0], cs[1]*cs[2]))
    return circuit
        
            
def disentangle(mps:list, unis:list) -> np.float64:
    """
    Given an MPS and a list of unitaries, this function disentangles
    the given MPS.

    Parameters
    ----------
    mps  : A list of MPS.
    unis : A list of unitaries that disentangle the given MPS.

    Returns
    -------
    cap : The final disentangled statevector.
    
    """
    shape = len(mps)
    assert shape == len(unis)
    cap = np.einsum('ia, ja', mps[0], unis[0].conjugate())
    if shape == 2:
        us = unis[1].shape
        cap = np.einsum('ac, ab, cbji', cap, mps[1], unis[1].conjugate())
        cap = cap.reshape((us[3]*us[2], 1))
        return cap
    cap = np.einsum('ab, ack, bcij', cap, mps[1], unis[1].conjugate())
    for i in range(2, shape-1):
        us = unis[i].shape
        cs = cap.shape
        ms = mps[i].shape
        # print(ms, us, cs)
        cap = np.einsum('abl, cbik, jca', mps[i], unis[i].conjugate(), cap)
        cap = cap.reshape((us[2]*cs[0], us[3], ms[2]))
    us = unis[-1].shape
    cs = cap.shape
    ms = mps[-1].shape
    # print(ms, us, cs)
    cap = np.einsum('ab, cbji, kca', mps[-1], unis[-1].conjugate(), cap)
    cap = cap.reshape((us[3]*us[2]*cs[0], 1))
    return cap
    
        
if __name__ == '__main__':        
    L = 5
    mps_list = [np.random.normal(size=(2,2,2)) + 1j*np.random.normal(size=(2,2,2)) for i in range(L-2)]
    mps_list.insert(0, np.random.normal(size=(2,2)) + 1j*np.random.normal(size=(2,2)))
    mps_list.append(np.random.normal(size=(2,2)) + 1j*np.random.normal(size=(2,2)))

    left_orthgonalize(mps_list)
    U = mps_to_unitaries(mps_list)

    # vec = disentangle(mps_list, U)
    # print(vec)

    Umat = build_circuit(U)
    # print(Umat.dot(Umat.conjugate().transpose()))
    # print(Umat.transpose().conjugate().dot(Umat))

    wf = build_wavefunction(mps_list)
    print(wf.conjugate().transpose().dot(wf))
    # print(Umat.transpose().conjugate().dot(wf))

