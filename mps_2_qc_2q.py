import numpy as np
import scipy as sp


def left_orthgonalize(mps):
    """
    Takes an MPS of bond dimension 2 and left orthonalizes it.

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
    # deal with weird shapes here
    q, norm = np.linalg.qr(mps[0])
    mps[0] = q
    for i in range(1, shape-1):
        temp = np.einsum('ia, ajk', norm,  mps[i]).reshape((4,2))
        q, norm = np.linalg.qr(temp)
        mps[i] = q.reshape((2,2,2))
    temp = np.einsum('ia, aj', norm, mps[-1])
    q, norm = np.linalg.qr(temp)
    mps[-1] = q/np.sqrt(2)
    return norm

    
def mps_to_unitaries(mps):
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
    A = mps_list[-1].reshape((4,1))
    X = sp.linalg.null_space(A.conjugate().transpose())
    G = np.hstack((A, X)).reshape((2,2,2,2))
    unitaries.append(G)
    return unitaries


def build_wavefunction(mps):
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


def build_circuit(unis):
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
        
            
def disentangle(mps, unis):
    """
    Given an MPS and a list of unitaries, this function disentangles
    the given MPS into a product state of all zeros.

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
        cap = np.einsum('abl, cbik, jca', mps[i], unis[i].conjugate(), cap)
        cap = cap.reshape((us[2]*cs[0], ms[2], us[3]))
    us = unis[-1].shape
    cs = cap.shape
    ms = mps[-1].shape
    cap = np.einsum('ab, cbji, kca', mps[-1], unis[-1].conjugate(), cap)
    cap = cap.reshape((us[3]*us[2]*cs[0], 1))
    return cap
    
        
# if __name__ == '__main__':        
L = 3
mps_list = [np.random.normal(size=(2,2,2)) + 1j*np.random.normal(size=(2,2,2)) for i in range(L-2)]
mps_list.insert(0, np.random.normal(size=(2,2)) + 1j*np.random.normal(size=(2,2)))
mps_list.append(np.random.normal(size=(2,2)) + 1j*np.random.normal(size=(2,2)))



# print(mps_list)
left_orthgonalize(mps_list)
# print(len(mps_list))
# print(np.einsum('abi, abj', mps_list[1], mps_list[1]))
U = mps_to_unitaries(mps_list)
# print(len(U))
# print([x.shape for x in U])

vec = disentangle(mps_list, U)
print(vec)

# print(mps_list[1][j,k,l])
# print(U[1][j,k,0,l])

# print(np.einsum('abk, abij', mps_list[1], U[1]))
# print(np.einsum('ai, aj', mps_list[0], U[0]))
# print(np.einsum('ab, abij', mps_list[2], U[2]))

# print([x.shape for x in mps_list])
# print([x.shape for x in U])

# check = np.einsum('ab, bfc, cd, ag, gfke, edji', mps_list[0],
#                   mps_list[1],
#                   mps_list[2],
#                   U[0],
#                   U[1],
#                   U[2])
# check = np.einsum('ab, bc, cd, jida', mps_list[0],
#                   mps_list[1],
#                   U[1],
#                   U[0])
# print(np.einsum('ab, jiba', mps_list[0], U[0]))
# print(check)

# print(mps_list[-1].dot(U[-1]))
          

# Umat = build_circuit(U)
# print(Umat.dot(Umat.conjugate().transpose()))
# print(Umat.transpose().conjugate().dot(Umat))

# # # print([G_norm(x) for x in U[1:]])

# wf = build_wavefunction(mps_list)
# Umat.transpose().conjugate().dot(wf)
# print(Umat.dot(wf.transpose()))
# print(wf)
# print(wf.dot(wf.transpose()))
# wf_orig = np.einsum('ia,ajb,bk', mps_list[0], mps_list[1], mps_list[2])
# print(wf_orig.reshape((8,1)))


        

# print(mps_list[0].dot(mps_list[0].conjugate().transpose()))
# print(mps_list)

# A = mps_list[1].reshape((4,2))
# print(A.conjugate().transpose().dot(A))
# X = sp.linalg.null_space(A.conjugate().transpose())
# print(X)
# G = np.hstack((A, X)).reshape((2,2,2,2))
# # G = np.zeros((2,2,2,2))
# # G[0,:,:,:] = mps_list[1]
# # G[1,:,:,:] = X.reshape((2,2,2))
# # G = G.transpose((0,3,1,2))
# print(np.einsum('ikab, jlab', G, G.conjugate()))
# print(np.outer(np.eye(2), np.eye(2)).reshape((2,2,2,2)))

# A = mps_list[-1].reshape((4,1))
# print(A.conjugate().transpose().dot(A))
# X = sp.linalg.null_space(A.conjugate().transpose())
# print(X)
# # print(X.transpose().dot(X))
# # G = np.zeros((2,2,2,2))
# G = np.hstack((A, X)).reshape((2,2,2,2))

# # G[0,0,:,:] = A
# # G[0,1,:,:] = X[:,0].reshape((2,2))
# # G[1,0,:,:] = X[:,1].reshape((2,2))
# # G[1,1,:,:] = X[:,2].reshape((2,2))
# # # G = G.transpose((0,3,1,2))
# print(np.einsum('ikab, jlab', G, G.conjugate()))
# print(np.outer(np.eye(2), np.eye(2)).reshape((2,2,2,2)))

