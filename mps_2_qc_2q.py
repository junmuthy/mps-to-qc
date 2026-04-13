import numpy as np
import scipy as sp


def left_orthgonalize(mps):
    shape = len(mps)
    # deal with weird shapes here
    q,r = np.linalg.qr(mps[0])
    mps[0] = q/np.sqrt(2)
    for i in range(1, shape-1):
        temp = np.einsum('ia, ajk', r,  mps[i]).reshape((4,2))
        q,r = np.linalg.qr(temp)
        mps[i] = q.reshape((2,2,2))
    temp = np.einsum('ia, aj', r, mps[-1])
    q,r = np.linalg.qr(temp)
    mps[-1] = q/np.sqrt(2)

def mps_to_unitaries(mps):
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
    return wave_function.transpose()
    


# if __name__ == '__main__':        
L = 3
mps_list = [np.random.normal(size=(2,2,2)) for i in range(L-2)]
mps_list.insert(0, np.random.normal(size=(2,2)))
mps_list.append(np.random.normal(size=(2,2)))

# print(mps_list)
left_orthgonalize(mps_list)

wf = build_wavefunction(mps_list)
print(wf)
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

