import numpy as np
import scipy as sp
import mps_2_qc_2q as m2q


def split_tensor(left, right, dbond=2, where='mid'):
    if where == 'mid':
        ls = left.shape
        rs = right.shape
        mat = np.einsum('ija, akl', left, right).reshape((ls[0]*ls[1], rs[1]*rs[2]))
        U, s, Vd = np.linalg.svd(mat, full_matrices=False)
        sl = np.diag(np.sqrt(s))
        Al = U.dot(sl[:, :dbond]).reshape((ls[0], ls[1], dbond))
        Ar = sl[:dbond, :].dot(Vd).reshape((dbond, rs[1], rs[2]))
        return (Al, Ar)
    elif where == 'start':
        ls = left.shape
        rs = right.shape
        mat = np.einsum('ia, ajk', left, right).reshape((ls[0], rs[1]*rs[2]))
        U, s, Vd = np.linalg.svd(mat, full_matrices=False)
        sl = np.diag(np.sqrt(s))
        Al = U.dot(sl[:, :dbond]).reshape((ls[0], dbond))
        Ar = sl[:dbond, :].dot(Vd).reshape((dbond, rs[1], rs[2]))
        return (Al, Ar)
    elif where == 'end':
        ls = left.shape
        rs = right.shape
        mat = np.einsum('ija, ak', left, right).reshape((ls[0]*ls[1], rs[1]))
        U, s, Vd = np.linalg.svd(mat, full_matrices=False)
        sl = np.diag(np.sqrt(s))
        Al = U.dot(sl[:, :dbond]).reshape((ls[0], ls[1], dbond))
        Ar = sl[:dbond, :].dot(Vd).reshape((dbond, rs[1]))
        return (Al, Ar)
    else:
        raise KeyError("where argument must be 'mid', 'start', or 'end'")

    
def truncate_mps(mps):
    trunc_mps = list()
    current, coming = split_tensor(mps[0], mps[1], where='start')
    trunc_mps.append(current)
    for i in range(2, len(mps)-1):
        current, coming = split_tensor(coming, mps[i])
        trunc_mps.append(current)
    current, coming = split_tensor(coming, mps[-1], where='end')
    trunc_mps.append(current)
    trunc_mps.append(coming)
    return trunc_mps


if __name__ == '__main__':
    L = 3
    mps_list = [np.random.normal(size=(3,2,3)) for i in range(L-2)]
    mps_list.insert(0, np.random.normal(size=(2,3)))
    mps_list.append(np.random.normal(size=(3,2)))

    m2q.left_orthgonalize_general(mps_list)
    # print(mps_list[0].transpose().dot(mps_list[0]))

    trunc_mps = truncate_mps(mps_list)
    print([x.shape for x in trunc_mps])
    m2q.left_orthgonalize(trunc_mps)

