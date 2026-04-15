import numpy as np
import scipy as sp
import mps_2_qc_2q as m2q

def matrix_split(mat, dbond=2, svd_only=False):
    U, s, Vd = np.linalg.svd(mat, full_matrices=False)
    if svd_only:
        cut = len(s[s > 1e-8])
    else:
        cut = min(len(s[s > 1e-8]), dbond)
    sl = np.diag(np.sqrt(s))
    Al = U.dot(sl[:, :cut])
    Ar = sl[:cut, :].dot(Vd)
    return [Al, Ar, cut]


def split_tensor(left, right, dbond=2, where='mid', svd_only=False):
    if svd_only:
        if where == 'mid':
            ls = left.shape
            rs = right.shape
            mat = np.einsum('ija, akl', left, right).reshape((ls[0]*ls[1], rs[1]*rs[2]))
            U, s, Vd = np.linalg.svd(mat, full_matrices=False)
            cut = len(s[s > 1e-8])
            sl = np.diag(np.sqrt(s))
            Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
            Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
            return (Al, Ar)
        elif where == 'start':
            ls = left.shape
            rs = right.shape
            mat = np.einsum('ia, ajk', left, right).reshape((ls[0], rs[1]*rs[2]))
            U, s, Vd = np.linalg.svd(mat, full_matrices=False)
            cut = len(s[s > 1e-8])
            sl = np.diag(np.sqrt(s))
            Al = U.dot(sl[:, :cut]).reshape((ls[0], cut))
            Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
            return (Al, Ar)
        elif where == 'end':
            ls = left.shape
            rs = right.shape
            mat = np.einsum('ija, ak', left, right).reshape((ls[0]*ls[1], rs[1]))
            U, s, Vd = np.linalg.svd(mat, full_matrices=False)
            cut = len(s[s > 1e-8])
            sl = np.diag(np.sqrt(s))
            Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
            Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1]))
            return (Al, Ar)
        else:
            raise KeyError("where argument must be 'mid', 'start', or 'end'")
    else:
        if where == 'mid':
            ls = left.shape
            rs = right.shape
            mat = np.einsum('ija, akl', left, right).reshape((ls[0]*ls[1], rs[1]*rs[2]))
            U, s, Vd = np.linalg.svd(mat, full_matrices=False)
            cut = min(len(s[s > 1e-8]), dbond)
            sl = np.diag(np.sqrt(s))
            Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
            Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
            return (Al, Ar)
        elif where == 'start':
            ls = left.shape
            rs = right.shape
            mat = np.einsum('ia, ajk', left, right).reshape((ls[0], rs[1]*rs[2]))
            U, s, Vd = np.linalg.svd(mat, full_matrices=False)
            cut = min(len(s[s > 1e-8]), dbond)
            sl = np.diag(np.sqrt(s))
            Al = U.dot(sl[:, :cut]).reshape((ls[0], cut))
            Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
            return (Al, Ar)
        elif where == 'end':
            ls = left.shape
            rs = right.shape
            mat = np.einsum('ija, ak', left, right).reshape((ls[0]*ls[1], rs[1]))
            U, s, Vd = np.linalg.svd(mat, full_matrices=False)
            cut = min(len(s[s > 1e-8]), dbond)
            sl = np.diag(np.sqrt(s))
            Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
            Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1]))
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

def update_mps(mps, unis):
    shape = len(mps)
    assert shape == len(unis)
    new_mps = list()
    ms = mps[-1].shape
    us = unis[-1].shape
    cap = np.einsum('ia, jakl', mps[-1], unis[-1].conjugate()).reshape((ms[0]*us[0]*us[2], us[3]))
    ith, Ar, alpha = matrix_split(cap, svd_only=True)
    ith = ith.reshape((ms[0]*us[0], us[2], alpha))
    new_mps.append(Ar)
    # main block
    for i in range(shape-2, 1, -1):
        print(i)
        ms = mps[i].shape
        us = unis[i].shape
        print(ms, us)
        ith_minus_one = np.einsum('ial, jakm', mps[i], unis[i].conjugate()).reshape((ms[0]*us[0], us[2], ms[2]*us[3]))    
        ith, Ar = split_tensor(ith_minus_one, ith, svd_only=True)
        new_mps.append(Ar)
    ms = mps[0].shape
    us = unis[0].shape
    start = np.einsum('ai, aj', mps[0], unis[0].conjugate())
    ms = mps[1].shape
    us = unis[1].shape
    ith_minus_one = np.einsum('ab, acj, bcik', start, mps[1], unis[1].conjugate()).reshape((us[2], ms[2]*us[3]))
    ith, Ar = split_tensor(ith_minus_one, ith, where='start', svd_only=True)
    new_mps.append(Ar)
    new_mps.append(ith)
    return new_mps
    
    
    # np.einsum('ia, jakl', mps[-1], unis[-1])


if __name__ == '__main__':
    L = 4
    dim = 16
    mps_list = [np.random.normal(size=(dim,2,dim)) for i in range(L-2)]
    mps_list.insert(0, np.random.normal(size=(2,dim)))
    mps_list.append(np.random.normal(size=(dim,2)))

    print([x.shape for x in mps_list])    
    m2q.left_orthgonalize_general(mps_list)
    print([x.shape for x in mps_list])
    # print(mps_list[0].transpose().dot(mps_list[0]))

    trunc_mps = truncate_mps(mps_list)
    # print([x.shape for x in trunc_mps])
    m2q.left_orthgonalize(trunc_mps)

    units = m2q.mps_to_unitaries(trunc_mps)

    vec = m2q.disentangle(mps_list, units)
    print(vec)
    print(vec.conjugate().transpose().dot(vec))

    new_list = update_mps(mps_list, units)[::-1]
    print([x.shape for x in new_list])
