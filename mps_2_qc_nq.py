from functools import reduce
import numpy as np
import scipy as sp
import mps_2_qc_2q as m2q
import copy


def matrix_split(mat: np.ndarray, dbond: int=2, exact: bool=False, split: str='left') -> list[np.ndarray, np.ndarray, int]:
    U, s, V = np.linalg.svd(mat, full_matrices=False)
    # print(s)
    if exact:
        cut = len(s[s > 1e-12])
    else:
        cut = min(len(s[s > 1e-12]), dbond)
    if split == 'left':
        sl = np.diag(s)
        Al = U[:, :cut]
        Ar = sl[:cut, :].dot(V)
        return [Al, Ar, cut]
    elif split == 'right':
        sl = np.diag(s)
        Al = U.dot(sl[:, :cut])
        Ar = V[:cut, :]
        return [Al, Ar, cut]
    elif split == 'center':
        sl = np.diag(np.sqrt(s))
        Al = U.dot(sl[:, :cut])
        Ar = sl[:cut, :].dot(V)
        return [Al, Ar, cut]
    else:
        raise ValueError("split must be left, right, or center")
        


# def split_tensor(left: np.ndarray, right: np.ndarray, dbond: int = 2, where: str = 'mid', svd_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
#     if svd_only:
#         if where == 'mid':
#             ls = left.shape
#             rs = right.shape
#             mat = np.einsum('ija, akl', left, right).reshape((ls[0]*ls[1], rs[1]*rs[2]))
#             U, s, Vd = np.linalg.svd(mat, full_matrices=False)
#             print(s)
#             cut = len(s[s > 1e-8])
#             sl = np.diag(np.sqrt(s))
#             Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
#             Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
#             return (Al, Ar)
#         elif where == 'start':
#             ls = left.shape
#             rs = right.shape
#             mat = np.einsum('ia, ajk', left, right).reshape((ls[0], rs[1]*rs[2]))
#             U, s, Vd = np.linalg.svd(mat, full_matrices=False)
#             print(s)
#             cut = len(s[s > 1e-8])
#             sl = np.diag(np.sqrt(s))
#             Al = U.dot(sl[:, :cut]).reshape((ls[0], cut))
#             Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
#             return (Al, Ar)
#         elif where == 'end':
#             ls = left.shape
#             rs = right.shape
#             mat = np.einsum('ija, ak', left, right).reshape((ls[0]*ls[1], rs[1]))
#             U, s, Vd = np.linalg.svd(mat, full_matrices=False)
#             print(s)
#             cut = len(s[s > 1e-8])
#             sl = np.diag(np.sqrt(s))
#             Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
#             Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1]))
#             return (Al, Ar)
#         else:
#             raise KeyError("where argument must be 'mid', 'start', or 'end'")
#     else:
#         if where == 'mid':
#             ls = left.shape
#             rs = right.shape
#             mat = np.einsum('ija, akl', left, right).reshape((ls[0]*ls[1], rs[1]*rs[2]))
#             U, s, Vd = np.linalg.svd(mat, full_matrices=False)
#             cut = min(len(s[s > 1e-8]), dbond)
#             sl = np.diag(np.sqrt(s))
#             Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
#             Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
#             return (Al, Ar)
#         elif where == 'start':
#             ls = left.shape
#             rs = right.shape
#             mat = np.einsum('ia, ajk', left, right).reshape((ls[0], rs[1]*rs[2]))
#             U, s, Vd = np.linalg.svd(mat, full_matrices=False)
#             cut = min(len(s[s > 1e-8]), dbond)
#             sl = np.diag(np.sqrt(s))
#             Al = U.dot(sl[:, :cut]).reshape((ls[0], cut))
#             Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1], rs[2]))
#             return (Al, Ar)
#         elif where == 'end':
#             ls = left.shape
#             rs = right.shape
#             mat = np.einsum('ija, ak', left, right).reshape((ls[0]*ls[1], rs[1]))
#             U, s, Vd = np.linalg.svd(mat, full_matrices=False)
#             cut = min(len(s[s > 1e-8]), dbond)
#             sl = np.diag(np.sqrt(s))
#             Al = U.dot(sl[:, :cut]).reshape((ls[0], ls[1], cut))
#             Ar = sl[:cut, :].dot(Vd).reshape((cut, rs[1]))
#             return (Al, Ar)
#         else:
#             raise KeyError("where argument must be 'mid', 'start', or 'end'")

    
def truncate_mps_to_two(mps: list[np.ndarray]) -> list[np.ndarray]:
    trunc_mps = list()
    ms1 = mps[1].shape
    theta = np.einsum('ia, aj', mps[0], mps[1].reshape((ms1[0], ms1[1]*ms1[2])))
    Al, coming, alpha = matrix_split(theta)
    trunc_mps.append(Al)
    coming = coming.reshape((alpha, ms1[1], ms1[2]))
    for i in range(2, len(mps)-1):
        cs = coming.shape
        ms = mps[i].shape
        theta = np.einsum('ia, aj', coming.reshape((cs[0]*cs[1], cs[2])),
                          mps[i].reshape((ms[0], ms[1]*ms[2])))
        Al, coming, alpha = matrix_split(theta)
        trunc_mps.append(Al.reshape((cs[0], cs[1], alpha)))
        coming = coming.reshape((alpha, ms[1], ms[2]))
    cs = coming.shape
    ms = mps[-1].shape
    theta = np.einsum('ia, aj', coming.reshape((cs[0]*cs[1], cs[2])),
                      mps[-1])
    Al, coming, alpha = matrix_split(theta)
    cs = coming.shape
    trunc_mps.append(Al.reshape((cs[0], cs[1], alpha)))
    q, r = np.linalg.qr(coming.reshape((cs[0]*cs[1], 1)))
    trunc_mps.append(q.reshape((cs[0], cs[1])))
    return trunc_mps


# def truncate_mps_to_two(mps: list[np.ndarray]) -> list[np.ndarray]:
#     trunc_mps = list()
#     current, coming, _ = matrix_split(mps[0])
#     trunc_mps.append(current)
#     for i in range(1, len(mps)-1):
#         coming = np.einsum('ia, ajk', coming, mps[i])
#         cs = coming.shape
#         current, coming, _ = matrix_split(coming.reshape((cs[0]*cs[1], cs[2])))
#         trunc_mps.append(current.reshape((cs[0], cs[1], 2)))
#     coming = np.einsum('ia, aj', coming, mps[-1])
#     trunc_mps.append(coming)
#     return trunc_mps



def update_mps(mps: list[np.ndarray], unis: list[np.ndarray]) -> list[np.ndarray]:
    """
    Given an MPS and a list of unitaries, this function disentangles
    the given MPS.

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
    cap = np.einsum('ai, aj', mps[0], unis[0].conjugate())
    ms = mps[1].shape
    us = unis[1].shape
    cap = np.einsum('ab, ack, bcij', cap, mps[1], unis[1].conjugate()).reshape((us[2], us[3]*ms[2]))
    us = unis[2].shape
    cs = cap.shape
    ms = mps[2].shape
    coming = np.einsum('jam, iakl', mps[2], unis[2].conjugate()).reshape((us[0]*ms[0], us[2]*us[3]*ms[2]))
    coming = np.einsum('ia, aj', cap, coming)
    Al, cap, alpha = matrix_split(coming, exact=True)
    Al = Al.reshape((us[2], alpha))
    cap = cap.reshape((alpha, us[2], us[3]*ms[2]))
    new_mps.append(Al)
    for i in range(3, shape-1):
        us = unis[i].shape
        cs = cap.shape
        ms = mps[i].shape
        coming = np.einsum('jam, iakl', mps[i], unis[i].conjugate()).reshape((us[0]*ms[0], us[2]*us[3]*ms[2]))
        coming = np.einsum('ia, aj', cap.reshape((cs[0]*cs[1], cs[2])), coming)
        Al, cap, alpha = matrix_split(coming, exact=True)
        Al = Al.reshape((cs[0], cs[1], alpha))
        cap = cap.reshape((alpha, us[2], us[3]*ms[2]))
        new_mps.append(Al)
    us = unis[-1].shape
    cs = cap.shape
    ms = mps[-1].shape
    coming = np.einsum('ja, iakl', mps[-1], unis[-1].conjugate()).reshape((us[0]*ms[0], us[2]*us[3]))
    coming = np.einsum('ia, aj', cap.reshape((cs[0]*cs[1], cs[2])), coming)
    Al, cap, alpha = matrix_split(coming, exact=True)
    Al = Al.reshape((cs[0], cs[1], alpha))
    cap = cap.reshape((alpha, us[2], us[3]))
    new_mps.append(Al)
    cs = cap.shape
    cap = cap.reshape((cs[0]*cs[1], cs[2]))
    Al, cap, alpha = matrix_split(cap, exact=True)
    new_mps.append(Al.reshape((cs[0], cs[1], alpha)))
    cs = cap.shape
    q,r = np.linalg.qr(cap.reshape((cs[0]*cs[1], 1)))
    new_mps.append(q.reshape((cs[0], cs[1])))
    return new_mps




if __name__ == '__main__':
    L = 16
    dim = 3
    mps_list = [np.random.normal(size=(dim,2,dim)) for i in range(L-2)]
    mps_list.insert(0, np.random.normal(size=(2,dim)))
    mps_list.append(np.random.normal(size=(dim,2)))

    m2q.left_orthgonalize_general(mps_list)
    # print(mps_list[0])
    # print([x.shape for x in mps_list])
    psi0 = reduce(np.kron, [np.array([1., 0.]) for x in range(L)])
    # print(mps_list[0].dot(mps_list[0].conjugate().transpose()))
    # print(np.einsum('abi, abj', mps_list[2], mps_list[2].conjugate()))
    # print(np.trace(mps_list[-1].dot(mps_list[-1].conjugate().transpose())))
    # print([x.shape for x in mps_list])
    # print(mps_list[0].transpose().dot(mps_list[0]))
    num_layers = 100
    current_list = copy.deepcopy(mps_list)
    # current_list = mps_list.copy()
    # unitary_layers = list()
    for i in range(num_layers):
        trunc_mps = truncate_mps_to_two(current_list)
        # print([np.einsum('abi, abj', trunc_mps[i], trunc_mps[i].conjugate()) for
        #        i in range(1, len(trunc_mps)-1)])
        # print(np.einsum('ab, ab', trunc_mps[-1], trunc_mps[-1].conjugate()))
        # m2q.left_orthgonalize(trunc_mps)
        # print([x.shape for x in trunc_mps])
        units = m2q.mps_to_unitaries(trunc_mps)
        # print(units[1])
        # print("+")
        # U = m2q.build_circuit(units)
        # wf = m2q.build_wavefunction(current_list)
        # new = U.conjugate().transpose().dot(wf)
        # print(units[0])

        # cap = np.einsum('ai, aj', current_list[0], units[0].conjugate())
        # print(cap)
        # U,s,V = np.linalg.svd(cap)
        # print(U, s, V)
        # vec = m2q.disentangle(trunc_mps, units)
        # print(vec)
        #     unitary_layers.append(units)

        current_list = update_mps(current_list, units)
        # print(mps_list[0])
        # print(current_list[0])
        # print([np.einsum('abi, abj', current_list[i], current_list[i].conjugate()) for
        #        i in range(1, len(current_list)-1)])
        # print(np.einsum('ab, ab', current_list[-1], current_list[-1].conjugate()))
        # # vec = m2q.disentangle(current_list, units)
        # # print(vec.conjugate().transpose().dot(psi0))
        # # # print([x.shape for x in current_list])
        # # # print([x.shape for x in current_list])
        # # print([x.shape for x in current_list])
        # m2q.left_orthgonalize_general(current_list)
        # print(np.einsum('ab, ab', current_list[-1], current_list[-1].conjugate()))
        # vec = m2q.disentangle(current_list, units)
        # print(np.abs(new.conjugate().transpose().dot(psi0))**2)
        # print('=')
        vec = m2q.build_wavefunction(current_list)
        # print(vec)
        print(-1*np.log(np.abs(vec.conjugate().transpose().dot(psi0)))/L)
        print([x.shape for x in current_list])
