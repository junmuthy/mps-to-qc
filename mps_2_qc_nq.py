import numpy as np
import scipy as sp
import mps_2_qc_2q as m2q



if __name__ == '__main__':
    L = 3
    mps_list = [np.random.normal(size=(3,2,3)) for i in range(L-2)]
    mps_list.insert(0, np.random.normal(size=(2,3)))
    mps_list.append(np.random.normal(size=(2,3)))

    

