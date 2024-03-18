import numpy as np

# poly_n 多项式阶数
# der_order 求导阶数
def compute_qall(poly_n, line_number, der_order, t):
    Q_all = np.empty((0, 0))
    for i in range(line_number):
        Q_i = _compute_q(poly_n, der_order, t[i], t[i+1])
        if Q_all.size == 0:
            Q_all = Q_i
        else:
            Q_all = np.block([[Q_all, np.zeros((Q_all.shape[0], Q_i.shape[1]))],
                            [np.zeros((Q_i.shape[0], Q_all.shape[1])), Q_i]])
    b_all = np.zeros((Q_all.shape[0], 1))  

    return Q_all, b_all

def _compute_q(poly_n, der_order, t1, t2):
    T = np.zeros((poly_n-der_order)*2+1)
    for i in range(1, (poly_n-der_order)*2+2):
        T[i-1] = t2**i - t1**i
    
    Q = np.zeros((poly_n, poly_n))
    for i in range(der_order+1, poly_n+1):
        for j in range(i, poly_n+1):
            k1 = i - der_order - 1
            k2 = j - der_order - 1
            k = k1 + k2 + 1
            Q[i-1, j-1] = np.prod(range(k1+1, k1+der_order+1)) * np.prod(range(k2+1, k2+der_order+1)) / k * T[k-1]
            Q[j-1, i-1] = Q[i-1, j-1]
    
    return Q

# def rearrange():
