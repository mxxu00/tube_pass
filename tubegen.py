import numpy as np
import matplotlib.pyplot as plt
import globalvar as gv
import common_compute as cc
import time
import qpsolvers
from scipy import sparse
import pickle

class TubeGen():
    def __init__(self, ts_mindis, route_l, route_r, mindis_l, mindis_r):
        self.poly_n = gv.tubegen_poly_n
        self.ts = ts_mindis
        self.line_number = len(self.ts) - 1
        self.route_l = route_l
        self.route_r = route_r
        self.mindis_l = mindis_l
        self.mindis_r = mindis_r

    def _construct_eq_constraints(self, ts, v0, a0, ve, ae, waypts):

        n_coef = self.poly_n + 1

        p0 = waypts[0]  # 初始位置
        pe = waypts[-1]  # 终点位置

        Aeq = np.zeros((2*self.line_number + 2, n_coef*self.line_number))
        beq = np.zeros((2*self.line_number + 2))

        neq = 0
        # 起始/终止位置、速度、加速度约束 (4 个方程)
        Aeq[0:2, 0:n_coef] = [cc.calc_tvec(ts[0], self.poly_n, 0),
                            cc.calc_tvec(ts[0], self.poly_n, 1)]
        Aeq[2:4, n_coef*(self.line_number-1):n_coef*self.line_number] = [cc.calc_tvec(ts[-1], self.poly_n, 0),
                                                    cc.calc_tvec(ts[-1], self.poly_n, 1)]
        beq[0:4] = [1/p0, v0, 1/pe, ve]

        neq = 3
        # 连续性约束 ((self.line_number - 1)*2 个方程)
        for i in range(self.line_number - 1):
            tvec_p = cc.calc_tvec(ts[i + 1], self.poly_n, 0)
            tvec_v = cc.calc_tvec(ts[i + 1], self.poly_n, 1)
            
            Aeq[neq, n_coef*(i):n_coef*(i + 2)] = np.concatenate((tvec_p, -tvec_p))
            beq[neq] = 0
            neq += 1
            Aeq[neq, n_coef*(i):n_coef*(i + 2)] = np.concatenate((tvec_v, -tvec_v))
            beq[neq] = 0
            neq += 1

        return Aeq, beq

    def _construct_ieq_constraints(self, ts, waypts, mindis_k):

        n_coef = self.poly_n + 1
        Aieq = np.zeros((2 * self.line_number, n_coef * self.line_number))
        bieq = np.zeros(2 * self.line_number)
        neq = 0

        for i in range(self.line_number):
            # 曲率约束
            coeff_vec = cc.calc_tvec(ts[i + 1], self.poly_n, 0)
            Aieq[neq, n_coef * i:n_coef * (i + 1)] = coeff_vec
            bieq[neq] = 1 / waypts[i]
            neq += 1
            
            # 半径长度
            coeff_vec = cc.calc_tvec(mindis_k[i, 2], self.poly_n, 0)
            Aieq[neq, n_coef*(i):n_coef*(i + 1)] = coeff_vec
            bieq[neq] = 1 / mindis_k[i, 1]
            neq += 1

        Aieq = -Aieq
        bieq = -bieq     

        return Aieq, bieq
    
    def start(self):
        
        q_all_d1 = cc.compute_qall(self.poly_n, self.line_number, 1, self.ts) # 1 阶导
        q_all_d0 = cc.compute_qall(self.poly_n, self.line_number, 0, self.ts) # 0 阶导
        b_all = np.zeros(self.line_number * (self.poly_n + 1))

        w1 = 1
        w2 = 0.01 # 加入这个系数 可以让管道更连续？？
        w1 = w1 / (w1 + w2)
        w2 = w2 / (w1 + w2)

        t1 = time.time()
        # 进行优化        
        
        a_eq, b_eq = self._construct_eq_constraints(self.ts, 0, 0, 0, 0, self.route_l)
        a_ieq, b_ieq = self._construct_ieq_constraints(self.ts, self.route_l, self.mindis_l)
        optimized_coeffs_l = qpsolvers.solve_qp(sparse.csr_matrix(q_all_d1 * w1 + q_all_d0 * w2), b_all,
                                                sparse.csr_matrix(a_ieq), b_ieq,
                                                sparse.csr_matrix(a_eq), b_eq, solver = 'clarabel')
        
        a_eq, b_eq = self._construct_eq_constraints(self.ts, 0, 0, 0, 0, self.route_r)
        a_ieq, b_ieq = self._construct_ieq_constraints(self.ts, self.route_r, self.mindis_r)
        optimized_coeffs_r = qpsolvers.solve_qp(sparse.csr_matrix(q_all_d1 * w1 + q_all_d0 * w2), b_all,
                                                sparse.csr_matrix(a_ieq), b_ieq,
                                                sparse.csr_matrix(a_eq), b_eq, solver = 'clarabel')
        
        # a_eq, b_eq = self._construct_eq_constraints(self.ts, 0, 0, 0, 0, self.route_l)
        # a_ieq, b_ieq = self._construct_ieq_constraints(self.ts, self.route_l, self.mindis_l)
        # optimized_coeffs_l = qpsolvers.solve_qp((q_all_d1 * w1 + q_all_d0 * w2), b_all,
        #                                         a_ieq, b_ieq,
        #                                         a_eq, b_eq, solver = 'osqp')
        
        # a_eq, b_eq = self._construct_eq_constraints(self.ts, 0, 0, 0, 0, self.route_r)
        # a_ieq, b_ieq = self._construct_ieq_constraints(self.ts, self.route_r, self.mindis_r)
        # optimized_coeffs_r = qpsolvers.solve_qp((q_all_d1 * w1 + q_all_d0 * w2), b_all,
        #                                         a_ieq, b_ieq,
        #                                         a_eq, b_eq, solver = 'osqp')

        t2 = time.time()
        print("radius optimized over! cost time = ", ((t2-t1)*1000))
        
        # 获取最优解
        optimized_coeffs_l = np.reshape(optimized_coeffs_l, (self.line_number, self.poly_n + 1)) 
        optimized_coeffs_r = np.reshape(optimized_coeffs_r, (self.line_number, self.poly_n + 1)) 

        return optimized_coeffs_l, optimized_coeffs_r

def main():
    # 测试数据
    f = open('tube_test_data_5.pkl', 'rb')
    save_data = pickle.load(f)
    axes, ts_mindis, tsk_mindis, route_l, route_r, mindis_l, mindis_r, fx, fx1, fx2, fy, fy1, fy2, ft = save_data

    # 半径生成
    tg = TubeGen(ts_mindis, route_l, route_r, mindis_l, mindis_r)
    coeffs_l, coeffs_r = tg.start()

    fr1 = fr2 = np.empty([0])
    # 点集生成
    for k in range(len(coeffs_l)):
        coeffs_l1 = coeffs_l[k]
        coeffs_r1 = coeffs_r[k]
        tt = ft[tsk_mindis[k]:tsk_mindis[k + 1]]
        rr1 = np.polyval(coeffs_l1[::-1], tt)
        fr1 = np.concatenate((fr1, rr1))
        rr2 = np.polyval(coeffs_r1[::-1], tt)
        fr2 = np.concatenate((fr2, rr2))

    tt = ft[tsk_mindis[k + 1]]
    fr1 = np.append(fr1, np.polyval(coeffs_l1[::-1], tt))
    fr2 = np.append(fr2, np.polyval(coeffs_r1[::-1], tt))

    tangent = np.array([fx1, fy1])
    rot = np.array([[0, -1], [1, 0]])
    normal = rot @ tangent
    normal = normal / np.sqrt(np.sum(normal ** 2, axis=0))


    fo1 = np.array([fx, fy]) - 1 / fr1 * normal
    fo2 = np.array([fx, fy]) + 1 / fr2 * normal

    # 画图

    axes.plot(fo1[0], fo1[1], color = 'black', linewidth=2)
    axes.plot(fo2[0], fo2[1], color = 'black', linewidth=2)
    axes.plot(fx, fy, color = 'green', linewidth = 1.5)

    fig2, ax = plt.subplots()
    
    ax.plot(1 / fr1)
    ax.plot(- 1 / fr2)
    ax.plot(ft, label = 'ft')

    plt.show()

    a = 1

if __name__ == '__main__':
    main()