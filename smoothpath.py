import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import globalvar as gv
import time
import common_compute as cc
import qpsolvers
from scipy import sparse

class SmoothPath:
    def __init__(self,coord_input):
        self.initial_coord = np.array(coord_input)
        self.filter_line_coord = self._coord_preprocess(self.initial_coord)
        self.line_number = len(self.filter_line_coord) - 1
        
        self.optimized_coeffs = 0
        self.poly_n = gv.smooth_poly_n
        self.q_all = 0
        self.a_eq = 0
        self.b_eq = 0

    # 坐标预处理 只计算长度 路径点全保留
    def _coord_preprocess(self, coord):
        total_length = 0
        filter_coord = []
        for i in range(len(coord) - 1):
            vector = [coord[i+1][0] - coord[i][0], coord[i+1][1] - coord[i][1]]
            length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
            filter_coord.append(np.append(coord[i], total_length))
            # print(filter_coord)
            total_length += length

        filter_coord.append(np.append(coord[len(coord) - 1], total_length))
        return np.array(filter_coord)

    # 计算初始的多项式参数 因为都是直线所以除了一次项和常系数都为0
    def initial_coeffs_calc(self, coord):
        coeffs_x = np.empty([0, self.poly_n + 1]) # 建立一个空白数组，以方便后续np.append
        coeffs_y = np.empty([0, self.poly_n + 1])
        for i in range(self.line_number):

            x = np.array([coord[i][0], coord[i+1][0]])
            y = np.array([coord[i][1], coord[i+1][1]])
            t = np.array([[coord[i][2], coord[i+1][2]], [1, 1]])
            t = np.matrix(t)
            # 求解系数矩阵
            a_x = x @ t.I
            a_y = y @ t.I 
            zero_matrix = np.zeros((1, self.poly_n - 1))
            a_x = np.append(zero_matrix, a_x, axis = 1)
            a_y = np.append(zero_matrix, a_y, axis = 1)
            coeffs_x = np.append(coeffs_x, a_x, axis = 0)
            coeffs_y = np.append(coeffs_y, a_y, axis = 0)
            
        return coeffs_x.flatten(), coeffs_y.flatten() #展平后返回，用于minimize
            
    # 画结果图
    def plot_curve_test(self, coeffs, coord):
        coeffs = np.reshape(coeffs, (self.line_number * 2, self.poly_n + 1))
        
        # coeffs = np.array(coeffs).flatten()
        for j in range(self.line_number):
            i = j * 2
            a_x = coeffs[i]
            a_y = coeffs[i + 1]

            t = np.linspace(coord[j][2], coord[j+1][2], 100)

            x = np.polyval(a_x[::-1], t)
            y = np.polyval(a_y[::-1], t) 
            # print(x,y)
            plt.plot(x, y, label=f'Curve {j+1}')

        plt.legend()
        plt.axis('equal')
        plt.show()

    def _construct_constraints(self, ts, v0, a0, ve, ae, waypts):
        n_coef = self.poly_n + 1
        Aeq = np.zeros((4 * self.line_number + 2, n_coef * self.line_number))
        beq = np.zeros(4 * self.line_number + 2)

        p0 = waypts[0]  # 初始位置
        pe = waypts[-1]  # 终点位置

        # 初始端和末端的时间、加速度、速度、位置的约束 (6 equations)
        Aeq[0:3, 0:n_coef] = [cc.calc_tvec(ts[0], self.poly_n, 0),
                            cc.calc_tvec(ts[0], self.poly_n, 1),
                            cc.calc_tvec(ts[0], self.poly_n, 2)]
        Aeq[3:6, n_coef * (self.line_number - 1):n_coef * self.line_number] = [cc.calc_tvec(ts[-1], self.poly_n, 0),
                                                        cc.calc_tvec(ts[-1], self.poly_n, 1),
                                                        cc.calc_tvec(ts[-1], self.poly_n, 2)]
        beq[0:6] = [p0, v0, a0, pe, ve, ae]

        # 中间点位置约束 (self.line_number-1 equations)
        neq = 6 # 从第7个约束开始
        for i in range(self.line_number - 1):
            Aeq[neq, n_coef * i:n_coef * (i + 1)] = cc.calc_tvec(ts[i + 1], self.poly_n, 0)
            beq[neq] = waypts[i + 1]
            neq += 1

        # 连续约束 ((self.line_number-1)*3 equations)
        for i in range(self.line_number - 1):
            tvec_p = cc.calc_tvec(ts[i + 1], self.poly_n, 0)
            tvec_v = cc.calc_tvec(ts[i + 1], self.poly_n, 1)
            tvec_a = cc.calc_tvec(ts[i + 1], self.poly_n, 2)

            Aeq[neq, n_coef * i:n_coef * (i + 1)] = tvec_p
            Aeq[neq, n_coef * (i + 1):n_coef * (i + 2)] = -tvec_p
            neq += 1

            Aeq[neq, n_coef * i:n_coef * (i + 1)] = tvec_v
            Aeq[neq, n_coef * (i + 1):n_coef * (i + 2)] = -tvec_v
            neq += 1

            Aeq[neq, n_coef * i:n_coef * (i + 1)] = tvec_a
            Aeq[neq, n_coef * (i + 1):n_coef * (i + 2)] = -tvec_a
            neq += 1

        return Aeq, beq

    def start(self):
        # 预处理线段 根据路径长度增加参数t （时间）
        v0 = (self.filter_line_coord[1, :2] - self.filter_line_coord[0, :2]) / np.linalg.norm(self.filter_line_coord[1, :2] - self.filter_line_coord[0, :2])
        ve = (self.filter_line_coord[-1, :2] - self.filter_line_coord[-2, :2]) / np.linalg.norm(self.filter_line_coord[-1, :2] - self.filter_line_coord[-2, :2])
        
        # 初始输入
        initial_coeffs_x, initial_coeffs_y = self.initial_coeffs_calc(self.filter_line_coord)

        t1 = time.time()
        # 进行优化        
        q_all = cc.compute_qall(self.poly_n, self.line_number, 2, self.filter_line_coord[:,2])
        b_all = np.zeros(self.line_number * (self.poly_n + 1))

        # a_eq, b_eq = self._construct_constraints(self.filter_line_coord[:,2], v0[0], 0, ve[0], 0, self.filter_line_coord[:,0])  
        # optimized_coeffs_x = qpsolvers.solve_qp(q_all, b_all, None, None, a_eq, b_eq, solver = 'quadprog')
        
        # a_eq, b_eq = self._construct_constraints(self.filter_line_coord[:,2], v0[1], 0, ve[1], 0, self.filter_line_coord[:,1])
        # optimized_coeffs_y = qpsolvers.solve_qp(q_all, b_all, None, None, a_eq, b_eq, solver = 'quadprog')

        a_eq, b_eq = self._construct_constraints(self.filter_line_coord[:,2], v0[0], 0, ve[0], 0, self.filter_line_coord[:,0])  
        optimized_coeffs_x = qpsolvers.solve_qp(sparse.csr_matrix(q_all), b_all, None, None, sparse.csr_matrix(a_eq), b_eq, solver = 'clarabel')
        
        a_eq, b_eq = self._construct_constraints(self.filter_line_coord[:,2], v0[1], 0, ve[1], 0, self.filter_line_coord[:,1])
        optimized_coeffs_y = qpsolvers.solve_qp(sparse.csr_matrix(q_all), b_all, None, None, sparse.csr_matrix(a_eq), b_eq, solver = 'clarabel')

        t2 = time.time()
        print("optimizing over! cost time = ", ((t2-t1)*1000))
        
        # reshape
        optimized_coeffs_x = np.reshape(optimized_coeffs_x, (self.line_number, self.poly_n + 1)) 
        optimized_coeffs_y = np.reshape(optimized_coeffs_y, (self.line_number, self.poly_n + 1)) 

        self.optimized_coeffs = np.append(optimized_coeffs_x, optimized_coeffs_y, axis = 1)
        self.optimized_coeffs = np.reshape(self.optimized_coeffs, (self.line_number * 2, self.poly_n + 1))

        return optimized_coeffs_x, optimized_coeffs_y, self.filter_line_coord
    
    def plot(self):
        plt.plot(self.initial_coord[:, 0], self.initial_coord[:, 1], label='Initial',color='orange')
        self.plot_curve_test(self.optimized_coeffs, self.filter_line_coord)

def main():
    
    gv.length = 30
    gv.width = 30
    initial_coord = np.array([[13, 7], [14, 6], [15, 5], [16, 5], [17, 6], [18, 6], [19, 6], [20, 6], [21, 6], [22, 6], [23, 6]])

    sp = SmoothPath(initial_coord)
    # # 案例输入
    
    sp.start()
    sp.plot()

if __name__ == '__main__':
    main()