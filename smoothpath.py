import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import globalvar as gv

class SmoothPath:
    def __init__(self,coord_input):
        self.filter_line_coord = 1
        self.initial_coord = np.array(coord_input)
        self.line_number = 0
        self.optimized_coeffs = 0
        self.poly_n = 0 
        self.filter_line_coord = 0

    poly_n = 5  # 多项式次数
    def _objective_function(self, coeffs):
        coeffs_matrix = np.reshape(coeffs, (self.line_number * 2, self.poly_n + 1))

        # 初始化目标函数值
        objective = 0

        # 遍历线段
        for i in range(self.line_number):
            j = i * 2
            # 获取线段的起点和终点
            t_start = self.filter_line_coord[i][2]
            t_end = self.filter_line_coord[i+1][2]
            length = (t_end - t_start) / gv.sim_step_length
            t = np.linspace(t_start, t_end, gv.sim_step_length)
            derivative2_x = np.polyval(np.polyder(coeffs_matrix[j], 2), t)
            derivative2_y = np.polyval(np.polyder(coeffs_matrix[j+1], 2), t)
            derivative1_x = np.polyval(np.polyder(coeffs_matrix[j], 1), t)
            derivative1_y = np.polyval(np.polyder(coeffs_matrix[j+1], 1), t)
            objective += np.sum((derivative2_x**2 + derivative2_y**2) * np.sqrt(derivative1_x**2 + derivative1_y**2)) * length
                
        return objective 

    # 定义约束条件函数
    def _constraint_function(self, coeffs):
        # 将系数重新整形为形状为(line_number, poly_n+1)的矩阵
        coeffs_matrix = np.reshape(coeffs, (self.line_number * 2, self.poly_n+1))

        # 初始化用于存储约束条件值的列表
        constraints = []

        # 约束1：0到3阶导相等
        for i in range(1, self.line_number):
            # 获取相邻线段的连接点t值
            j = i * 2 - 2 # 1 <-> 0123   2 <-> 2345
            t = self.filter_line_coord[i][2]

            # 计算多项式在连接点的0到3阶导数值
            for k in range(3):
                line1_x = np.polyval(np.polyder(coeffs_matrix[j], k), t)
                line2_x = np.polyval(np.polyder(coeffs_matrix[j+2], k), t)
                line1_y = np.polyval(np.polyder(coeffs_matrix[j+1], k), t)
                line2_y = np.polyval(np.polyder(coeffs_matrix[j+3], k), t)
                # print(derivative1,derivative2,derivative1-derivative2)
                # 添加约束条件，确保连接点的0到3阶导数相等
                # print(derivative1_x,derivative1_y,derivative2_x,derivative2_y)
                constraints.append(line1_x - line2_x)
                constraints.append(line1_y - line2_y)
        
        # 约束2：过连接点约束 
        i=0
        for i in range(self.line_number):
            j = i * 2
            t = self.filter_line_coord[i][2]
            derivative_start_x = np.polyval(coeffs_matrix[j], t)
            constraints.append(derivative_start_x - self.filter_line_coord[i][0])
            derivative_start_y = np.polyval(coeffs_matrix[j+1], t)
            constraints.append(derivative_start_y - self.filter_line_coord[i][1])
        
        t = self.filter_line_coord[self.line_number][2]
        derivative_start_x = np.polyval(coeffs_matrix[j], t)
        constraints.append(derivative_start_x - self.filter_line_coord[self.line_number][0])
        derivative_start_y = np.polyval(coeffs_matrix[j+1], t)
        constraints.append(derivative_start_y - self.filter_line_coord[self.line_number][1])
                
        # print(constraints)
        return np.array(constraints) # 将约束条件作为numpy数组返回

    # 坐标预处理 模式1 只保留折角及其两端的路径点
    def coord_preprocess_1(self, coord):
        current_vector = None
        coord_number = len(coord)
        total_length = 0
        filter_coord_t = []
        filter_coord_final = []
        filter_coord_flag = np.zeros((gv.width + 1, gv.length + 1))
        # 计算路径长度s参数，同时标记折角点
        for i in range(coord_number - 1):
            vector = [coord[i+1][0] - coord[i][0], coord[i+1][1] - coord[i][1]]
            length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
            filter_coord_t.append(np.append(coord[i], total_length))
            if(vector != current_vector or current_vector is None):
                current_vector = vector
                filter_coord_flag[coord[i-1][0]][coord[i-1][1]] = 1
                filter_coord_flag[coord[i][0]][coord[i][1]] = 1
                filter_coord_flag[coord[i+1][0]][coord[i+1][1]] = 1
            total_length += length
        filter_coord_t.append(np.append(coord[coord_number - 1], total_length))
        filter_coord_flag[coord[0][0]][coord[0][1]] = filter_coord_flag[coord[coord_number - 1][0]][coord[coord_number - 1][1]] = 1
        filter_coord_flag[coord[1][0]][coord[1][1]] = filter_coord_flag[coord[coord_number - 2][0]][coord[coord_number - 2][1]] = 1

        print(filter_coord_t[0][0])
        for i in range(coord_number):
            if(filter_coord_flag[coord[i][0]][coord[i][1]] == 1):
                filter_coord_final.append(filter_coord_t[i]) 
        
        print(filter_coord_final)

        return np.array(filter_coord_final)

    # 坐标预处理 模式2 只计算长度 路径点全保留
    def coord_preprocess_2(self, coord):
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
        coeffs = np.empty([0, self.poly_n + 1]) # 建立一个列为4的空白数组，以方便后续np.append
        for i in range(self.line_number):

            xy = np.array([[coord[i][0], coord[i+1][0]], [coord[i][1], coord[i+1][1]]])
            t = np.array([[coord[i][2], coord[i+1][2]], [1, 1]])
            t = np.matrix(t)
            # 求解系数矩阵
            a = xy @ t.I 
            zero_matrix = np.zeros((2, self.poly_n - 1))
            a = np.append(zero_matrix, a, axis = 1)
            coeffs = np.append(coeffs, a, axis = 0)
            
        return coeffs.flatten() #展平后返回，用于minimize
            
    # 画结果图
    def plot_curve_test(self, coeffs, coord):
        coeffs = np.reshape(coeffs, (self.line_number * 2, self.poly_n + 1))
        
        # coeffs = np.array(coeffs).flatten()
        for j in range(self.line_number):
            i = j * 2
            a_x = coeffs[i]
            a_y = coeffs[i + 1]

            t = np.linspace(coord[j][2], coord[j+1][2], 100)

            x = np.polyval(a_x, t)
            y = np.polyval(a_y, t) 
            # print(x,y)
            plt.plot(x, y, label=f'Curve {j+1}')

        plt.legend()
        plt.axis('equal')
        plt.show()

    def start(self):
        self.poly_n = gv.smooth_poly_n
        
        # 定义约束条件
        constraint = {'type': 'eq', 'fun': self._constraint_function}  # 使用'eq'表示等式约束

        # 预处理线段
        self.filter_line_coord = self.coord_preprocess_2(self.initial_coord)
        self.line_number = len(self.filter_line_coord) - 1
        initial_coeffs = self.initial_coeffs_calc(self.filter_line_coord)
        print(initial_coeffs)

        # 进行优化
        result = minimize(self._objective_function, initial_coeffs, constraints=constraint)
        print(result)

        # 获取最优解
        self.optimized_coeffs = result.x
        print(self.optimized_coeffs)

        # 绘制最优曲线 
        # plt.plot(initial_coord[:, 0], initial_coord[:, 1], label='Initial',color='orange')
        # plot_curve(optimized_coeffs, self.filter_line_coord)

        return self.optimized_coeffs, self.filter_line_coord
    
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