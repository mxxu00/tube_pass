import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import globalvar as gv
import pickle

poly_n = 0
line_number = 0
# 定义目标函数
def objective_function(coeffs):
    # 将系数重新整形为形状为(line_number, poly_n+1)的矩阵
    coeffs_matrix = np.reshape(coeffs, (line_number * 2, poly_n + 1))

    # 初始化目标函数值
    objective = 0

    # 遍历线段
    for i in range(line_number):
        j = i * 2
        # 获取线段的起点和终点
        t_start = filter_line_coord[i][2]
        t_end = filter_line_coord[i+1][2]
        length = (t_end - t_start)/100
        # 计算多项式二阶导数的模的平方 θ=0
        for t in np.linspace(t_start, t_end, 100):
            derivative2_x = np.polyval(np.polyder(coeffs_matrix[j], 2), t)
            derivative2_y = np.polyval(np.polyder(coeffs_matrix[j+1], 2), t)
            derivative1_x = np.polyval(np.polyder(coeffs_matrix[j], 1), t)
            derivative1_y = np.polyval(np.polyder(coeffs_matrix[j+1], 1), t)
            objective += ((derivative2_x ** 2 + derivative2_y ** 2) * np.sqrt(derivative1_x ** 2 + derivative1_y ** 2)) * length
            # 因为x和y没有关联性 所以曲线对t的导数就是x对t和y对t的导数的平方和
            
        # 计算多项式二阶导数的模的平方 θ=pi
        for t in np.linspace(t_start, t_end, 100):
            derivative2_x = np.polyval(np.polyder(coeffs_matrix[j], 2), t)
            derivative2_y = np.polyval(np.polyder(coeffs_matrix[j+1], 2), t)
            derivative1_x = np.polyval(np.polyder(coeffs_matrix[j], 1), t)
            derivative1_y = np.polyval(np.polyder(coeffs_matrix[j+1], 1), t)
            objective += ((derivative2_x ** 2 + derivative2_y ** 2) * np.sqrt(derivative1_x ** 2 + derivative1_y ** 2)) * length
            # 因为x和y没有关联性 所以曲线对t的导数就是x对t和y对t的导数的平方和
            
    return objective 

# 定义约束条件函数
def equality_constraint(coeffs):
    # 四个参数分别对应：θ的0和pi、线段数量、x和y、多项式阶数
    coeffs_matrix = np.reshape(coeffs, (2, line_number, 2, (poly_n + 1) * 2)) # 
    # 初始化用于存储约束条件值的列表
    constraints = []
    # 等式约束1：0到3阶导相等
    for i in range(1, line_number):
        # 获取相邻线段的连接点t值
        j = i * 2 - 2 # 1 <-> 0123   2 <-> 2345
        t = filter_line_coord[i][2]

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

    # print(constraints)
    return np.array(constraints) # 将约束条件作为numpy数组返回

def inequality_constraint(coeffs):
    coeffs_matrix = np.reshape(coeffs, (line_number * 2, (poly_n + 1) * 2))
    # 不等式约束1：收缩率
    # 不等式约束2：简单连接
    # 不等式约束3：regular tube
    # 不等式约束4：障碍物距离控制

def start(coeffs, coord):
    
    global line_number, poly_n, filter_line_coord
    poly_n = gv.smooth_poly_n

    # 定义约束条件
    equality_constraints = {'type': 'eq', 'fun': equality_constraint}
    inequality_constraints = {'type': 'ineq', 'fun': inequality_constraint}
    constraints = [equality_constraints, inequality_constraints]

    # 案例输入
    initial_coord = np.array(coord)

    initial_coeffs = 0

    # 进行优化
    result = minimize(objective_function, initial_coeffs, constraints=constraints)
    print(result)

    # 获取最优解
    optimized_coeffs = result.x
    print(optimized_coeffs)

    # 绘制最优曲线 
    # plt.plot(initial_coord[:, 0], initial_coord[:, 1], label='Initial',color='orange')
    # plot_curve(optimized_coeffs, filter_line_coord)

    return optimized_coeffs


def find_lambda_min(obstacle, coeffs_gamma, coord):
    coeffs_matrix = np.reshape(coeffs_gamma, (line_number * 2, poly_n + 1))
    lambda_min_set = []
    for i in obstacle:
        lambda_temp = gv.lambda_max
        t_temp = 0
        findout_flag = 0
        for j in range(line_number):
            t_start = coord[j][2]
            t_end = coord[j+1][2]
            # 计算多项式二阶导数的模的平方
            for t in np.linspace(t_start, t_end, 100):
                x_value = np.polyval(np.polyder(coeffs_matrix[j], 2), t)
                y_value = np.polyval(np.polyder(coeffs_matrix[j+1], 2), t)

                distance = np.sqrt((x_value - i.x)**2 + (y_value - i.y)**2) - i.r 
                if(distance < lambda_temp):
                    lambda_temp = distance
                    t_temp = t
                    findout_flag = 1
        if(findout_flag == 1):
            lambda_min_set.append([t_temp, lambda_temp])
    
    return lambda_min_set
       
def get_derivative(coeffs, order):
    coeffs_matrix = np.reshape(coeffs, (line_number * 2, poly_n+1))
    derivative = []
    for i in range(line_number * 2):
        derivative.append(np.polyder(coeffs_matrix[i], order))
    return derivative    

def gen_tube(coeffs, coord, derivative, radius):
    coeffs_matrix = np.reshape(coeffs, (line_number * 2, poly_n+1))
    derivative_matrix = np.reshape(derivative, (line_number * 2, poly_n))
    tube = np.zeros((line_number, gv.sim_step_length, 2, 2)) # 分别对应 点的位置、θ=0 or pi、x和y值
    for i in range(line_number):
        j = i * 2
        t_start = coord[i][2]
        t_end = coord[i+1][2]
        for t,k in zip(np.linspace(t_start, t_end, gv.sim_step_length), range(gv.sim_step_length)):
            
            x_value = np.polyval(coeffs_matrix[j], t)
            y_value = np.polyval(coeffs_matrix[j+1], t)
            
            derivative_x = np.polyval(derivative_matrix[j],t)
            derivative_y = np.polyval(derivative_matrix[j+1],t)

            derivative_x_1 = derivative_x / np.sqrt(derivative_x ** 2 + derivative_y ** 2)
            derivative_y_1 = derivative_y / np.sqrt(derivative_x ** 2 + derivative_y ** 2)

            normal_x = - derivative_y_1
            normal_y = derivative_x_1

            # tube[i][k][0][0] = x_value + radius
            # tube[i][k][0][1] = y_value - radius
            # tube[i][k][1][0] = x_value - radius
            # tube[i][k][1][1] = y_value + radius

            tube[i][k][0][0] = x_value + normal_x * radius
            tube[i][k][0][1] = y_value + normal_y * radius
            tube[i][k][1][0] = x_value - normal_x * radius + 2.5
            tube[i][k][1][1] = y_value - normal_y * radius - 1
 
    return tube

# def getns():
# def getbs(): 暂时不需要 升级三维后添加

def main():
    global line_number, poly_n
    
    # 调试参数
    gv.smooth_poly_n = 5
    poly_n = gv.tubegen_poly_n = 5
    gv.lambda_max = 3
    
    # 暂存数据读取
    f = open('savedata.pkl', 'rb')
    savedata = pickle.load(f)
    myobstacle = savedata['obstacle']
    mycoeffs_gamma = savedata['pathcoeffs']
    mycoord = savedata['pathpoint']
    line_number = len(mycoord) - 1
    
    # lambda_min_set = find_lambda_min(myobstacle, mycoeffs_gamma, mycoord)
    # print(lambda_min_set)
    derivative_set = get_derivative(mycoeffs_gamma, 1)
    tube_set = gen_tube(mycoeffs_gamma, mycoord, derivative_set, 1)

    
    # 画图调试
    coeffs = np.reshape(mycoeffs_gamma, (line_number * 2, gv.smooth_poly_n + 1))
    coord = np.array(mycoord)
    for i in range(line_number):
        j = i * 2
        t = np.linspace(coord[i][2], coord[i+1][2], 100)
        x = np.polyval(coeffs[j], t)
        y = np.polyval(coeffs[j+1], t) 
        # print(x,y)
        plt.scatter(tube_set[i,:,0,1],gv.width - tube_set[i,:,0,0], s = 0.2)
        plt.scatter(tube_set[i,:,1,1],gv.width - tube_set[i,:,1,0], s = 0.2)
        plt.plot(y, gv.width - x, color = 'green', linewidth = 0.5)

    plt.axis('equal')
    plt.show()

    f2 = open('tubedata.pkl','wb')
    pickle.dump(tube_set,f2)
    # start(mycoeffs, mypathpoint)

if __name__ == '__main__':
    main()