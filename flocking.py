import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Boids:
    def __init__(self, num, dt, target_array, obstacle_set, scale_coeff):
        self.num = num  # agent
        self.d = 2
        # self.target_now = np.zeros((num, self.d))
        self.target_now = np.array([0,0])
        self.target_step = 0
        self.target = target_array
        # self.target = np.array([[50,50]])
        # self.predators = np.random.random((Np, self.d*2)) 
        # 前半部分是速度 后半部分是位置s
        self.agents_v = np.random.random((num, self.d))
        self.agents_c = np.random.random((num, self.d))

        self.agents_v_next = np.zeros((num, self.d))
        self.agents_c_next = np.zeros((num, self.d))

        # self.agents_v = np.zeros((num, self.d))
        # self.agents_c = np.zeros((num, self.d))
        self.dt = dt
        self.scale_coeff = scale_coeff

        self.radius_repulsion = 30 * self.scale_coeff
        self.radius_obstacle = 10 * self.scale_coeff
        self.radius_alignment = 10 * self.scale_coeff

        self.v_target_max = 20  * self.scale_coeff
        self.v_target = 10  * self.scale_coeff
        self.v_max = 20  * self.scale_coeff

        self.obstacle_set = obstacle_set

        self.obstacle_p = np.mat([[0,10],[15,0]]) 

        self.repulsion_p = np.mat([[0,7],[10,0]]) * self.scale_coeff

        self.closure_p = np.mat([[0,2],[15,0]]) * self.scale_coeff

    def init(self, scale):
        self.agents_v = np.zeros((self.num, self.d))
        self.agents_c = np.random.random((self.num, self.d)) * scale

    def transition(self,m,str):
        if(str == 'closure'):
            p = self.closure_p
        elif(str == 'obstacle'):
            p = self.obstacle_p
        elif(str == 'repulsion'):
            p = self.repulsion_p
        k = (p[1,1] - p[0,1]) / (p[1,0] - p[0,0])
        n = - k * p[1,0]
        result = np.piecewise(m, [m<p[0,0], np.logical_and(m>=p[0,0], m<p[1,0]), m>=p[1,0]], [p[0,1],lambda x:k*x + n,0])
        return result

    def _pairwise_distances(self, X, Y):  # 这里注意一下 X的长度决定最后计算出来的速度长度
        # 计算相对距离distance 返回标量
        D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
        # D[D < 0] = 0
        return np.sqrt(D)

    def _target_update(self):
        # target_temp = np.zeros((self.num, self.d))
        target_temp = np.array([0,0])
        for i in range(int(np.size(self.target)/2)):
            if(self.target_step == i):
                d_x = self.target[i,0] - self.target_now[0]
                
                d_y = self.target[i,1] - self.target_now[1]
                d = np.sqrt(d_x**2 + d_y**2)
                # print(d_x,d_y,d,self.dt,self.v_target)
                if(d > (self.dt * self.v_target)):
                    target_temp = [self.target_now[0] + (d_x/d * self.v_target * self.dt), self.target_now[1] + (d_y/d * self.v_target * self.dt)]
                    self.target_now = target_temp
                else:
                    self.target_step +=1
                
                # print(self.target_now)
    
    def _v_obstacle(self):
        point_number = int(np.size(self.obstacle_set)/2)
        point_set = np.reshape(self.obstacle_set, (point_number, 2))
        d_ao_matrix = self._pairwise_distances(self.agents_c, point_set)

        # # 方式1
        # obstacle_flag_set = (d_ao_matrix < self.radius_obstacle).astype(int)
        # d_o_matrix = (1/d_ao_matrix) * obstacle_flag_set

        # 方式2
        d_o_matrix = self.transition(d_ao_matrix, 'obstacle') 
        v_o_matrix = np.diag(np.sum(d_o_matrix, axis=1)) @ self.agents_c  - d_o_matrix @ point_set

        # v_o_matrix = np.clip(v_o_matrix, -self.v_obstacle_max, +self.v_obstacle_max)
            
        return v_o_matrix
    
    def _velocity(self):

        d_aa_matrix = self._pairwise_distances(self.agents_c, self.agents_c)

        # # 方式1
        # d_r_matrix = (1/(d_aa_matrix+np.eye(self.num))) * repulsion_flag_set
        # # 本质是 根据倒数 加权 再乘上相对向量差
        # v_r_matrix = np.diag(np.sum(d_r_matrix, axis=1)) @ self.agents_c - d_r_matrix @ self.agents_c
        # # v_r_matrix = np.clip(v_r_matrix, -self.v_repulsion_max, +self.v_repulsion_max)

        # 方式2
        d_r_matrix = self.transition(d_aa_matrix,'repulsion') 
        v_r_matrix = np.diag(np.sum(d_r_matrix, axis=1)) @ self.agents_c  - d_r_matrix @ self.agents_c

        # 牵引
        # 当前坐标与目标点做差
        # clip方法 把数据截断
        v_target_matrix = self.target_now - self.agents_c
        # d_at_matrix = self._pairwise_distances(self.agents_c, self.target_now)
        # target_flag_set_1 = (d_at_matrix > 10).astype(int)
        # target_flag_set_2 = (d_at_matrix <= 10).astype(int)
        # v_target_matrix = target_flag_set_1 * v_target_matrix / d_at_matrix * 10 * 1.414 + target_flag_set_2 * v_target_matrix
        
        v_target_matrix = np.clip(v_target_matrix, -self.v_target_max, self.v_target_max) 

        # 速度对齐项
        # 物理意义为其他所有速度之和 与 自己的速度 的向量差
        alignment_flag_set = (d_aa_matrix < self.radius_alignment).astype(int)
        v_a_matrix = alignment_flag_set @ self.agents_v - 2*self.agents_v

        # 避障项
        v_o_matrix = self._v_obstacle()
        
        v_total = v_r_matrix + v_target_matrix + v_o_matrix + v_a_matrix * 0.05
        # print(v_r_matrix[0],v_target_matrix[0],v_o_matrix[0])
        
        return v_total 
        # return 10 * np.tanh(0.2 * v_total) # 双曲正切 对v进行一个平滑 确保v的变化在可控的范围内
        
    def evolve(self):
        self._target_update()
        v = 0.1 * self.agents_v + 0.9 * self._velocity()
        self.agents_v = np.clip(v, -self.v_max, +self.v_max)

        # self.agents_v = np.clip(self._velocity(), -self.v_max, +self.v_max)
        self.agents_c_next = self.agents_c + self.dt*self.agents_v

        # self.agents_v = np.clip(self._velocity(), -self.v_max, +self.v_max)
        # self.agents_c = self.agents_c + self.dt*self.agents_v

# def main():       
#     f = open('tubedata.pkl','rb')
#     tube_set = pickle.load(f)
#     num = 1
#     d = 2
#     n_history = 1
#     film = np.zeros((num, d*n_history))
#     boids = Boids(num, tube_set)
    
#     line_number = int(np.size(tube_set) / 10 / 4)
#     for t in range(100):        
#         boids.evolve()
#         film = np.hstack([boids.agents_c, film[:,:-d]])
#         if t > n_history:
#             plt.figure()
#             for i in range(num):
#                 plt.plot(film[i,0],film[i,1], 'ko', markersize = 2)
#                 plt.plot(film[i,::2],film[i,1::2], 'k-', alpha=0.5)
#             bound = 20
#             plt.xlim([-5,bound])
#             plt.ylim([-5,bound])
#             plt.title('t={:.2f}'.format(t * boids.dt))

#             for i in range(line_number):
#                 plt.scatter(tube_set[i,:,0,0],tube_set[i,:,0,1],s = 0.2)
#                 plt.scatter(tube_set[i,:,1,0],tube_set[i,:,1,1],s = 0.2)
#             # plt.axis('equal')
#             plt.savefig('./fig/{}.jpg'.format(t))
#             # plt.show()

def main_2():       
    # f = open('tubedata.pkl','rb')
    # obstacle_set = np.array([[50,50]])
    obstacle_set = np.array([[50,1],[20,5],[70,-6],[100,49],[50,99],[0,49]])
    num = 8
    d = 2
    step_set = np.zeros((num, 1000, d))
    velocity_set = np.zeros((num, 1000, d))
    target_array = np.array([[100,0],[100,100],[0,100],[0,0]])
    boids = Boids(num, 0.05, target_array, obstacle_set, 1)
    # 迭代s
    for t in range(1000):        
        boids.evolve()
        boids.agents_c = boids.agents_c_next

        step_set[:,t,:] = boids.agents_c
        velocity_set[:,t,:] = boids.agents_v

    # for t in range(1000):        
    #     boids.evolve()
    #     step_set[:,t,:] = boids.agents_c
    #     velocity_set[:,t,:] = boids.agents_v

    fig, ax = plt.subplots(1,2,figsize = (16,8))
    bound = 120 
    ax[0].set_xlim(-10,bound)
    ax[0].set_ylim(-10,bound)
    ax[0].axis('equal')
    for i in range(num):
        ax[1].plot(np.sqrt(velocity_set[i,:,0]**2 + velocity_set[i,:,1]**2)) 
    for i in range(1000):        
        ax[0].cla()
        
        for j in range(num):
            ax[0].plot(step_set[j,:i,0],step_set[j,:i,1])
            ax[0].scatter(step_set[j,i,0],step_set[j,i,1])
            ax[0].set_title('t={:.2f}'.format(i * boids.dt))
            
        ax[0].scatter(obstacle_set[:,0],obstacle_set[:,1],color='r',marker='x')
        plt.pause(0.001)

if __name__ == '__main__':
    main_2()
