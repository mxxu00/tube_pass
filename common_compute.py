import numpy as np
import globalvar as gv
# poly_n 多项式阶数
# der_order 求导阶数
def compute_qall(poly_n, line_number, der_order, t):
    Q_all = np.empty((0, 0))
    for i in range(line_number):
        Q_i = _compute_q(poly_n, der_order, t[i], t[i+1])
        Q_all = np.block([[Q_all, np.zeros((Q_all.shape[0], Q_i.shape[1]))],
                          [np.zeros((Q_i.shape[0], Q_all.shape[1])), Q_i]])

    return Q_all

def _compute_q(poly_n, der_order, t1, t2):
    T = np.zeros((poly_n-der_order)*2+1)
    for i in range(1, (poly_n-der_order)*2+2):
        T[i-1] = t2**i - t1**i
    
    Q = np.zeros((poly_n + 1, poly_n + 1))
    for i in range(der_order+1, poly_n+1):
        for j in range(i, poly_n+1):
            k1 = i - der_order - 1
            k2 = j - der_order - 1
            k = k1 + k2 + 1
            Q[i-1, j-1] = np.prod(range(k1+1, k1+der_order+1)) * np.prod(range(k2+1, k2+der_order+1)) / k * T[k-1]
            Q[j-1, i-1] = Q[i-1, j-1]
    
    return Q

def calc_tvec(t, n_order, r):
    tvec = np.zeros(n_order+1)
    for i in range(r+1, n_order+2):
        tvec[i-1] = np.prod(range(i-r, i)) * t**(i-r-1)
    return tvec

# def rearrange_t(fx, fx1, fx2, fy, fy1, fy2, ft):

#     rot = np.array([[0, -1], [1, 0]])
#     ff = np.array([fx, fy])
#     ff1 = np.array([fx1, fy1])
#     ff2 = np.array([fx2, fy2])
#     n_o = rot @ ff1  # 定向法向量
#     ff1_unit = ff1 / np.sum(ff1**2, axis=0)**0.5
#     n = np.diff(ff1_unit, axis=1)  # 单位法向量，指向曲线内侧 % 求ff1_unit的一阶差分 某种形式上是曲率的正相关
#     n = np.hstack((n, n[:, -1].reshape(-1, 1)))  # 把最后一列重复一下
#     test = n * n_o
#     flag_dir = np.sum(test, axis=0)  # 单位切向量 乘 差分（曲率） 然后再x+y
#     # 这里x和y总是相同符号的 加在一起判断曲率大小

#     # 查询曲率最大处
#     ts_k = [1]
#     for k in range(2, flag_dir.shape[0] - 1):
#         if flag_dir[k - 1] * flag_dir[k] < 0:
#             # 判断曲线弯曲方向是否反向
#             ts_k.append(k)
#     ts_k.append(k + 1)  # 记录转折点处下标

#     ts_k1 = []  # 记录曲率最大处下标
#     route_r = []  # 记录曲率最大处曲率大小
#     route_l = []  # 记录曲率最大处曲率大小
#     ts = []
#     for k in range(len(ts_k) - 1):
#         curvature = 1 / gv.tube_r_max
#         tmp_k1 = ts_k[k]
#         # 在分段中找到曲率最大的地方的k和曲率大小
#         for k1 in range(ts_k[k], ts_k[k + 1]):
#             tmp = np.linalg.norm(np.cross([ff1[:, k1], 0], [ff2[:, k1], 0])) / np.linalg.norm(ff1[:, k1])**3  # 计算曲率
#             if tmp > curvature:
#                 curvature = tmp
#                 tmp_k1 = k1
#         ts_k1.append(tmp_k1)  # 记录该点下标
#         ts.append(ft[tmp_k1])  # 记录该点t值
#         if flag_dir[tmp_k1] > 0:  # 判断是朝左还是朝右的
#             route_r.append(curvature)
#             route_l.append(1 / gv.tube_r_max)  # 另一边就取最大
#         else:
#             route_l.append(curvature)
#             route_r.append(1 / gv.tube_r_max)

#     # 处理一下开头结尾
#     if ts_k1[0] > 1:
#         ts_k1 = [1] + ts_k1
#         ts = [0] + ts
#         route_r = [1 / gv.tube_r_max] + route_r
#         route_l = [1 / gv.tube_r_max] + route_l
#     if ts_k1[-1] < len(flag_dir):
#         ts_k1.append(len(flag_dir))
#         ts.append(ft[len(flag_dir) - 1])
#         route_l.append(1 / gv.tube_r_max)
#         route_r.append(1 / gv.tube_r_max)

#     # 最后拿到四个数据 ts_k1 下标  ts 此处t值 route_l route_r 代表了左右的曲率

#     # 查询与管道冲突的障碍物
#     mindis_k = np.zeros((len(ts_k1) - 1, 3))
#     for k in range(len(ts_k1) - 1):  # 遍历所有段 先弄 右边 r
#         mindis = gv.tube_r_max
#         mindis_k[k, 0] = ts_k1[k]
#         mindis_k[k, 1] = gv.tube_r_max
#         mindis_k[k, 2] = ft[ts_k1[k]]
#         for k1 in range(ts_k1[k], ts_k1[k + 1]):
#             xp = ff[0, k1]
#             yp = ff[1, k1]
#             bili = n_o[1, k1] / n_o[0, k1]
#             theta = np.arctan(bili)
#             if n_o[0, k1] < 0 and n_o[1, k1] > 0:
#                 theta += np.pi
#             elif n_o[0, k1] < 0 and n_o[1, k1] < 0:
#                 theta -= np.pi
#             for k2 in range(1, gv.tube_r_max + 1):
#                 bw_x = int(k2 * np.cos(theta) + xp)
#                 bw_y = int(k2 * np.sin(theta) + yp)
#                 if occup_map[bw_y, bw_x] == 1:  # bw1是 障碍的集合
#                     break
#             mindis_tmp = k2
#             if mindis_tmp < mindis:
#                 mindis_k[k, 0] = k1  # 下标
#                 mindis_k[k, 1] = mindis_tmp  # 半径
#                 mindis_k[k, 2] = ft[k1]  # t值
#                 mindis = mindis_tmp
#         if mindis_k[k, 0] != 0:
#             tmp_k1 = int(mindis_k[k, 0])
#             tmp_dis = mindis_k[k, 1] 
#             # plot([ff(1,tmp_k1),ff(1,tmp_k1)+tmp_dis*n_o(1,tmp_k1)/norm(n_o(:,tmp_k1))],...
#             #     [ff(2,tmp_k1),ff(2,tmp_k1)+tmp_dis*n_o(2,tmp_k1)/norm(n_o(:,tmp_k1))],'--r');
#     mindis_r = mindis_k

#     n_o = -n_o
#     mindis_k = np.zeros((len(ts_k1) - 1, 3))
#     for k in range(len(ts_k1) - 1):  # 遍历所有段
#         mindis = gv.tube_r_max
#         mindis_k[k, 1] = gv.tube_r_max
#         mindis_k[k, 0] = ts_k1[k]
#         mindis_k[k, 2] = ft[ts_k1[k]]
#         for k1 in range(ts_k1[k], ts_k1[k + 1]):
#             xp = ff[0, k1]
#             yp = ff[1, k1]
#             bili = n_o[1, k1] / n_o[0, k1]
#             theta = np.arctan(bili)
#             if n_o[0, k1] < 0 and n_o[1, k1] > 0:
#                 theta += np.pi
#             elif n_o[0, k1] < 0 and n_o[1, k1] < 0:
#                 theta -= np.pi
#             for k2 in range(1, gv.tube_r_max + 1):
#                 bw_x = int(k2 * np.cos(theta) + xp)
#                 bw_y = int(k2 * np.sin(theta) + yp)
#                 if occup_map[bw_y, bw_x] == 1:
#                     break
#             mindis_tmp = k2
#             if mindis_tmp < mindis:
#                 mindis_k[k, 0] = k1
#                 mindis_k[k, 1] = mindis_tmp
#                 mindis_k[k, 2] = ft[k1]
#                 mindis = mindis_tmp
#         if mindis_k[k, 0] != 0:
#             tmp_k1 = int(mindis_k[k, 0])
#             tmp_dis = mindis_k[k, 1]
#     mindis_l = mindis_k

#     route_l_out = 1 / np.array(route_l)
#     route_r_out = 1 / np.array(route_r)
#     ts_out = np.array(ts)

#     return ts_out, route_l_out, route_r_out, np.array(ts_k1), mindis_r, mindis_l