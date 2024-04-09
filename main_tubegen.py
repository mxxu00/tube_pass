import map
import matplotlib.pyplot as plt
import numpy as np
import globalvar as gv
import astar
import smoothpath
import rearranget
import tubegen
import time
import pickle

# 计算时以map为参考系 即00代表左上角

gv.obstacle_number = 20
gv.obstacle_safedist_occup = 0
gv.obstacle_safedist_astar = 10
gv.obstacle_radius = [10,150]
gv.length_start = 2 * gv.obstacle_radius[1]
gv.length = 10 * 100 + 2 * gv.length_start
gv.width = 8 * 100
gv.smooth_poly_n = 5
gv.tubegen_poly_n = 5
gv.lambda_max = 0
gv.sim_step_length = 0.01
gv.tube_r_max = 80
gv.astarmap_scale = 100 # 100cm 为一个astar的采样点

jupyter_figsize = 0.3

t1 = time.time()

fig, axes = plt.subplots()
occup_map, astar_map, myobstacle = map.init(axes) # class Obstacle

t2 = time.time()
print("map init! cost time = ", ((t2-t1)*1000))

# 统一坐标系
axes.xaxis.set_ticks_position('top')   #将X坐标轴移到上面
axes.invert_yaxis()     
axes.set_aspect(1)
more_vision = 100
axes.set(xlim=(0 - more_vision, gv.length + more_vision),ylim=(gv.width + more_vision, 0 - more_vision))
# plt.rcParams['figure.figsize'] = ((gv.length + 2)*jupyter_figsize, (gv.width + 2)*jupyter_figsize)

# A*寻路
astar_result, solution_flag = astar.start(astar_map) #逆序

astar_result.reverse()
print(astar_result)
x_coord = [coord[0] * gv.astarmap_scale for coord in astar_result] 
y_coord = [coord[1] * gv.astarmap_scale for coord in astar_result]
axes.plot(x_coord, y_coord, 'orange', linewidth = 2.0)

print(solution_flag,'Astar finished!')

# 路径平滑
sp = smoothpath.SmoothPath(np.array(astar_result))
optimized_coeffs_x, optimized_coeffs_y, filter_line_coord = sp.start()
astar_line_number = len(astar_result) - 1
print("astar_line_number:",astar_line_number,",smooth path finished")

# 重新安排时间参数t
re = rearranget.RearrangeT(optimized_coeffs_x, optimized_coeffs_y, occup_map, filter_line_coord[:,2], axes)
t1 = time.time()
fx, fx1, fx2, fy, fy1, fy2, ft = re.get_f()
t2 = time.time()
ts_mindis, tsk_mindis, route_l, route_r, mindis_l, mindis_r = re.get_mindis()
t3 = time.time()
axes.scatter(fx[tsk_mindis], fy[tsk_mindis], marker = "*", color = 'purple')
print("get_f finished! cost time = ", (t2-t1)*1000)
print("get_mindis finished! cost time = ", (t3-t2)*1000)

# save_data = [axes, ts_mindis, tsk_mindis, route_l, route_r, mindis_l, mindis_r, fx, fx1, fx2, fy, fy1, fy2, ft]
# f = open('tube_test_data_5.pkl', 'wb')
# pickle.dump(save_data, f)

# 半径生成
tg = tubegen.TubeGen(ts_mindis, route_l, route_r, mindis_l, mindis_r)
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
normal = np.dot(rot, tangent)
normal = normal / np.sqrt(np.sum(normal ** 2, axis=0))


fo1 = np.array([fx, fy]) - 1 / fr1 * normal
fo2 = np.array([fx, fy]) + 1 / fr2 * normal

# 画图

axes.plot(fo1[0,:], fo1[1,:], color = 'black', linewidth=2)
axes.plot(fo2[0,:], fo2[1,:], color = 'black', linewidth=2)
axes.plot(fx, fy, color = 'green', linewidth = 1.5)

plt.show()

save_data = [astar_map, myobstacle,  # map
             x_coord, y_coord,  # astar 
             ts_mindis, tsk_mindis, route_l, route_r, mindis_l, mindis_r, fx, fx1, fx2, fy, fy1, fy2, ft, fo1, fo2] # tube
f = open('tube_data_for_flocking_2.pkl', 'wb')
pickle.dump(save_data, f)