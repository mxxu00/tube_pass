import map
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import globalvar as gv
import astar
import smoothpath
import rearranget

# 计算时以map为参考系 即00代表左上角

gv.obstacle_number = 30
gv.obstacle_safedist_occup = 0
gv.obstacle_safedist_astar = 40
gv.obstacle_radius = [10,100]
gv.length_start = 2 * gv.obstacle_radius[1]
gv.length = 10 * 100 + 2 * gv.length_start
gv.width = 10 * 100
gv.smooth_poly_n = 5
gv.tubegen_poly_n = 5
gv.lambda_max = 0
gv.sim_step_length = 0.1
gv.tube_r_max = 100
gv.astarmap_scale = 100 # 100cm 为一个astar的采样点

jupyter_figsize = 0.3

fig, axes = plt.subplots()
occup_map, astar_map, myobstacle = map.init(axes) # class Obstacle

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
sp = smoothpath.SmoothPath(np.array(astar_result) * gv.astarmap_scale)
optimized_coeffs_x, optimized_coeffs_y, filter_line_coord = sp.start()
line_number = len(astar_result) - 1
print("line_number:",line_number,",smooth path finished")

# 重新安排时间参数t
re = rearranget.rearranget()
re.start(optimized_coeffs_x, optimized_coeffs_y, occup_map, filter_line_coord[:,2], axes)

axes.scatter(re.fx[re.ts_k1], re.fy[re.ts_k1], marker = "*", color = 'purple')

coeffs_x = np.reshape(optimized_coeffs_x, (line_number, gv.smooth_poly_n + 1))
coeffs_y = np.reshape(optimized_coeffs_y, (line_number, gv.smooth_poly_n + 1))
coord = np.array(filter_line_coord)



for i in range(line_number):
    a_x = coeffs_x[i]
    a_y = coeffs_y[i]
    t = np.linspace(coord[i][2], coord[i+1][2], 100)
    x = np.polyval(a_x[::-1], t)
    y = np.polyval(a_y[::-1], t) 
    # print(x,y)
    axes.plot(x, y, color = 'green', linewidth = 1.5)

plt.show() 
fig