import map
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import globalvar as gv
import astar
import smoothpath

# 计算时以map为参考系 即00代表左上角

gv.obstacle_number = 30
gv.obstacle_safedist = 20
gv.obstacle_radius = [10,100]
gv.length_start = 2 * gv.obstacle_radius[1]
gv.length = 10 * 100 + 2 * gv.length_start
gv.width = 10 * 100
gv.smooth_poly_n = 5

jupyter_figsize = 0.3

fig, axes = plt.subplots()
occup_map, astar_map, myobstacle = map.init(axes) # class Obstacle

# 统一坐标系
axes.xaxis.set_ticks_position('top')   #将X坐标轴移到上面
axes.invert_yaxis()     
axes.set_aspect(1)
axes.set(xlim=(0, gv.length),ylim=(gv.width, 0))
# plt.rcParams['figure.figsize'] = ((gv.length + 2)*jupyter_figsize, (gv.width + 2)*jupyter_figsize)


astar_result, solution_flag = astar.start(astar_map) #逆序

astar_result.reverse()
print(astar_result)
x_coord = [coord[0] * gv.astarmap_scale for coord in astar_result] 
y_coord = [coord[1] * gv.astarmap_scale for coord in astar_result]
axes.plot(x_coord, y_coord, 'orange')

print(solution_flag,'Astar finished!')

sp = smoothpath.SmoothPath(np.array(astar_result) * gv.astarmap_scale)
optimized_coeffs, filter_line_coord = sp.start()

line_number = len(astar_result) - 1

print("line_number:",line_number,",smooth path finished")
coeffs = np.reshape(optimized_coeffs, (line_number * 2, gv.smooth_poly_n + 1))
coord = np.array(filter_line_coord)

for j in range(line_number):
    i = j * 2
    a_x = coeffs[i]
    a_y = coeffs[i + 1]
    t = np.linspace(coord[j][2], coord[j+1][2], 100)
    x = np.polyval(a_x[::-1], t)
    y = np.polyval(a_y[::-1], t) 
    # print(x,y)
    axes.plot(x, y, color = 'green', linewidth = 0.5)

plt.show() 
fig