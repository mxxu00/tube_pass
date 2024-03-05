import map
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import globalvar as gv
import astar
import astarclass
import smoothpath

gv.obstacle_number = 8
gv.obstacle_safedist = 0.5
gv.obstacle_radius = [0.5,3]
gv.length_start = 2 * gv.obstacle_radius[1]
gv.length = 10 + 2 * gv.length_start
gv.width = 10
gv.smooth_poly_n = 3

jupyter_figsize = 0.3

fig, axes = plt.subplots()
myobstacle = map.generate(axes) # class Obstacle
mymap = np.zeros((gv.width + 1, gv.length + 1))


mymap = astar.astarmapinit(mymap, myobstacle,axes)
#print(mymap)
axes.set_aspect(1)
axes.set(xlim=(-1, gv.length + 1),ylim=(-1, gv.width + 1))
plt.rcParams['figure.figsize'] = ((gv.length + 2)*jupyter_figsize, (gv.width + 2)*jupyter_figsize)


# plt.show()

astarmap = astarclass.AstarMap(mymap,0,0,gv.width,gv.length)
result, solution_flag = astar.start(astarmap) #逆序


result.reverse()
result_plot = [(gv.width - x, y) for x,y in result]
print(result_plot)
x_coord = [coord[0] for coord in result_plot]
y_coord = [coord[1] for coord in result_plot]
axes.plot(y_coord,x_coord,'orange')
# figure setup
# plt.show()
# fig
print('finished!')

optimized_coeffs, filter_line_coord = smoothpath.start(result)

print(optimized_coeffs)

line_number = len(result) - 1
coeffs = np.reshape(optimized_coeffs, (line_number * 2, gv.smooth_poly_n + 1))
coord = np.array(filter_line_coord)

for j in range(line_number):
    i = j * 2
    a_x = coeffs[i]
    a_y = coeffs[i + 1]
    t = np.linspace(coord[j][2], coord[j+1][2], 100)
    x = np.polyval(a_x, t)
    y = np.polyval(a_y, t) 
    # print(x,y)
    axes.plot(y, gv.width - x, color = 'green', linewidth = 0.5)

plt.show() 
fig