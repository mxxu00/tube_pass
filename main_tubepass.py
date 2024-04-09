import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as pc
import numpy as np
import globalvar as gv
import pickle
import flocking

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

uav_number = 20
iteration = 1000
scale_coeff = 3
dt = 0.05
pos_range = 50

f = open('tube_data_for_flocking_1.pkl  ', 'rb')
save_data = pickle.load(f)
[astar_map, myobstacle,  # map
 x_coord, y_coord,  # astar 
 ts_mindis, tsk_mindis, route_l, route_r, mindis_l, mindis_r, fx, fx1, fx2, fy, fy1, fy2, ft, fo1, fo2] = save_data 

# floking

target_array = np.array((fx[tsk_mindis], fy[tsk_mindis])).T
obstacle_set = np.concatenate((fo1, fo2), axis = 1).T

flk = flocking.Boids(uav_number, dt, target_array, obstacle_set, scale_coeff)
flk.init(pos_range)

step_set = np.zeros((uav_number, iteration, 2))
velocity_set = np.zeros((uav_number, iteration, 2))

for t in range(iteration):        
    flk.evolve()
    flk.agents_c = flk.agents_c_next

    step_set[:,t,:] = flk.agents_c
    velocity_set[:,t,:] = flk.agents_v

# plt init
fig, axes = plt.subplots(figsize = (14,8))

axes.xaxis.set_ticks_position('top')   #将X坐标轴移到上面
axes.invert_yaxis()     
axes.set_aspect(1)
more_vision = 100
axes.set(xlim=(0 - more_vision, gv.length + more_vision), ylim=(gv.width + more_vision, 0 - more_vision))

# map plt
axes.plot(x_coord, y_coord, 'orange', linewidth = 2.0) # astar
axes.plot(fo1[0,:], fo1[1,:], color = 'black', linewidth=2)
axes.plot(fo2[0,:], fo2[1,:], color = 'black', linewidth=2)
axes.plot(fx, fy, color = 'green', linewidth = 1.5)
axes.scatter(fx[tsk_mindis], fy[tsk_mindis], marker = "*", color = 'purple')
for i in range(gv.obstacle_number):
    circ = pc.Circle((myobstacle[i].x, myobstacle[i].y), myobstacle[i].r)
    axes.add_artist(circ)
for i in range(np.size(astar_map, axis = 0)):
    for j in range(np.size(astar_map, axis = 1)):
        if(astar_map[i][j] == 1):
            axes.scatter(i * gv.astarmap_scale, j * gv.astarmap_scale, color='r', marker='x')
# ani init
scat = axes.scatter(step_set[:, 0, 0], step_set[:, 0, 1], c="b")

def ani_update(frame):
    # uav trajectory
    scat.set_offsets(np.array((step_set[:, frame, 0], step_set[:, frame, 1])).T)
    return scat,

fps = int(1/dt)
ani = animation.FuncAnimation(fig=fig, func = ani_update, frames = iteration, interval = fps)
plt.show()
ani.save(filename="tube_pass_1.mp4", fps = fps, writer="ffmpeg")