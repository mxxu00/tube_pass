import matplotlib.pyplot as plt
import matplotlib.patches as pc
import random
import globalvar as gv
import math
import numpy as np

class Obstacle(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


def init(axes):

    # 障碍物生成
    myobstacle = []
    for i in range(gv.obstacle_number):
        myobstacle.append(Obstacle(random.random() * (gv.length - gv.length_start * 2) + gv.length_start,
                                    random.random() * (gv.width + 100 * 2) - 100,
                                    random.uniform(gv.obstacle_radius[0], gv.obstacle_radius[1])))

    for i in range(gv.obstacle_number):
        circ = pc.Circle((myobstacle[i].x, myobstacle[i].y), myobstacle[i].r)
        axes.add_artist(circ)

    # for i in range(gv.length+1):
    #     for j in range(gv.width+1):
    #         circ = pc.Circle((i, j), 0.01, color='r')
    #         axes.add_artist(circ)
        
    # 栅格地图生成
    occup_map = np.zeros((gv.length + 1, gv.width + 1))
    
    for i in range(gv.obstacle_number):
        closest_x = round(myobstacle[i].x)
        closest_y = round(myobstacle[i].y)
        leastradius = math.ceil(myobstacle[i].r) + gv.obstacle_safedist_occup
        # j <-> x <-> length  k <-> y <-> width
        for j in range(max(closest_x - leastradius, 0), min(closest_x + leastradius, gv.length + 1)):
            for k in range(max(closest_y - leastradius, 0), min(closest_y + leastradius, gv.width + 1)):
                if(occup_map[j][k] != 1):
                    p1 = np.array((j, k))
                    p2 = np.array((myobstacle[i].x, myobstacle[i].y))
                    dist = np.sqrt(np.sum((p1 - p2)**2))
                    if(dist < myobstacle[i].r + gv.obstacle_safedist_occup):
                        occup_map[j][k] = 1
                        # axes.scatter(j, k, color='r', marker='x')
    print('occup_map init finished')    

    # Astar地图生成
    l1 = int(gv.length / gv.astarmap_scale)
    w1 = int(gv.width / gv.astarmap_scale)
    astar_map = np.zeros((l1 + 1,w1 + 1))
    for i in range(l1 + 1): 
        for j in range(w1 + 1):
            for k in range(gv.obstacle_number):
                p1 = np.array((i * gv.astarmap_scale, j * gv.astarmap_scale))
                p2 = np.array((myobstacle[k].x, myobstacle[k].y))
                dist = np.sqrt(np.sum((p1 - p2)**2))
                if(dist < myobstacle[k].r + gv.obstacle_safedist_astar):
                    astar_map[i][j] = 1
                    axes.scatter(i * gv.astarmap_scale, j * gv.astarmap_scale, color='r', marker='x')

    print('astar_map init finished')    

    return occup_map, astar_map, myobstacle

def main():
    fig, axes = plt.subplots()
    init(axes)
    # show(axes)


if __name__ == '__main__':
    main()
