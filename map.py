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
        myobstacle.append(Obstacle(random.random()*gv.width + gv.length_start,
                          random.random()*gv.width, random.uniform(gv.obstacle_radius[0], gv.obstacle_radius[1])))

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
        leastradius = math.ceil(myobstacle[i].r) + gv.obstacle_safedist
        # j <-> x <-> length    k <-> y <-> width
        # 这里要注意 坐标系的变换 计算时以左上角为(0,0) 但实际上画图时左下角为(0,0) 所以写入到mymap之前要进行映射
        for j in range(closest_x - leastradius, closest_x + leastradius):
            for k in range(closest_y - leastradius, closest_y + leastradius):
                if(j >= 0 and j <= gv.length  and k >= 0 and k <= gv.width and occup_map[j][k] != 1):
                    p1 = np.array((j, k))  # 此时还位于matplotlib坐标系
                    p2 = np.array((myobstacle[i].x, myobstacle[i].y))
                    dist = np.sqrt(np.sum((p1 - p2)**2))
                    # print(dist)
                    if(dist < myobstacle[i].r + gv.obstacle_safedist):
                        occup_map[j][k] = 1
                        # axes.scatter(j, k, color='r', marker='x')
    print('occup_map init finished')    

    # Astar地图生成
    w1 = int(gv.width / gv.astarmap_scale)
    l1 = int(gv.length / gv.astarmap_scale)
    astar_map = np.zeros((l1 + 1,w1 + 1))
    for i in range(l1):
        for j in range(w1):
            # print(i * gv.astarmap_scale, j * gv.astarmap_scale, occup_map[i * gv.astarmap_scale][j * gv.astarmap_scale])
            if(occup_map[i * gv.astarmap_scale][j * gv.astarmap_scale] == 1):
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
