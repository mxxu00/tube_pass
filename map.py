import matplotlib.pyplot as plt
import matplotlib.patches as pc
import random
import globalvar as gv


class Obstacle(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


def generate(axes):
    # confirm
    print(gv.length, gv.width, gv.obstacle_number)

    myobstacle = []
    for i in range(gv.obstacle_number):
        myobstacle.append(Obstacle(random.random()*gv.width + gv.length_start,
                          random.random()*gv.width, random.uniform(gv.obstacle_radius[0], gv.obstacle_radius[1])))

    for i in range(gv.obstacle_number):
        circ = pc.Circle((myobstacle[i].x, myobstacle[i].y), myobstacle[i].r)
        axes.add_artist(circ)

    for i in range(gv.length+1):
        for j in range(gv.width+1):
            circ = pc.Circle((i, j), 0.01, color='r')
            axes.add_artist(circ)
    return myobstacle


def main():
    fig, axes = plt.subplots()
    generate(axes)
    # show(axes)


if __name__ == '__main__':
    main()
