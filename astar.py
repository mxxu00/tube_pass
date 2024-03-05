import numpy as np
import globalvar as gv
import math
import astarclass

def getKeyforSort(element: astarclass.Node):
    return element.g  # element#不应该+element.h，否则会穿墙

def astarmapinit(mymap, myobstacle,axes):
    for i in range(gv.obstacle_number):
        closest_x = round(myobstacle[i].x)
        closest_y = round(myobstacle[i].y)
        leastradius = math.ceil(myobstacle[i].r) + 1
        # print(closest_x,closest_y)

        # j <-> x <-> length    k <-> y <-> width
        # 这里要注意 坐标系的变换 计算时以左上角为(0,0) 但实际上画图时左下角为(0,0) 所以写入到mymap之前要进行映射
        for j in range(closest_x - leastradius, closest_x + leastradius + 1):
            for k in range(closest_y - leastradius, closest_y + leastradius + 1):
                if(j >= 0 and j <= gv.length and k >= 0 and k <= gv.width):
                    p1 = np.array((k, j))  # 此时还位于matplotlib坐标系
                    p2 = np.array((myobstacle[i].y, myobstacle[i].x))
                    dist = np.sqrt(np.sum((p1 - p2)**2))
                    # print(dist)
                    if(dist < myobstacle[i].r + gv.obstacle_safedist):
                        # 写入时y轴需要变换 y轴对应的是mymap的第一个参量
                        mymap[gv.width - k][j] = 1
                        axes.scatter(j, k, color='r', marker='x')
                        
    return mymap


def start(mymap):

    startx, starty = mymap.startx, mymap.starty
    endx, endy = mymap.endx, mymap.endy
    startNode = astarclass.Node(startx, starty, 0, 0, None)
    # openList和closeList应该只存坐标
    openList = []
    closeList = []
    solution_flag = 1
    closeList.append(startNode.coord)
    openList.append(startNode)
    while(1):
        currNode = openList.pop(0)
        closeList.append(currNode.coord)
        workList = currNode.getNeighbor(mymap, closeList) # 提取列表的同时 改变F值
        openList.sort(key=getKeyforSort)

        # for i in workList:
        #     print(i.coord) 

        for i in workList:
            if (i.coord not in closeList):
                if(i.hasNode(openList)): # 当前节点在 openList 里面
                    i.changeG(openList)
                else:
                    openList.append(i)
                    
        
        if(endx, endy) == (currNode.coord[0], currNode.coord[1]):
            break
        elif(len(openList) == 0):
            solution_flag = 0
            break
        
        # print('calculating...')
        

    result = []
    if(solution_flag == 0):
        return result,solution_flag
    else:
        while(currNode.father != None):
            result.append((currNode.coord[0], currNode.coord[1]))
            currNode = currNode.father
        result.append((currNode.coord[0], currNode.coord[1]))
        return result,solution_flag

def main():
    print('test')


if __name__ == '__main__':
    main()
