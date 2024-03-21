import numpy as np
import globalvar as gv
import math
import astarclass

def getKeyforSort(element: astarclass.Node):
    return element.g  # element#不应该+element.h，否则会穿墙

def start(astar_map):
    mymap = astarclass.Map(astar_map, 0, 0, int(gv.length / gv.astarmap_scale), int(gv.width / gv.astarmap_scale))

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
