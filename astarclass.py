class Node(object):
    
    # 初始化节点信息
    def __init__(self, x, y, g, h, father):
        self.coord = (x,y)
        self.g = g
        self.h = h
        self.father = father

    # 处理边界和障碍点
    def getNeighbor(self, mymap, closeList):
        x = self.coord[0]
        y = self.coord[1]
        # print(self.coord,x,y)
        result = []
        # 上
        if x != 0:
            if mymap.data[x-1][y] != 1:
                g_gain = 10
            else:
                g_gain = 200
            upNode = Node(x-1, y, self.g + g_gain,
                        (abs(x-1-mymap.endx) + abs(y-mymap.endy)) * 10, self)
            result.append(upNode)

        # 下
        if x != len(mymap.data) - 1:
            if mymap.data[x + 1][y] != 1:
                g_gain = 10
            else:
                g_gain = 200
            downNode = Node(x + 1, y, self.g + g_gain,
                            (abs(x + 1 - mymap.endx) + abs(y - mymap.endy)) * 10, self)
            result.append(downNode)
        # 左
        if y != 0:
            if mymap.data[x][y-1] != 1:
                g_gain = 10
            else:
                g_gain = 200
            leftNode = Node(x, y-1, self.g + g_gain,
                            (abs(x-mymap.endx) + abs(y-1-mymap.endy)) * 10, self)
            result.append(leftNode)

        # 右
        if y != len(mymap.data[0]) - 1:
            if mymap.data[x][y+1] != 1:
                g_gain = 10
            else:
                g_gain = 200
            rightNode = Node(x, y+1, self.g + g_gain,
                            (abs(x-mymap.endx) + abs(y+1-mymap.endy)) * 10, self)
            result.append(rightNode)

        # 西北 14
        if x != 0 and y != 0:
            if mymap.data[x-1][y-1] != 1:
                g_gain = 14
            else:
                g_gain = 200
            wnNode = Node(x-1, y-1, self.g + g_gain,
                        (abs(x-1-mymap.endx) + abs(y-1-mymap.endy)) * 10, self)
            result.append(wnNode)

        # 东北
        if x != 0 and y != len(mymap.data[0]) - 1:
            if mymap.data[x-1][y+1] != 1:
                g_gain = 14
            else:
                g_gain = 200
            enNode = Node(x-1, y+1, self.g + g_gain,
                        (abs(x-1-mymap.endx) + abs(y+1-mymap.endy)) * 10, self)
            result.append(enNode)

        # 西南
        if x != len(mymap.data) - 1 and y != 0:
            if mymap.data[x+1][y-1] != 1:
                g_gain = 14
            else:
                g_gain = 200
            wsNode = Node(x+1, y-1, self.g + g_gain,
                        (abs(x+1-mymap.endx) + abs(y-1-mymap.endy)) * 10, self)
            result.append(wsNode)

        # 东南
        if x != len(mymap.data) - 1 and y != len(mymap.data[0]) - 1:
            if mymap.data[x+1][y+1] != 1:
                g_gain = 14
            else:
                g_gain = 200
            esNode = Node(x+1, y+1, self.g + g_gain,
                        (abs(x+1-mymap.endx) + abs(y+1-mymap.endy)) * 10, self)
            result.append(esNode)
        #如果节点在关闭节点 则不进行处理
        # finaResult = []
        # for i in result:
        #     if(i not in lockList):
        #         finaResult.append(i)
        # result = finaResult
        
        return result

    def hasNode(self, worklist):
        for i in worklist:
            if(i.coord[0] == self.coord[0] and i.coord[1] == self.coord[1]):
                return True
        return False
    # 在存在的前提下

    def changeG(self, worklist):
        for i in worklist:
            if(i.coord[0] == self.coord[0] and i.coord[1] == self.coord[1]):
                if(i.g > self.g):
                    i.g = self.g


class AstarMap(object):
    def __init__(self, data, startx, starty, endx, endy):
        self.data = data
        self.startx = startx
        self.starty = starty
        self.endx = endx
        self.endy = endy
