
import math
import heapq
import numpy as np


class Node:

    def __init__(self, index, goal):
        self.index = index
        self.g = float('inf')
        self._h = self._get_dist(self.index, goal)
        self._f = self.g + self._h
        self._parent = None

    def __eq__(self, other):
        return self._index == other._index

    def __lt__(self, other):
        return self._f < other._f
        
    def set_cost(self, g):
        if g < self.g:
            self.g = g
            self._f = self.g + self._h
            
    def set_parent(self, parent):
        self._parent = parent
        
    def _get_dist(self, node_A, node_B):
        dist_x = math.fabs(node_A[0] - node_B[0])
        dist_y = math.fabs(node_A[1] - node_B[1])
        
        if dist_x > dist_y:
            dist = math.sqrt(2)*dist_y + (dist_x - dist_y)
        else:
            dist = math.sqrt(2)*dist_x + (dist_x - dist_y)
        return dist

class A_star:
    
    def __init__(self, grid):
        self._grid = np.array(grid)
       
    
    def _get_movements(self, movements):
        if movements == "8n":
            s2 = math.sqrt(2)
            return [(1, 0, 1.0),
                    (0, 1, 1.0),
                    (-1, 0, 1.0),
                    (0, -1, 1.0),
                    (1, 1, s2),
                    (-1, 1, s2),
                    (-1, -1, s2),
                    (1, -1, s2)]
        else:
            return [(1, 0, 1.0),
                    (0, 1, 1.0),
                    (-1, 0, 1.0),
                    (0, -1, 1.0)]
    
    def _path_constructor(self, final_node):
        path = []
        construct = final_node
        
        while construct is not None:
            path.append(construct)
            construct = construct.parent
            
        return path
    
    def find_path(self, start, goal, movements = "8n"):
        start_node = Node(start, goal)
        open_set = [start_node]
        closed_set = []
        heapq.heapify(open_set)
        
        while open_set:
            current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.append(current)
            if current.index == goal:
                return self._path_constructor(current)
            
        for dx, dy, dg in self._get_movements(movements):
            coord = (current.index[0] + dx, current.index[1] + dy)
            if coord[0] < 0 or coord[1] < 0 or coord[0] >= len(self._grid) or coord[0] >= len(self._grid[1]):
                neighbour = Node(coord, goal, None)
            else:
                continue

            if neighbour in closed_set or self._grid[coord]:
                continue
            
            if neighbour in open_set:
                for obj in open_set:
                    if obj == neighbour:
                        del neighbour
                        neighbour = obj
            else:
                heapq.heappush(open_set, neighbour)

            if neighbour.g > current.g + dg:
                neighbour.set_cost(current.g + dg)
                neighbour.set_parent(current)
                    
        return []
            
        
        
        
    

                


                
