
import math
import heapq
import numpy as np

class Node:

    def __init__(self, coord, goal):
        self.coord = coord
        self.g = float('inf')
        self._h = self._get_dist(self.coord, goal)
        self._f = self.g + self._h
        self.parent = None

    def __eq__(self, other):
        return self.coord == other.coord

    def __lt__(self, other):
        return self._f < other._f
        
    def set_cost(self, g):
        if g < self.g:
            self.g = g
            self._f = self.g + self._h
            
    def set_parent(self, parent):
        if parent is not None:
            self.parent = parent
        
    def _get_dist(self, node_A, node_B):
        dist_x = math.fabs(node_A[0] - node_B[0])
        dist_y = math.fabs(node_A[1] - node_B[1])
        
        if dist_x > dist_y:
            dist = math.sqrt(2)*dist_y + (dist_x - dist_y)
        else:
            dist = math.sqrt(2)*dist_x + (dist_y - dist_x)
        return dist

class A_star:
    
    def __init__(self, grid, camera_coords):
        self._grid = grid
        self._camera_coords = camera_coords
    
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
        prev_movement = None
        
        while construct is not None:
            prev_point = construct.coord
            construct = construct.parent
            if construct is not None:
                if construct.coord[0] == prev_point[0]:
                    movement = 'horizontal'
                elif construct.coord[1] == prev_point[1]:
                    movement = 'vertical'
                else:
                    movement = 'diagonal'

                if prev_movement == movement:
                    path[-1] = self._camera_coords[self._grid.index(prev_point)]
                else:
                    path.append(self._camera_coords[self._grid.index(prev_point)])
                prev_movement = movement
                        
        return path
    
    def _is_traversible(self, coord):
        if coord in self._grid:
            return True
        else:
            return False
    
    def find_path(self, start, goal, movements = "8n"):
        start_node = Node(start, goal)
        start_node.set_cost(0)
        open_set = [start_node]
        closed_set = []
        heapq.heapify(open_set)
        
        while open_set:
            current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.append(current)
            if current.coord == goal:
                return self._path_constructor(current)
            
            for dx, dy, dg in self._get_movements(movements):
                coord = (current.coord[0] + dx, current.coord[1] + dy)
                if self._is_traversible(coord):
                    neighbour = Node(coord, goal)
                else:
                    continue

                if neighbour in closed_set:
                    continue
            
                if neighbour in open_set:
                    for obj in open_set:
                        if obj == neighbour:
                            del neighbour
                            neighbour = obj
                else:
                    neighbour.set_cost(current.g + dg)
                    neighbour.set_parent(current)
                    heapq.heappush(open_set, neighbour)

                if neighbour.g > current.g + dg:
                    neighbour.set_cost(current.g + dg)
                    neighbour.set_parent(current)
                    
        return None
            
        
        
        
    

                


                