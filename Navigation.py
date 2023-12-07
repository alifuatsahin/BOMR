import numpy as np
import math

def euclidean_distance(pointA, pointB):
    return math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)

def vector(start_p, end_p):
    return [end_p[0] - start_p[0], end_p[1] - start_p[1]]

def calculate_centroid(coords):
    x_center = 0
    y_center = 0
    for i in range(4):
        x_center += coords[i][0]/4
        y_center += coords[i][1]/4
    return [int(x_center), int(y_center)]

def calculate_orientation(coords):
    x_end = 0
    y_end = 0
    for i in range(2):
        x_end += coords[i][0]/2
        y_end += coords[i][1]/2
    start = calculate_centroid(coords)
    return vector(start, [x_end, y_end])

def calculate_state(coords):
    x = [1, 0] #angle reference
    pos = calculate_centroid(coords)
    orientation = calculate_orientation(coords)
    theta = -rel_angle(orientation, x)
    return [pos[0], pos[1], theta]

def rel_orientation(vec_path, vec_robot): #if negative robot is on the right of the path, if positive robot is on the left of the path
    return np.sign(np.dot(vec_path, rot_90_CCW(vec_robot)))

def rel_angle(vec_A, vec_B):
    cos = np.dot(vec_A, vec_B)/(np.linalg.norm(vec_A)*np.linalg.norm(vec_B))
    sin = np.cross(vec_A, vec_B)/(np.linalg.norm(vec_A)*np.linalg.norm(vec_B))
    return math.atan2(sin, cos)

def rot_90_CCW(vec):
    return [-vec[1], vec[0]]

def projected_position(path, robot_c, iter):
    robot_vec = vector(path[iter+1], robot_c)
    path_vec = vector(path[iter+1], path[iter])
    projection = np.dot(robot_vec, path_vec)
    return np.linalg.norm(path_vec) - projection

def calculate_error(path, robot_coords):
    robot_coord = calculate_centroid(robot_coords)
    prev_projection = -1
    for i in range(len(path)-1):
        robot_vec = vector(path[i], robot_coord)
        path_vec = vector(path[i], path[i+1])
        projection = np.dot(robot_vec, path_vec)
        if prev_projection > 0 and projection < 0:
            sign = rel_orientation(path_vec, robot_vec)
            return sign*euclidean_distance(robot_coord, np.add(robot_vec + path[i]))
        elif np.dot(robot_vec, path_vec) < euclidean_distance(path[i], path[i+1]):
            sign = rel_orientation(path_vec, robot_vec)
            return sign*euclidean_distance(robot_coord, np.add(robot_vec + path[i]))
        else:
            prev_projection = projection

class PD_controller:
    def __init__(self, P, D, time_step):
        self._P = P
        self._D = D/time_step

    def control(self, prev_error, error):
        return self._P*error + self._D*(error - prev_error)
    
class astolfi_controller:
    def __init__(self, k_rho, k_alpha, k_beta):
        self._k_rho = k_rho
        self._k_alpha = k_alpha
        self._k_beta = k_beta
    
    def _calculate_parameters(self, state, goal):
        robot_orientation = [np.cos(state[2]), np.sin(state[2])]
        theta = state[2]
        rho = euclidean_distance(state[:2], goal)
        alpha = rel_angle(vector(state[:2], goal), robot_orientation)
        beta = -theta - alpha
        return rho, alpha, beta

    def control(self, state, goal):
        rho, alpha, beta = self._calculate_parameters(state, goal)
        return self._k_rho*rho, self._k_alpha*alpha + self._k_beta*beta
    

# x = [1, 0]
# y = [1, 1]

# print(rel_angle(x,y))

# k = [2, -3]
# robot_orientation = [1, math.sqrt(3)]

# goal = [0,0]
# robot_c = [-5, -5]

# theta = rel_angle(x, robot_orientation)
# rho = euclidean_distance(robot_c, goal)
# alpha = -theta + math.atan2(np.cross(robot_to_goal, y), np.dot(robot_to_goal, x))
# beta = -theta - alpha
# print(theta*180/np.pi, alpha*180/np.pi, beta*180/np.pi)

# print(np.dot(k, x))