import numpy as np

def euclidean_distance(pointA, pointB):
    return (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2

def vector(start_p, end_p):
    return [end_p[1] - start_p[1], end_p[0] - start_p[0]]

def calculate_centroid(coords):
    x_center = 0
    y_center = 0
    for i in range(4):
        x_center += coords[i][0]/4
        y_center += coords[i][1]/4
    return [x_center, y_center]

def calculate_orientation(coords):
    x_end = 0
    y_end = 0
    for i in range(2):
        x_end += coords[i][0]/4
        y_end += coords[i][0]/4
    start = calculate_centroid(coords)
    return vector(start, [x_end, y_end])

def rel_orientation(vec_path, vec_robot): #if negative robot is on the right of the path, if positive robot is on the left of the path
    return np.sign(np.dot(vec_path, rot_90_CCW(vec_robot)))

def rot_90_CCW(vec):
    return [-vec[1], vec[0]]

def calculate_error(path, robot_coords):
    robot_coord = calculate_centroid(robot_coords)
    prev_projection = -1
    for i in range(len(path-1)):
        robot_vec = vector(path[i], robot_coord)
        path_vec = vector(path[i+1], robot_coord)
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