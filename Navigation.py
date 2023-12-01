import numpy as np

# euclidean distance between two points without taking the square root
def euclidean_distance(pointA, pointB):
    return (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2

# returns the vector from start_p to end_p, [x, y]
def vector(start_p, end_p):
    return [ end_p[0] - start_p[0], end_p[1] - start_p[1]]

# returns the centroid of a quadrilateral. :coords is a list of 4 points
def calculate_centroid(coords):
    x_center = 0
    y_center = 0
    for i in range(4):
        x_center += coords[i][0]/4
        y_center += coords[i][1]/4
    return [x_center, y_center]

# returns the orientation vector of an object based on the first two points in coords. :coords is a list of 4 points
def calculate_orientation(coords):
    x_end = 0
    y_end = 0
    for i in range(2):
        x_end += coords[i][0]/2
        y_end += coords[i][1]/2
    start = calculate_centroid(coords)
    return vector(start, [x_end, y_end])

# Determines the relative orientation of a robot with respect to a path. It uses the sign of the dot product of vec_path and rot_90_CCW(vec_robot) (which rotates vec_robot 90 degrees counter-clockwise) to determine this. A negative result implies the robot is to the right of the path, and a positive result implies it is to the left.
#if negative robot is on the left of the path, if positive robot is on the right of the path
def rel_orientation(vec_path, vec_robot):
    return np.sign(np.dot(vec_path, rot_90_CCW(vec_robot)))

# rotates a vector 90 degrees counter-clockwise
def rot_90_CCW(vec):
    return [-vec[1], vec[0]]

# calculates the error between the robot's position and the path. :path is a list of points, :robot_coords is a list of 4 points
def calculate_error(path, robot_coords):
    robot_coord = calculate_centroid(robot_coords)
    prev_projection = -1
    for i in range(len(path)-1):
        robot_vec = vector(robot_coord, path[i])
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
    
# test = vector([3, 10], [10, 4])
# test2 = calculate_centroid([[0, 0], [0, 10], [10, 10], [10, 0]])
# test3= calculate_orientation([[0, 0], [0, 10], [10, 10], [10, 0]])
# test4 = rel_orientation([0, 4], [-2, 2])
# print(test4)