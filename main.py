from Thymio import Thymio
import time
import numpy as np
import cv2

from ImageProcessing import aruco_read, change_perpective, image_threshold, define_grid, find_pos
from Navigation import astolfi_controller, calculate_centroid, euclidean_distance, rel_angle, vector, projected_position, calculate_state
from PathFinding import A_star
from ThymioFunctions import set_motors, local_nn
from KalmanFilter import extended_kalman	  

def main(th):
	#image processing parameters
	cap = cv2.VideoCapture(0)
	FPS = 10
	spacing = 80 #mm
	contour_thickness = 110 #mm

	#initialization parameters
	movement_start = True
	transform_start = True
	correction_start = True
	size = (1000, 1000)
	prev_goal = [size[0]**2, size[1]**2]
	prev_pos = prev_goal

	#controller parameters
	controller = astolfi_controller(k_rho=0.4, k_alpha=1.2, k_beta=-0.01)
	sampling_rate = 0.1 #s
	u = None

	#kalman filter parameters
	H = np.eye(3) #fully observable system
	Q = np.eye(3) * 0.0001
	R = np.eye(3) * 0.0001
	P = np.eye(3) #initial P

	filter = extended_kalman(Q, H, R, sampling_rate)

	while cap.isOpened():
		_, image = cap.read()

		if transform_start:
			(coords, image) = aruco_read(image, transform=True, start=True)
			if coords is not None:
				transform_start = False
			continue
		else:
			image = change_perpective(image, coords, size)

		if movement_start:
			(positions, image) = aruco_read(image, transform=False, start=True)
			if positions is not None:
				for el in positions:              
					if el.get('ID') == 4:
						robot_coords = el.get('POS')
						aruco_robot = robot_coords
					elif el.get('ID') == 5:
						goal_coords = el.get('POS')
					else:
						pass
				state = calculate_state(robot_coords)
				goal_c = calculate_centroid(goal_coords)
				movement_start = False
			continue
		else:
			(pos, image) = aruco_read(image, transform=False, start=False)
			
			robot_coords = None
			
			if pos is not None:
				for el in pos:              
					if el.get('ID') == 4:
						robot_coords = el.get('POS')
					elif el.get('ID') == 5:
						goal_coords = el.get('POS')
					else:
						pass

			goal_c = calculate_centroid(goal_coords)
			if robot_coords is not None:
				measured_state = calculate_state(robot_coords)
			if u is not None and robot_coords is not None:
				state, P = filter.correct(predicted_state, P, measured_state)
			if u is not None and robot_coords is None:
				state = predicted_state
			if correction_start:
				state = measured_state
			

			if euclidean_distance(goal_c, prev_goal) > 120 or euclidean_distance(state[:2], prev_pos) > 200:
				set_motors(th)
				thresh = image_threshold(image, contour_thickness, aruco_robot, goal_coords)
				grid, coord, background = define_grid(size, spacing, thresh)
				start = find_pos(state[:2], grid, coord)
				goal = find_pos(goal_c, grid, coord)

				pathfinder = A_star(grid, coord)
				path = pathfinder.find_path(start, goal)
				if path is not None:
					prev_pos = state[:2]
					prev_goal = goal_c
					correction_start = True
					path[0] = goal_c
					path[-1] = [int(state[0]), int(state[1])]
					iter = len(path) - 2 

			else:
				prev_pos = state[:2]
				pass
			
			if path is None:
				print('No path found')
				continue
			else:
				for i in range(len(path)-1):
					image = cv2.line(image, path[i], path[i+1], (0, 255, 0), 6)

				if correction_start:
					robot_vec = [np.cos(state[2]), np.sin(state[2])]
					goal_vec = vector(path[-1], path[iter])
					if abs(rel_angle(robot_vec, goal_vec)) > np.pi/18:
						set_motors(th, 0, -np.sign(rel_angle(robot_vec, goal_vec))*np.pi/4)
					else:
						set_motors(th)
						correction_start = False
				else:
					if euclidean_distance(state[:2], goal_c) > 30 and iter >=0:
						if projected_position(path, state[:2], iter) < 25 or euclidean_distance(state[:2], path[iter]) < 30:
							iter -= 1
						u = controller.control(state, path[iter])
						u = local_nn(th, u[0], u[1])
						predicted_state, P = filter.predict(state, P, u)

						set_motors(th, u[0], u[1])
					else:
						set_motors(th)

			image = cv2.resize(image, (500,500))
			# time.sleep(0.1)
			cv2.imshow('background', background)
			cv2.imshow('camera', image)
			cv2.waitKey(int(1000/FPS))

with Thymio.serial(port="COM9", refreshing_rate=0.1) as th:
	dir(th)

	time.sleep(1)

	variables = th.variable_description()

	for var in variables:
		print(var)

	main(th)
