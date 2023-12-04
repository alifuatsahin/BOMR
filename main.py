from Thymio import Thymio
import time
import numpy as np
import cv2

from ImageProcessing import aruco_read, change_perpective, image_threshold, define_grid, find_pos
from Navigation import rel_orientation, calculate_centroid, astolfi_controller, euclidean_distance, rel_angle, vector, calculate_orientation, projected_position
from PathFinding import A_star

def set_motors(th, speedR=0, speedL=0):
	conv_factor = 500/140
	speedR = conv_factor*speedR
	speedL = conv_factor*speedL
	if speedR > 250:
		speedR = 250
	if speedL > 250:
		speedL = 250
	if speedR < 0:
		if speedR < -250:
			speedR = -250
		speedR = 2**16+speedR
	if speedL < 0:
		if speedL < -250:
			speedL = -250
		speedL = 2**16+speedL
	th.set_var("motor.left.target", int(speedL))
	th.set_var("motor.right.target", int(speedR))

def local_navigation(th):
	wl = [3, 2, 1, -2, -3, 1, -1]
	wr = [-3, -2, -1, 2, 3, -1, 1]
	scale = 200
	prev_time = 0

	try:
		while True:
			if time.time_ns() - prev_time > 10**8:
				nominal = 50
				speedL = nominal + np.dot(th["prox.horizontal"], wl)//scale
				speedR = nominal + np.dot(th["prox.horizontal"], wr)//scale
				print(speedR, speedL)
				if speedR < 0:
					speedR = 2**16+speedR
				if speedL < 0:
					speedL = 2**16+speedL
				set_motors(th, speedR, speedL)
				prev_time = time.time_ns()
			if th.get_var('button.center'):
				break
	except ValueError:
		set_motors(th)

	set_motors(th)

def correct_orientation(th, robot_vec, goal_vec, speed):
	speedR = rel_orientation(robot_vec, goal_vec)*speed
	speedL = -speedR
	print(speedR, speedL)
	set_motors(th, int(speedR), int(speedL))
	  

def test(th):
	cap = cv2.VideoCapture(0)
	FPS = 10
	coords = None
	pos = None
	start = True
	size = (800, 800)
	prev_goal = [size[0]**2, size[1]**2]
	c_scale = 1
	controller = astolfi_controller(k_rho=c_scale*8, k_alpha=c_scale*45, k_beta=c_scale*-12)
	spacing = 80 #mm
	L = 100 #mm
	R = 40 #mm

	while cap.isOpened():
		ret, image = cap.read()

		if coords is None:
			(coords, image) = aruco_read(image, transform=True, start=True)
			continue
		else:
			(temp_coords, temp_im) = aruco_read(image, transform=True, start=False)
			if temp_coords is not None:
				coords = temp_coords
				image = temp_im
			image = change_perpective(image, coords, size)

		if pos is None:
			(pos, image) = aruco_read(image, transform=False, start=True)
			if pos is not None:
				for el in pos:              
					if el.get('ID') == 4:
						robot_coords = el.get('POS')
						pos_init = calculate_centroid(robot_coords)
					elif el.get('ID') == 5:
						goal_coords = el.get('POS')
					else:
						pass
			continue
		else:
			(temp_pos, temp_image) = aruco_read(image, transform=False, start=False)
			if temp_pos is not None:
				pos = temp_pos
				image = temp_image
			
			for el in pos:              
				if el.get('ID') == 4:
					robot_coords = el.get('POS')
				elif el.get('ID') == 5:
					goal_coords = el.get('POS')
				else:
					pass

			robot_c = calculate_centroid(robot_coords)
			goal_c = calculate_centroid(goal_coords)

			if euclidean_distance(goal_c, prev_goal) > 120:
				set_motors(th)
				thresh = image_threshold(image, 80, robot_coords, goal_coords)
				grid, coord, background = define_grid(size, spacing, thresh)
				start = find_pos(robot_c, grid, coord)
				goal = find_pos(goal_c, grid, coord)

				pathfinder = A_star(grid, coord)
				path = pathfinder.find_path(start, goal)
				if path is not None:
					start = True
					prev_goal = goal_c
					iter = len(path) - 2 

			else:
				pass
			
			if path is None:
				print('No path found')
				continue
			else:
				for i in range(len(path)-1):
					image = cv2.line(image, path[i], path[i+1], (0, 255, 0), 6)
				image = cv2.line(image, pos_init, path[len(path)-1], (0, 255, 0), 6)
				image = cv2.line(image, path[0], goal_c, (0, 255, 0), 6)

				if start:
					robot_vec = calculate_orientation(robot_coords)
					goal_vec = vector(pos_init, path[iter])
					if abs(rel_angle(robot_vec, goal_vec)) > np.pi/18:
						correct_orientation(th, robot_vec, goal_vec, 50)
					else:
						set_motors(th)
						start = False
				else:
					if euclidean_distance(robot_c, goal_c) > 60:
						if projected_position(path, robot_c, iter) < 25 or euclidean_distance(robot_c, path[iter]) < 35:
							iter -= 1
						if iter > 0:
							(v, w) = controller.control(robot_coords, path[iter])
						else:
							(v, w) = controller.control(robot_coords, path[iter])

						speedR = (2*v - w*L)/(2*R)
						speedL = (2*v + w*L)/(2*R)
						set_motors(th, speedR, speedL)
					else:
						set_motors(th)

			image = cv2.resize(image, (500,500))
			cv2.imshow('background', background)
			cv2.imshow('camera', image)
			cv2.waitKey(int(1000/FPS))

with Thymio.serial(port="COM9", refreshing_rate=0.1) as th:
	dir(th)

	time.sleep(1)

	variables = th.variable_description()

	for var in variables:
		print(var)

	test(th)
