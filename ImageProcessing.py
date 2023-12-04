import cv2
import matplotlib.pyplot as plt
import numpy as np

from PathFinding import A_star
from Navigation import euclidean_distance

def aruco_read(image, transform, start):
	ARUCO_DICT = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
		"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
		"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
		"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
		"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
		"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
		"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
		"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
		"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
		"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
		"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
		"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
		"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
		"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
		"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
		"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
		"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
	}

	aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])
	aruco_params = cv2.aruco.DetectorParameters()
	aruco_det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

	(corners, ids, rejected) = aruco_det.detectMarkers(image)
	if ids is None:
		return None, image
	elif transform and 0 in ids and 1 in ids and 2 in ids and 3 in ids:
		pass
	elif not transform and start and 4 in ids and 5 in ids:
		pass
	elif not transform and not start and (4 in ids or 5 in ids):
		pass
	else:
		return None, image

	coord_list = []
	ids = ids.flatten()

	for (marker_corners, marker_ids) in zip(corners, ids):
		corners = marker_corners.reshape((4,2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		coord_list.append({'ID': marker_ids, 'POS': np.squeeze(corners)})			
	
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))
		
		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
		
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		
		cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
		# cv2.putText(image, str(marker_ids),
		# 	(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
		# 	0.5, (0, 255, 0), 4)
		# print("[INFO] ArUco marker ID: {}".format(marker_ids))
			
	return coord_list, image

def calculate_centroid(coords):
    x_center = 0
    y_center = 0
    for i in range(4):
        x_center += coords[i][0]/4
        y_center += coords[i][1]/4
    return [x_center, y_center]

def define_grid(size, spacing, thresh):
	background = np.zeros([size[0], size[1], 1], dtype=np.uint8)

	vert = size[0]//spacing
	horz = size[1]//spacing
	v_blank = (size[0] % spacing)//2 + spacing
	h_blank = (size[1] % spacing)//2 + spacing

	coord_init = (v_blank, h_blank)
	grid = []
	coord = []

	for h in range(horz-1):
		coord_init = (h_blank + h*spacing, v_blank)
		for v in range(vert-1):
			if thresh.T[coord_init] == 255:
				background = cv2.circle(background, coord_init, 5, 255, -1)
				grid.append((h,v))
				coord.append(coord_init)
				# cv2.putText(image, str(grid[-1]),
				# 		(coord[-1][0], coord[-1][1]), cv2.FONT_HERSHEY_SIMPLEX,
				# 		0.5, (0, 255, 0), 4, cv2.LINE_AA)
			coord_init = (coord_init[0], coord_init[1] + spacing)
			
	return grid, coord, background

def change_perpective(image, coords, size):
	transform_coords = np.zeros((4,2))
	for el in coords:
		if el.get('ID') == 0:
			transform_coords[0] = el.get('POS')[2]
		elif el.get('ID') == 1:
			transform_coords[1] = el.get('POS')[3]
		elif el.get('ID') == 2:
			transform_coords[2] = el.get('POS')[1]
		elif el.get('ID') == 3:
			transform_coords[3] = el.get('POS')[0]
		else:
			pass

	corners = np.float32(transform_coords)
	dim = np.float32([[0,0],[size[0],0],[0,size[1]],[size[0],size[1]]])

	perspective = cv2.getPerspectiveTransform(corners, dim)
	image = cv2.warpPerspective(image, perspective, (size[0], size[1]))
	
	return image

def image_threshold(image, border_size, robot_coords, goal_coords):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	thresh = delete_aruco(thresh, robot_coords)
	thresh = delete_aruco(thresh, goal_coords)

	kernel = np.ones((5, 5), np.uint8) 
	thresh = cv2.dilate(thresh, kernel, iterations=2)

	contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(thresh, contours, -1, 0, border_size)
	
	return thresh

def delete_aruco(thresh, pos):
	max_p = [0, 0]
	min_p = [float('inf'), float('inf')]
	for point in pos:
		if point[0] > max_p[0]:
			max_p[0] = int(point[0])
		if point[0] < min_p[0]:
			min_p[0] = int(point[0])
		if point[1] > max_p[1]:
			max_p[1] = int(point[1])
		if point[1] < min_p[1]:
			min_p[1] = int(point[1])
	thresh = cv2.rectangle(thresh, min_p, max_p, 255, -1)
	return thresh

def find_pos(pos, grid, coord):
	grid_c = None
	dist = float('inf')
	for point in coord:
		temp = euclidean_distance(point, pos)
		if temp < dist:
			grid_c = grid[coord.index(point)]
			dist = temp

	return grid_c

# def test():
# 	cap = cv2.VideoCapture(0)
# 	FPS = 10
# 	coords = None
# 	pos = None
# 	size = (800, 800)

# 	while cap.isOpened():
# 		ret, image = cap.read()

# 		if coords is None:
# 			(coords, image) = aruco_read(image, True)
# 			continue
# 		else:
# 			(temp_coords, temp_im) = aruco_read(image, True)
# 			if temp_coords is not None:
# 				coords = temp_coords
# 				image = temp_im
# 			image = change_perpective(image, coords, size)

# 		if pos is None:
# 			(pos, image) = aruco_read(image, False)
# 			continue
# 		else:
# 			(temp_pos, temp_image) = aruco_read(image, False)
# 			if temp_pos is not None:
# 				pos = temp_pos
# 				image = temp_image

# 			thresh = image_threshold(image, 70, pos)
# 			grid, coord, background = define_grid(size, 60, thresh)

# 			(start, goal) = find_pos(pos, grid, coord)

# 			pathfinder = A_star(grid, coord)

# 			path = pathfinder.find_path(start, goal)

# 		for i in range(len(path)-1):
# 			image = cv2.line(image, path[i], path[i+1], (0, 255, 0), 4)

# 		cv2.resize(background, (500,500))
# 		# cv2.imshow('image', image)
# 		cv2.imshow('grid', background)

# 		cv2.waitKey(int(1000/FPS))

# test()
# size = (500, 500)

# image = cv2.imread('env.jpg')

# (temp_coords, temp_im) = aruco_read(image, True, True)
# if temp_coords is not None:
# 	coords = temp_coords
# 	image = temp_im
# image = change_perpective(image, coords, size)

# (pos, image) = aruco_read(image, False, False)
# thresh = image_threshold(image, 50, pos)

# cv2.imshow('test', thresh)
# cv2.waitKey(0)