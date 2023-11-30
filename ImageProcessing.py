import cv2
import matplotlib.pyplot as plt
import numpy as np

from PathFinding import A_star

def aruco_read(image, show_image):
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
	if np.all(ids) is None:
		return None, None

	if len(corners) > 0:
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
		
			if show_image:
				cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
				cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
				cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
				cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
				
				cX = int((topLeft[0] + bottomRight[0]) / 2.0)
				cY = int((topLeft[1] + bottomRight[1]) / 2.0)
				
				cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
				cv2.putText(image, str(marker_ids),
					(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (0, 255, 0), 4)
				print("[INFO] ArUco marker ID: {}".format(marker_ids))
			
		return coord_list, image

def calculate_centroids(coords):
	centroids = []
	for coord in coords:
		x_center = 0
		y_center = 0
		for i in range(4):
			x_center += coord['POS'][i][0]/4
			y_center += coord['POS'][i][1]/4
		centroids.append([x_center, y_center])
	return centroids

def define_grid(size, spacing, thresh):
	background = np.zeros([size[0], size[1], 1], dtype=np.uint8)

	vert = size[0]//spacing
	horz = size[1]//spacing
	v_blank = (size[0] % spacing)//2 + vert
	h_blank = (size[1] % spacing)//2 + horz

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

def image_threshold(image, border_size):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	kernel =  np.ones((5, 5), np.uint8) 
	thresh = cv2.dilate(thresh, kernel, iterations=2)

	contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(thresh, contours, -1, 0, border_size)
	
	return thresh

# im = cv2.imread('env_final.jpg')
cap = cv2.VideoCapture(0)
FPS = 10

while cap.isOpened():
	ret, image = cap.read()
	size = (1000, 1000)

	(coords, image) = aruco_read(image, True)
	if coords is None:
		continue

	centroids = calculate_centroids(coords)

	image = change_perpective(image, coords, size)
	thresh = image_threshold(image, 50)

	grid, coord, background = define_grid(size, 50, thresh)

	pathfinder = A_star(grid)

	path = pathfinder.find_path((0,0), (20,20))

	for i in range(len(path)-1):
		begin = grid.index(path[i].index)
		end = grid.index(path[i+1].index)
		image = cv2.line(image, coord[begin], coord[end], (0, 255, 0), 4)

	cv2.imshow('image', image)
	#cv2.imshow('grid', background)

	cv2.waitKey(int(1000/FPS))
