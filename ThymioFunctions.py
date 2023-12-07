from Navigation import rel_orientation
import numpy as np
import time

def set_motors(th, v=0, w=0):
	conv_factor = 450/140
	L = 95 #mm
	speedR = conv_factor*(2*v + w*L)/2
	speedL = conv_factor*(2*v - w*L)/2
	# if speedR > 250:
	# 	speedR = 250
	# if speedL > 250:
	# 	speedL = 250
	if speedR < 0:
		# if speedR < -250:
		# 	speedR = -250
		speedR = 2**16+speedR
	if speedL < 0:
		# if speedL < -250:
		# 	speedL = -250
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

def local_nn(th, v, w):
	wl = [1, 2, 3, -2, -1, 1, -1]
	wr = [-1, -2, -3, 2, 1, -1, 1]
	scale = 40
	L = 95 #mm
	speedL = np.dot(th["prox.horizontal"], wl)//scale
	speedR = np.dot(th["prox.horizontal"], wr)//scale

	w = w + (speedR - speedL)/(2*L)

	return v, w