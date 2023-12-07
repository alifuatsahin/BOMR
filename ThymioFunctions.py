from Navigation import rel_orientation
import numpy as np
import time

def set_motors(th, v=0, w=0):
	conv_factor = 450/140
	L = 95 #mm
	speedR = conv_factor*(2*v - w*L)/2
	speedL = conv_factor*(2*v + w*L)/2
	if abs(speedR) > 450 or abs(speedL) > 450:
		v = (speedR + speedL)/(2*conv_factor)
		w = (speedL - speedR)/(2*conv_factor*L)
	print(speedR, speedL)
	if speedR < 0:
		speedR = 2**16+speedR
	if speedL < 0:
		speedL = 2**16+speedL
	th.set_var("motor.left.target", int(speedL))
	th.set_var("motor.right.target", int(speedR))
	return v, w

def local_nn(th, v, w):
	wl = [-1, -2, -4, 2, 1, 0, 0]
	wr = [1, 2, 4, -2, -1, 0, 0]
	scale = 40
	L = 95 #mm
	speedL = np.dot(th["prox.horizontal"], wl)//scale
	speedR = np.dot(th["prox.horizontal"], wr)//scale

	w = w + (speedR - speedL)/(2*L)
	v = v + (speedR + speedL)/2

	return v, w