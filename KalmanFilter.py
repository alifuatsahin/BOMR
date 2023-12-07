import numpy as np

class extended_kalman:
	def __init__(self, Q, H, R, sampling_rate):
		self._Q = Q
		self._H = H
		self._R = R
		self._sampling_rate = sampling_rate        

	def predict(self, x_init, P_init, u):
		F = np.eye(3) #+ np.array([[0, 0, -u[0]*np.sin(x_init[2])], [0, 0, u[0]*np.cos(x_init[2])], [0, 0, 0]])*self._sampling_rate
		G = np.array([[np.cos(x_init[2]), 0], [np.sin(x_init[2]), 0], [0, 1]])*self._sampling_rate
		x_est = np.dot(F, x_init) + np.dot(G, u)
		P_est = np.dot(np.dot(F, P_init), F.T) + self._Q
		
		return x_est, P_est
	
	def correct(self, x_est, P_est, y):
		i = y - np.dot(self._H, x_est)
		j = np.eye(len(self._H))
		S = np.dot(self._H, np.dot(P_est, self._H.T)) + self._R
		Kn = np.dot(np.dot(P_est, self._H.T), np.linalg.inv(S))
		
		x_final = x_est + np.dot(Kn, i)
		P_final = np.dot(j, np.dot(P_est, j.T)) + np.dot(Kn, np.dot(self._R, Kn.T))
		
		return x_final, P_final
	

	def run(self, x_init, P_init, u, y):
		x_est, P_est = self.predict(x_init, P_init, u)
		return self.correct(x_est, P_est, y)

# from Thymio import Thymio
# import time
# from ThymioFunctions import set_motors

# x_init = [0, 0, np.pi/4]
# P_init = np.eye(3)

# H = np.eye(3)
# Q = np.eye(3) * 0.0001
# R = np.eye(3) * 0.001

# kalman = extended_kalman(Q, H, R, 0.1)

# with Thymio.serial(port="COM8", refreshing_rate=0.1) as th:
# 	dir(th)

# 	time.sleep(1)

# 	variables = th.variable_description()

# 	for var in variables:
# 		print(var)
	
# 	while not th.get_var('button.center'):
# 		u = [50, 0]
# 		v = u[0]
# 		w = u[1]
# 		L = 90
# 		R = 40
# 		x_init, P_init = kalman.predict(x_init, P_init, u)
# 		print(x_init[:2])
# 		set_motors(th, v, w)
# 		time.sleep(0.1)

# 	set_motors(th)