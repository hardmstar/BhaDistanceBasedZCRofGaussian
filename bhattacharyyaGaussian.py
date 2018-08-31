# -*- coding:utf-8 -*-
import numpy as np
import math


def calBhaDistanceGaussian(Mi, Mj, Sigmai, Sigmaj):
	Mi = np.array(Mi)
	Mj = np.array(Mj)
	Sigmai = np.mat(Sigmai)
	Sigmaj = np.mat(Sigmaj)
	d_B_i_j = np.dot((1.0 / 8 * (Mj - Mi).T * ((Sigmai + Sigmaj) / 2).I), (Mj - Mi))
	+ 1.0 / 2 * math.log(np.linalg.det(Sigmai + Sigmaj)
	 / math.sqrt(np.linalg.det(Sigmai) * np.linalg.det(Sigmaj)))
	return d_B_i_j
