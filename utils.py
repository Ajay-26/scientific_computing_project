import os 
import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from PIL import Image
import PIL
import argparse
from multiprocessing import Pool, TimeoutError
from multiprocessing import Process

SOFTENING = 1e-5

def print_matrix(matrix, filename):
	#Matrix of shape mxn
	matrix_str = str(matrix).replace('[','')
	matrix_str = matrix_str.replace(']','')
	with open(filename,'w') as f:
		f.write(matrix_str)
	return

#Plotting functions
def get_sortkey(filename = "iteration_i.png"):
	discard_str = len('iteration_')
	rem_str = filename[discard_str:]
	int_str = int(rem_str[:-4])
	return int_str

def get_3d_plots(filename,matrix,savename = 'plot.png'):
	if matrix is None:
		with open(filename,'r') as f:
		    lines = f.readlines() 
		matrix = []
		for line in lines:
		    line = line.strip()
		    matrix.append(np.array([float(elt) for elt in line.split(' ')]).reshape(1,-1))
	plt.clf()
	n = matrix.shape[0]
	ax = plt.axes(projection='3d')
	x = np.linspace(0,n,n)
	y = np.linspace(0,n,n)
	X,Y = np.meshgrid(x,y)
	#print(z.shape)
	ax.contour3D(X,Y,matrix)
	ax.set_zlim3d(bottom=0,top=1)
	#plt.matshow(matrix)
	plt.savefig(savename)
	#plt.close()
	return	

def make_video_cv2(NT,video_name = 'video.avi'):
	image_folder = 'plots'
	video_name = video_name
	images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	images.sort(key= get_sortkey)
	print(images)
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, 0, 5, (width,height))

	for image in images:
	    video.write(cv2.imread(os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()
	return

def init_vorticity(init_type,n):
	if init_type == 'zeros':
		return np.zeros([n,n])
	elif init_type == 'ones':
		return np.ones([n,n])
	elif init_type == 'random':
		return np.random.rand(n,n)
	elif init_type == 'sinusoid':
		return np.sin(np.random.rand(n,n))

def compute_velocities(x,y,vel_func):
	if vel_func == 'simple':
		x_vel = y 
		y_vel = -x
	elif vel_func == 'square':
		x_vel = y**2
		y_vel = -x**2
	elif vel_func == 'radial':
		r = np.sqrt(x**2 + y**2)
		cos_theta = x/r 
		sin_theta = y/r
		x_vel = r*cos_theta
		y_vel = r*sin_theta
	return x_vel,y_vel