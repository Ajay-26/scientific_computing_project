import os 
import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from PIL import Image
import PIL

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

def get_3d_plots(filename,matrix, savename = 'plot.png'):
	if matrix is None:
		with open(filename,'r') as f:
		    lines = f.readlines() 
		matrix = []
		for line in lines:
		    line = line.strip()
		    matrix.append(np.array([float(elt) for elt in line.split(' ')]).reshape(1,-1))
	plt.clf()
	ax = plt.axes(projection='3d')
	x = np.linspace(0,n,n)
	y = np.linspace(0,n,n)
	X,Y = np.meshgrid(x,y)
	z = np.concatenate(matrix,axis=0)	
	#print(z.shape)
	ax.contour3D(z,X,Y)
	plt.savefig(savename)
	print("Done saving "+ savename)	
	return	

def make_video_cv2(NT,video_name = 'video.avi'):
	image_folder = 'plots'
	video_name = video_name
	images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	images.sort(key= get_sortkey)
	print(images)
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, 0, 1, (width,height))

	for image in images:
	    video.write(cv2.imread(os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()
	return

