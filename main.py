from utils import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process filenames')
	parser.add_argument('--grid_size',type=int)
	parser.add_argument('--velocity_function',type=str,default='simple')

	args = parser.parse_args()
	n = args.grid_size
	vel_func = args.velocity_function

	vorticity = np.zeros([n,n])
	
