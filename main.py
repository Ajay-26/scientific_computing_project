from solver import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process filenames')
	parser.add_argument('--grid_size',type=int, default=100)
	parser.add_argument('--velocity_function',type=str,default='simple')
	parser.add_argument('--diffusivity',type=float,default=0.1)
	parser.add_argument('--dt',type=float,default=0.1)
	parser.add_argument('--dx',type=float,default=0.1)
	parser.add_argument('--timesteps',type=int,default=100)

	args = parser.parse_args()
	n = args.grid_size
	vel_func = args.velocity_function
	diffusivity = args.diffusivity
	dt = args.dt 
	dx = args.dx
	ntime_steps = args.timesteps

	vorticity = np.zeros([n,n])

	velocity = velocity_function(np.array(list(range(n))) - n//2,np.array(list(range(n))) - n//2,vel_fun)

	vorticity = vorticity_solver(vorticity,velocity,diffusivity,dt,dx,ntime_steps)
	print(vorticity)
