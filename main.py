from solver import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process filenames')
	parser.add_argument('--grid_size',type=int, default=100)
	parser.add_argument('--velocity_function',type=str,default='simple')
	parser.add_argument('--diffusivity',type=float,default=0.5)
	parser.add_argument('--dt',type=float,default=1e-3)
	parser.add_argument('--dx',type=float,default=0.1)
	parser.add_argument('--timesteps',type=int,default=100)
	parser.add_argument('--init_type',type=str,default='random')

	args = parser.parse_args()
	n = args.grid_size
	vel_func = args.velocity_function
	diffusivity = args.diffusivity
	dt = args.dt 
	dx = args.dx
	ntime_steps = args.timesteps
	init_type = args.init_type

	vorticity = init_vorticity(init_type,n)

	velocity = compute_velocities(np.array(list(range(n))) - n//2,np.array(list(range(n))) - n//2,vel_func)

	vorticity = vorticity_solver(vorticity,velocity,diffusivity,vel_func,dt,dx,ntime_steps)
	make_video_cv2(ntime_steps,'vorticity.avi')