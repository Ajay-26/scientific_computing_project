from utils import *

def euler_shift(vorticity, psi, velocities,dt,dx,diffusivity):
	#Using simple euler timestep shift, 
	#(w_new[i,j] - w[i,j])/dt + u*(w[i+1,j] - w[i-1,j])/2*dx + v*(w[i,j+1] - w[i,j-1])/2*dy = diffusivity*((w[i+1,j] + w[i-1,j] - 2*w[i,j])/dx*dx + (w[i,j+1] + w[i,j-1] - 2*w[i,j])/dy*dy)
	new_vorticity = vorticity.copy()
	m,n = vorticity.shape[0],vorticity.shape[1]
	#Try to reformulate using numpy
	for i in range(m):
		for j in range(n):
			new_vorticity[i,j] = diffusivity*((w[i+1,j] + w[i-1,j] - 2*w[i,j])/(dx*dx) + (w[i,j+1] + w[i,j-1] - 2*w[i,j])/(dx*dx)) - u*(w[i+1,j] - w[i-1,j])/2*dx - v*(w[i,j+1] - w[i,j-1])/(2*dx)
	return new_vorticity

def vorticity_solver(vorticity,velocity,diffusivity,vel_func,dt,dx,ntime_steps,L=1):
	#Vorticity is a matrix of problem dimension size
	#velocity can be a function of the coordinates, or even constant
	#diffusivity is the coefficient of "damping"
	#dt is the time steps
	#ntime_steps is the number of time steps
	#L is the physical size of the problem
	
	for timestep in range(ntime_steps):
		#Let the problem grid be nxn which is of length L
		n = vorticity.shape[0]
		grid_range = np.array(zip(list(range(n)),list(range(n))))
		velocities = compute_velocities(grid_range,vel_func)
		#new_vorticity = vorticity.copy()

		#equation is del^2(psi) = vorticity - compute psi using fft
		#Reference = https://www.youtube.com/watch?v=hDeARtZdq-U
		vort_f = np.fft.fft(vorticity)
		kappa = np.fft.fftfreq(n,L/n)
		psi_f = vort_f/kappa

		#psi is the term which determines incompressibility
		psi = np.fft.ifft(psi_f)

		#compute new_vorticity using psi using a simple Euler time shift
		vorticity = euler_shift(vorticity,psi,velocities,dt,dx,diffusivity)
	return vorticity