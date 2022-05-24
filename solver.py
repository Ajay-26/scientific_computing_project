from utils import *

SOFTENING = 1.0

def euler_shift(vorticity, psi, velocities,dt,dx,diffusivity):
	#Using simple euler timestep shift, 
	#(w_new[i,j] - w[i,j])/dt + u*(w[i+1,j] - w[i-1,j])/2*dx + v*(w[i,j+1] - w[i,j-1])/2*dy = diffusivity*((w[i+1,j] + w[i-1,j] - 2*w[i,j])/dx*dx + (w[i,j+1] + w[i,j-1] - 2*w[i,j])/dy*dy)
	new_vorticity = vorticity.copy()
	m,n = vorticity.shape[0],vorticity.shape[1]
	#Try to reformulate using numpy
	#Boundary conditions are periodic => wrap around
	for i in range(m):
		for j in range(n):
			u = velocities[0][i]
			v = velocities[1][j]
			new_vorticity[i,j] = vorticity[i,j] + dt*(diffusivity*((vorticity[(i+1)%m,j] + vorticity[i-1,j] - 2*vorticity[i,j])/(dx*dx) + (vorticity[i,(j+1)%n] + vorticity[i,j-1] - 2*vorticity[i,j])/(dx*dx)) - u*(psi[(i+1)%m,j] - psi[i-1,j])/2*dx - v*(psi[i,(j+1)%n] - psi[i,j-1])/(2*dx))
	return new_vorticity

def euler_shift_parallel(tid,nt,final_vort,vorticity,psi,velocities,dt,dx,diffusivity):
	#Using simple euler timestep shift, 
	#(w_new[i,j] - w[i,j])/dt + u*(w[i+1,j] - w[i-1,j])/2*dx + v*(w[i,j+1] - w[i,j-1])/2*dy = diffusivity*((w[i+1,j] + w[i-1,j] - 2*w[i,j])/dx*dx + (w[i,j+1] + w[i,j-1] - 2*w[i,j])/dy*dy)
	new_vorticity = np.zeros_like(final_vort)
	m,n = vorticity.shape[0],vorticity.shape[1]
	mt,nt = vorticity.shape[0]//nt,vorticity.shape[1]//nt
	#Try to reformulate using numpy
	#Boundary conditions are periodic => wrap around
	for i in range(tid*mt,(tid+1)*mt):
		for j in range(tid*nt,(tid+1)*nt):
			u = velocities[0][i]
			v = velocities[1][j]
			new_vorticity[i,j] = vorticity[i,j] + dt*(diffusivity*((vorticity[(i+1)%m,j] + vorticity[i-1,j] - 2*vorticity[i,j])/(dx*dx) + (vorticity[i,(j+1)%n] + vorticity[i,j-1] - 2*vorticity[i,j])/(dx*dx)) - u*(psi[(i+1)%m,j] - psi[i-1,j])/2*dx - v*(psi[i,(j+1)%n] - psi[i,j-1])/(2*dx))
	final_vort = final_vort + new_vorticity

def call_parallel_shift_func(vorticity,psi,velocities,dt,dx,diffusivity,nt):
	p_list = []
	final_vorticity = np.zeros_like(vorticity)
	for tid in range(nt):
		p = Process(target = euler_shift_parallel, args = (tid,nt,final_vorticity,vorticity,psi,velocities,dt,dx,diffusivity))
		p.start()
		p_list.append(p)
	for p in p_list:
		p.join()
	return final_vorticity

def vorticity_solver(vorticity,velocity,diffusivity,vel_func,dt,dx,ntime_steps,L=1):
	#Vorticity is a matrix of problem dimension size
	#velocity can be a function of the coordinates, or even constant
	#diffusivity is the coefficient of "damping"
	#dt is the time steps
	#ntime_steps is the number of time steps
	#L is the physical size of the problem
	
	n = vorticity.shape[0]	
	for timestep in range(ntime_steps):
		fname = "plots/iteration_{_i}.png".format(_i=timestep)
		get_3d_plots(None,vorticity,fname)
		print("Iteration: ",timestep)
		#Let the problem grid be nxn which is of length L
		
		#equation is del^2(psi) = vorticity - compute psi using fft
		#Reference = https://www.youtube.com/watch?v=hDeARtZdq-U
		vort_f = np.real(np.fft.fft2(vorticity))
		kappa = np.pi*2*np.fft.fftfreq(n,dx) + SOFTENING
		psi_f = vort_f/kappa**2
		#psi is the term which determines incompressibility
		psi = np.real(np.fft.ifft2(psi_f))

		#compute new_vorticity using psi using a simple Euler time shift
		nt = 4
		vorticity = call_parallel_shift_func(vorticity,psi,velocity,dt,dx,diffusivity,nt)
	return vorticity