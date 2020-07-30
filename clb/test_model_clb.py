from scipy.integrate import ode
import matplotlib.pyplot as plt

from models import *
from parameters import *


rho_x = 0
rho_y = 0

rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 5, 5, 0, 5, 0, 5, 0

params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
         rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b)


Y0 = np.zeros(59)


# number of cells: toggle switches
N_I0 = np.array([1,1])
N_I1 = np.array([1,1])
N_I2 = np.array([1,1])
N_I3 = np.array([1,1])

Y0[4:6] = N_I0
Y0[10:12] = N_I1
Y0[16:18] = N_I2
Y0[22:24] = N_I3

# number of cells: mux
#Y0[22-4+24:38-4+24] = 1 # number of cells
Y0[42:58] = 1 # number of cells

# S0, S1
S = np.array([0, 0])
Y0[24:26] = S

"""
simulations
"""

# simulation parameters
t_end = 1500
N = t_end

# initialization

T = np.linspace(0, t_end, N)

t1 = t_end
dt = t_end/N
T = np.arange(0,t1+dt,dt)
Y = np.zeros([1+N,59])
Y[0,:] = Y0


# simulation
r = ode(CLB_model_ODE).set_integrator('zvode', method='bdf')
r.set_initial_value(Y0, T[0]).set_f_params(params)

i = 1
while r.successful() and r.t < t1:
    Y[i,:] = r.integrate(r.t+dt)
    i += 1



I0_a, I0_b = Y[:,2], Y[:,3]
I1_a, I1_b = Y[:,8], Y[:,9]
I2_a, I2_b = Y[:,14], Y[:,15]
I3_a, I3_b = Y[:,20], Y[:,21]

out = Y[:,-1]

# plot

ax1 = plt.subplot(241)
ax1.plot(T, I0_a)
ax1.plot(T, I0_b)
ax1.legend(["I0_a = I0", "I0_b"])
ax1.set_title('I0 toggle')

ax2 = plt.subplot(242)
ax2.plot(T, I1_a)
ax2.plot(T, I1_b)
ax2.legend(["I1_a = I1", "I1_b"])
ax2.set_title('I1 toggle')

ax3 = plt.subplot(243)
ax3.plot(T, I2_a)
ax3.plot(T, I2_b)
ax3.legend(["I2_a = I2", "I2_b"])
ax3.set_title('I2 toggle')

ax4 = plt.subplot(244)
ax4.plot(T, I3_a)
ax4.plot(T, I3_b)
ax4.legend(["I3_a = I3", "I3_b"])
ax4.set_title('I3 toggle')

ax5 = plt.subplot(212)
ax5.plot(T,out)
ax5.set_title('out')

plt.suptitle(f"S = [{S[1]},{S[0]}]")
plt.show()
