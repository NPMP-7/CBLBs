from models import *
import numpy as np

def CLB_model_8_1(state, T, params):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b, rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b = params

    """
    latches
    """
    #########
    # params

    # set params for symmetric toggle switch topology
    gamma_L_Y, theta_L_Y = gamma_L_X, theta_L_X
    n_x, m_y = n_y, m_x
    eta_y, omega_y = eta_x, omega_x

    params_toggle = [delta_L, gamma_L_X, gamma_L_Y, n_x, n_y, theta_L_X, theta_L_Y, eta_x, eta_y, omega_x, omega_y, m_x,
                     m_y, delta_x, delta_y, rho_x, rho_y, r_X, r_Y]

    # degradation rates for induction of switches are specific for each toggle switch
    params_toggle_I0 = params_toggle.copy()
    params_toggle_I0[-4:-2] = rho_I0_a, rho_I0_b
    params_toggle_I1 = params_toggle.copy()
    params_toggle_I1[-4:-2] = rho_I1_a, rho_I1_b
    params_toggle_I2 = params_toggle.copy()
    params_toggle_I2[-4:-2] = rho_I2_a, rho_I2_b
    params_toggle_I3 = params_toggle.copy()
    params_toggle_I3[-4:-2] = rho_I3_a, rho_I3_b
    params_toggle_I4 = params_toggle.copy()
    params_toggle_I4[-4:-2] = rho_I4_a, rho_I4_b
    params_toggle_I5 = params_toggle.copy()
    params_toggle_I5[-4:-2] = rho_I5_a, rho_I5_b
    params_toggle_I6 = params_toggle.copy()
    params_toggle_I6[-4:-2] = rho_I6_a, rho_I6_b
    params_toggle_I7 = params_toggle.copy()
    params_toggle_I7[-4:-2] = rho_I7_a, rho_I7_b

    #########
    # states

    # latch I0
    I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b = state[:6]
    state_toggle_IO = I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b

    # latch I1
    I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b = state[6:12]
    state_toggle_I1 = I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b

    # latch I2
    I2_L_A, I2_L_B, I2_a, I2_b, I2_N_a, I2_N_b = state[12:18]
    state_toggle_I2 = I2_L_A, I2_L_B, I2_a, I2_b, I2_N_a, I2_N_b

    # latch I3
    I3_L_A, I3_L_B, I3_a, I3_b, I3_N_a, I3_N_b = state[18:24]
    state_toggle_I3 = I3_L_A, I3_L_B, I3_a, I3_b, I3_N_a, I3_N_b

    # latch I4
    I4_L_A, I4_L_B, I4_a, I4_b, I4_N_a, I4_N_b = state[24:30]
    state_toggle_I4 = I4_L_A, I4_L_B, I4_a, I4_b, I4_N_a, I4_N_b

    # latch I5
    I5_L_A, I5_L_B, I5_a, I5_b, I5_N_a, I5_N_b = state[30:36]
    state_toggle_I5 = I5_L_A, I5_L_B, I5_a, I5_b, I5_N_a, I5_N_b

    # latch I6
    I6_L_A, I6_L_B, I6_a, I6_b, I6_N_a, I6_N_b = state[36:42]
    state_toggle_I6 = I6_L_A, I6_L_B, I6_a, I6_b, I6_N_a, I6_N_b

    # latch I7
    I7_L_A, I7_L_B, I7_a, I7_b, I7_N_a, I7_N_b = state[42:48]
    state_toggle_I7 = I7_L_A, I7_L_B, I7_a, I7_b, I7_N_a, I7_N_b

    #########
    # models
    dstate_toggle_IO = toggle_model(state_toggle_IO, T, params_toggle_I0)
    dstate_toggle_I1 = toggle_model(state_toggle_I1, T, params_toggle_I1)
    dstate_toggle_I2 = toggle_model(state_toggle_I2, T, params_toggle_I2)
    dstate_toggle_I3 = toggle_model(state_toggle_I3, T, params_toggle_I3)
    dstate_toggle_I4 = toggle_model(state_toggle_I4, T, params_toggle_I4)
    dstate_toggle_I5 = toggle_model(state_toggle_I5, T, params_toggle_I5)
    dstate_toggle_I6 = toggle_model(state_toggle_I6, T, params_toggle_I6)
    dstate_toggle_I7 = toggle_model(state_toggle_I7, T, params_toggle_I7)

    dstate_toggles = np.append(np.append(np.append(np.append(np.append(
        np.append(np.append(dstate_toggle_IO, dstate_toggle_I1, axis=0), dstate_toggle_I2, axis=0), dstate_toggle_I3,
        axis=0), dstate_toggle_I4, axis=0), dstate_toggle_I5, axis=0), dstate_toggle_I6, axis=0), dstate_toggle_I7,
        axis=0)

    """
    mux
    """
    #########
    # params
    params_mux = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X

    #########
    # state
    I0, I1, I2, I3, I4, I5, I6, I7 = I0_a, I1_a, I2_a, I3_a, I4_a, I5_a, I6_a, I7_a
    state_mux = np.append([I0, I1, I2, I3, I4, I5, I6, I7], state[48:], axis=0)

    ########
    # model
    dstate_mux = MUX_8_1_model(state_mux, T, params_mux)
    dstate_mux = dstate_mux[8:]  # ignore dI0, dI1, dI2, dI3, dI4, dI5, dI6, dI7

    """
    return
    """
    dstate = np.append(dstate_toggles, dstate_mux, axis=0)
    return dstate

def MUX_8_1_model(state, T, params):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X = params
    params_yes = gamma_x, n_y, theta_x, delta_x, rho_x
    params_not = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x

    I0, I1, I2, I3, I4, I5, I6, I7, S0, S1, S2 = state[:11]
    I0_out, I1_out, I2_out, I3_out, I4_out, I5_out, I6_out, I7_out = state[11:19]
    L_I0_I0, L_I1_S0, L_I1_I1, L_I2_S1, L_I2_I2, L_I3_S0, L_I3_S1, L_I3_I3, L_I4_S2, L_I4_I4, L_I5_S0, L_I5_S2, L_I5_I5, L_I6_S1, L_I6_S2, L_I6_I6, L_I7_S0, L_I7_S1, L_I7_S2, L_I7_I7, L_I0, L_I1, L_I2, L_I3, L_I4, L_I5, L_I6, L_I7 = state[19:47]
    N_I0_S0, N_I0_S1, N_I0_S2, N_I0_I0, N_I1_S0, N_I1_S1, N_I1_S2, N_I1_I1, N_I2_S0, N_I2_S1, N_I2_S2, N_I2_I2, N_I3_S0, N_I3_S1, N_I3_S2, N_I3_I3, N_I4_S0, N_I4_S1, N_I4_S2, N_I4_I4, N_I5_S0, N_I5_S1, N_I5_S2, N_I5_I5, N_I6_S0, N_I6_S1, N_I6_S2, N_I6_I6, N_I7_S0, N_I7_S1, N_I7_S2, N_I7_I7, N_I0, N_I1, N_I2, N_I3, N_I4, N_I5, N_I6, N_I7 = state[47:87]
    out = state[87]

    m1 = [I0, I1, I2, I3, S0, S1, I0_out, I1_out, I2_out, I3_out, L_I0_I0, L_I1_S0, L_I1_I1, L_I2_S1, L_I2_I2, L_I3_S0,
          L_I3_S1, L_I3_I3, L_I0, L_I1, L_I2, L_I3, N_I0_S0, N_I0_S1, N_I0_I0, N_I1_S0, N_I1_S1, N_I1_I1, N_I2_S0,
          N_I2_S1, N_I2_I2, N_I3_S0, N_I3_S1, N_I3_I3, N_I0, N_I1, N_I2, N_I3, out]

    m2 = [I4, I5, I6, I7, S0, S1, I4_out, I5_out, I6_out, I7_out, L_I4_I4, L_I5_S0, L_I5_I5, L_I6_S1, L_I6_I6, L_I7_S0,
          L_I7_S1, L_I7_I7, L_I4, L_I5, L_I6, L_I7, N_I4_S0, N_I4_S1, N_I4_I4, N_I5_S0, N_I5_S1, N_I5_I5, N_I6_S0,
          N_I6_S1, N_I6_I6, N_I7_S0, N_I7_S1, N_I7_I7, N_I4, N_I5, N_I6, N_I7, out]

    dI0, dI1, dI2, dI3, dS0, dS1, dI0_out, dI1_out, dI2_out, dI3_out, dL_I0_I0, dL_I1_S0, dL_I1_I1, dL_I2_S1, dL_I2_I2, dL_I3_S0, dL_I3_S1, dL_I3_I3, dL_I0, dL_I1, dL_I2, dL_I3, dN_I0_S0, dN_I0_S1, dN_I0_I0, dN_I1_S0, dN_I1_S1, dN_I1_I1, dN_I2_S0, dN_I2_S1, dN_I2_I2, dN_I3_S0, dN_I3_S1, dN_I3_I3, dN_I0, dN_I1, dN_I2, dN_I3, d1out = MUX_4_1_model(m1, T, params)
    dI4, dI5, dI6, dI7, dS0, dS1, dI4_out, dI5_out, dI6_out, dI7_out, dL_I4_I4, dL_I5_S0, dL_I5_I5, dL_I6_S1, dL_I6_I6, dL_I7_S0, dL_I7_S1, dL_I7_I7, dL_I4, dL_I5, dL_I6, dL_I7, dN_I4_S0, dN_I4_S1, dN_I4_I4, dN_I5_S0, dN_I5_S1, dN_I5_I5, dN_I6_S0, dN_I6_S1, dN_I6_I6, dN_I7_S0, dN_I7_S1, dN_I7_I7, dN_I4, dN_I5, dN_I6, dN_I7, d2out = MUX_4_1_model(m2, T, params)

    dS2, dL_I4_S2, dL_I5_S2, dL_I6_S2, dL_I7_S2, dN_I0_S2, dN_I1_S2, dN_I2_S2, dN_I3_S2, dN_I4_S2, dN_I5_S2, dN_I6_S2, dN_I7_S2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # S2 acts as Enable bit on 4-1 MUX
    dout = d1out if S2==0 else d2out

    dstate = np.array([dI0, dI1, dI2, dI3, dI4, dI5, dI6, dI7, dS0, dS1, dS2,
                       dI0_out, dI1_out, dI2_out, dI3_out, dI4_out, dI5_out, dI6_out, dI7_out,
                       dL_I0_I0, dL_I1_S0, dL_I1_I1, dL_I2_S1, dL_I2_I2, dL_I3_S0, dL_I3_S1, dL_I3_I3, dL_I4_S2,
                       dL_I4_I4, dL_I5_S0, dL_I5_S2, dL_I5_I5, dL_I6_S1, dL_I6_S2, dL_I6_I6, dL_I7_S0, dL_I7_S1,
                       dL_I7_S2, dL_I7_I7,
                       dL_I0, dL_I1, dL_I2, dL_I3, dL_I4, dL_I5, dL_I6, dL_I7,
                       dN_I0_S0, dN_I0_S1, dN_I0_S2, dN_I0_I0,
                       dN_I1_S0, dN_I1_S1, dN_I1_S2, dN_I1_I1,
                       dN_I2_S0, dN_I2_S1, dN_I2_S2, dN_I2_I2,
                       dN_I3_S0, dN_I3_S1, dN_I3_S2, dN_I3_I3,
                       dN_I4_S0, dN_I4_S1, dN_I4_S2, dN_I4_I4,
                       dN_I5_S0, dN_I5_S1, dN_I5_S2, dN_I5_I5,
                       dN_I6_S0, dN_I6_S1, dN_I6_S2, dN_I6_I6,
                       dN_I7_S0, dN_I7_S1, dN_I7_S2, dN_I7_I7,
                       dN_I0, dN_I1, dN_I2, dN_I3, dN_I4, dN_I5, dN_I6, dN_I7,
                       dout])

    return dstate


def MUX_8_1_model_ODE(T, state, params):
    return MUX_8_1_model(state, T, params)


def CLB_cascade_model_ODE(T, state, params):
    return CLB_model_8_1(state, T, params)