#! /usr/bin/env python3

import time
import serial
import sys
import os
import threading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from threading import Timer
import datetime as dt
import numpy as np
import scipy as sp
from scipy.stats import norm 
from scipy.stats import multivariate_normal
from numpy.linalg import inv
from matplotlib.animation import FuncAnimation

current_directory = os.getcwd()
relative_paths = [
    'imu_data/imu_data_accel_x.txt',
    'imu_data/imu_data_accel_y.txt',
    'imu_data/imu_data_accel_z.txt',
    'imu_data/imu_data_gyro_x.txt',
    'imu_data/imu_data_gyro_y.txt',
    'imu_data/imu_data_gyro_z.txt'
]

relative_path1 = 'imu_data/imu_data_accel_x.txt'
relative_path2 = 'imu_data/imu_data_accel_y.txt'
relative_path3 = 'imu_data/imu_data_accel_z.txt'
relative_path4 = 'imu_data/imu_data_gyro_x.txt'
relative_path5 ='imu_data/imu_data_gyro_y.txt'
relative_path6 ='imu_data/imu_data_gyro_z.txt'

absolute_path1 = os.path.join(current_directory, relative_path1)
absolute_path2 = os.path.join(current_directory, relative_path1)
absolute_path3 = os.path.join(current_directory, relative_path1)
absolute_path4 = os.path.join(current_directory, relative_path1)
absolute_path5 = os.path.join(current_directory, relative_path1)
absolute_path6 = os.path.join(current_directory, relative_path1)

file_path = [absolute_path1,absolute_path2,absolute_path3,absolute_path4,absolute_path5,absolute_path6]

# # 파일을 열어서 모든 줄을 읽고, 각 줄을 float로 변환하여 배열에 저장합니다.
with open(file_path[0], 'r') as file:
    accel_x = [float(line.strip()) for line in file.readlines()]
with open(file_path[1], 'r') as file:
    accel_y = [float(line.strip()) for line in file.readlines()]
with open(file_path[2], 'r') as file:
    accel_z = [float(line.strip()) for line in file.readlines()]
with open(file_path[3], 'r') as file:
    gyro_x = [float(line.strip()) for line in file.readlines()]
with open(file_path[4], 'r') as file:
    gyro_y = [float(line.strip()) for line in file.readlines()]
with open(file_path[5], 'r') as file:
    gyro_z = [float(line.strip()) for line in file.readlines()]









# Initialization for estimation.
x_0 = np.array([1, 0, 0, 0])  # (q0, q1, q2, q3) by my definition.
P_0 = np.eye(4)
n_samples = 300
dt = 0.02
A = None
H = np.eye(4)
Q = 0.0001 * np.eye(4)
R = 10 * np.eye(4)

time = np.arange(n_samples) * dt
phi_esti_save = np.zeros(n_samples)
the_esti_save = np.zeros(n_samples)
psi_esti_save = np.zeros(n_samples)

is_paused= True
x_esti = None
P_predic = None
x_pred = None
K_vals = None
K_vals_p = None



def kalman_filter(z_meas, x_esti, P,A,H,R,Q): 
    """Kalman Filter Algorithm."""
    # (1) Prediction.
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q
    # (2) Kalman Gain.
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred

    return x_esti, P , P_pred , x_pred , K

def on_press(event):
    global is_paused , current_frame
    if event.key == ' ':
        is_paused = not is_paused
            


def get_gyro(i):
    """Measure angular velocity using gyro."""
    p = gyro_x[i] 
    q = gyro_y[i]  
    r = gyro_z[i]  
    return p, q, r

def get_accel(i):
    """Measure angular acceleration using accelerometer (G-meter)."""
    ax = accel_x[i] 
    ay = accel_y[i] 
    az = accel_z[i] 
    return ax, ay, az

def accel2euler(ax, ay, az, phi, the, psi):
    """Calculate Euler angle (Pose Orientation)."""
    g = 9.8  # 9.8 [m/s^2]
    cosThe = np.cos(the)
    phi = np.arcsin(-ay / (g * cosThe))
    the = np.arcsin(ax / g)
    psi = psi
    return phi, the, psi

def quaternion2euler(q):
    """Quaternion to Euler angle for drawing."""
    phi_esti = np.arctan2(2 * (q[2]*q[3] + q[0]*q[1]), 1 - 2 * (q[1]**2 + q[2]**2))
    the_esti = -np.arcsin(2 * (q[1]*q[3] - q[0]*q[2]))
    psi_esti = np.arctan2(2 * (q[1]*q[2] + q[0]*q[3]), 1 - 2 * (q[2]**2 + q[3]**2))
    return phi_esti, the_esti, psi_esti

def euler2quaternion(phi, the, psi):
    """Euler angle to Quaternion for state variables."""
    sinPhi = np.sin(phi/2)
    cosPhi = np.cos(phi/2)
    sinThe = np.sin(the/2)
    cosThe = np.cos(the/2)
    sinPsi = np.sin(psi/2)
    cosPsi = np.cos(psi/2)

    q = np.array([cosPhi * cosThe * cosPsi + sinPhi * sinThe * sinPsi,
                  sinPhi * cosThe * cosPsi - cosPhi * sinThe * sinPsi,
                  cosPhi * sinThe * cosPsi + sinPhi * cosThe * sinPsi,
                  cosPhi * cosThe * sinPsi - sinPhi * sinThe * cosPsi])
    return q

def main(args=None):

    print(len(accel_x))
    print(len(gyro_x))
    phi, the, psi = 0, 0, 0
    x_esti, P = None, None
    for i in range(n_samples):
        p, q, r = get_gyro(i)
        A = np.eye(4) + dt / 2 * np.array([[0, -p, -q, -r],
                                        [p,  0,  r, -q],
                                        [q, -r,  0,  p],
                                        [r,  q, -p,  0]])  
        ax, ay, az = get_accel(i)
        phi, the, psi = accel2euler(ax, ay, az, phi, the, psi)
        z_meas = euler2quaternion(phi, the, psi)

        if i == 0:
            x_esti, P = x_0, P_0
        else:
            x_esti, P, P_predic , x_pred , K_i= kalman_filter(z_meas, x_esti, P , A, H, R, Q)
        phi_esti, the_esti, psi_esti = quaternion2euler(x_esti)

        phi_esti_save[i] = np.rad2deg(phi_esti)
        the_esti_save[i] = np.rad2deg(the_esti)
        psi_esti_save[i] = np.rad2deg(psi_esti)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

    plt.subplot(3, 1, 1)
    plt.plot(time, phi_esti_save, 'r', label='Roll ($\\phi$): Estimation (KF)', markersize=0.2)
    plt.legend(loc='lower right')
    plt.title('Roll ($\\phi$): Estimation (KF)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Roll ($\phi$) angle [deg]')

    plt.subplot(3, 1, 2)
    plt.plot(time, the_esti_save, 'b', label='Pitch ($\\theta$): Estimation (KF)', markersize=0.2)
    plt.legend(loc='lower right')
    plt.title('Pitch ($\\theta$): Estimation (KF)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Pitch ($\\theta$) angle [deg]')

    plt.subplot(3, 1, 3)
    plt.plot(time, psi_esti_save, 'g', label='Yaw ($\\psi$): Estimation (KF)', markersize=0.2)
    plt.legend(loc='lower right')
    plt.title('Yaw ($\\psi$): Estimation (KF)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Yaw ($\\psi$) angle [deg]')
    plt.show()











if __name__ == '__main__':

    main()